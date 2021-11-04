import torch
import torch.nn as nn
from raych.layers.embedding.embed import EmbeddingLayer
from raych.layers.encoder.general import TextCnnEncoder, BiLSTMEncoder
from raych.layers.loss.loss import FocalLoss
from raych.layers.aggregator.pooler import (
    SelfAttnAggregator, DynamicRoutingAggregator,
    MaxPoolerAggregator, AvgPoolerAggregator
)
from raych.layers.classifier.classify_head import Classifier


class AggregatorLayer(nn.Module):
    def __init__(self, d_model, device, aggregator_name=None):
        super(AggregatorLayer, self).__init__()
        self.aggregator_op_name = aggregator_name
        self.d_model = d_model

        self.aggregator_op = None
        if self.aggregator_op_name == "slf_attn_pooler":
            attn_vector = nn.Linear(self.d_model, 1)
            self.aggregator_op = SelfAttnAggregator(
                self.d_model,
                attn_vector=attn_vector,
            )
        elif self.aggregator_op_name == "dr_pooler":
            cap_num_ = 4   # capsule 大小
            iter_num_ = 3  # 迭代次数
            shared_fc_ = nn.Linear(
                self.d_model,
                self.d_model
            )
            self.aggregator_op = DynamicRoutingAggregator(
                self.d_model,
                cap_num_,
                int(self.d_model / cap_num_),
                iter_num_,
                shared_fc=shared_fc_,
                device=device
            )
        elif self.aggregator_op_name == "max_pooler":
            self.aggregator_op = MaxPoolerAggregator()
        else:
            self.aggregator_op = AvgPoolerAggregator()

    def forward(self, input_tensors, mask=None):
        output = self.aggregator_op(input_tensors, mask)
        return output


class ClassificationModel(nn.Module):
    def __init__(self, encoder, embed_dim, hidden_dim, agg_name,
                 max_seq_len, device, label_list, class_weights, 
                 use_focal_loss, focal_loss_gamma=1.0):
        super(ClassificationModel, self).__init__()

        self.num_labels = len(label_list)
        self.use_focal_loss = use_focal_loss
        self.focal_loss_gamma = focal_loss_gamma

        # embedding 层
        self.embeddings = EmbeddingLayer(embed_dim, max_seq_len, w2v_file="")

        # encoder 层
        if encoder == "textcnn":
            self.encoder = TextCnnEncoder(
                embed_dim, hidden_dim, dropout_rate=0.2, norm=False
            )
        elif encoder == "lstm":
            self.encoder = BiLSTMEncoder(
                embed_dim, hidden_dim, dropout_rate=0.2, norm=False
            )
        else:
            raise ValueError("un-supported encoder type: {}".format(encoder))

        # aggregator 层
        self.aggregator = AggregatorLayer(hidden_dim, device, agg_name)

        # 分类层
        self.classifier = Classifier(dropout_rate=0.2,
                                     input_dim=hidden_dim,
                                     num_labels=self.num_labels)

        # class weights
        self.class_weights = None
        if class_weights:
            self.class_weights = class_weights.split(",")
        else:
            self.class_weights = [1] * self.num_labels
        self.class_weights = [float(w) for w in self.class_weights]
        self.class_weights = torch.FloatTensor(self.class_weights).to(device)

    def forward(self, input_ids=None,
                attention_mask=None,
                label_ids,
                **kwargs):
        input_tensors = self.embeddings(input_ids)
        output_tensors = self.encoder(input_tensors)
        pooled_outputs = self.aggregator(output_tensors, mask=attention_mask)
        logits = self.classifier(pooled_outputs)

        outputs = (logits, )

        if label_ids is not None:
            if self.use_focal_loss:
                loss_fct = FocalLoss(
                    self.num_labels,
                    gamma=self.focal_loss_gamma,
                    weights=self.class_weights,
                    reduction='mean'
                )
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))

            outputs = (loss, ) + outputs

        return outputs