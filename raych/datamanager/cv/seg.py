import cv2
import numpy as np
import torch


class RCNNSegDa:
    def __init__(self,
                filepaths,
                bounding_boxes,
                classes=None,
                augopts=None,
                torchvision_format=True):
        """
        Arguments:
            filepaths {Iterable} -- 图片路径列表
            bounding_boxes {Iterable} -- 图片对应bounding box标记

        Keyword Arguments:
            classes {List} -- bounding box对应类别，可选 (default: {None})
            augopts {Callable} -- 数据增强函数 (default: {None})
            torchvision_format {bool} -- 为True，返回元组 （image，targets），为False返回字典
                有keys：｛"image"， "boxes", "area", "iscrowd", "labels"｝ (default: {True})
        """
        self.filepaths = filepaths
        self.bounding_boxes = bounding_boxes
        self.augopts = augopts
        self.torchvision_format = torchvision_format
        self.classes = classes

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, item):
        image = cv2.imread(self.filepaths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        bboxes = self.bounding_boxes[item]
        
        if self.augopts is not None:
            augmented = self.augopts(image=image, bboxes=bboxes)
            image = augmented["image"]
            bboxes = augmented["bboxes"]

        # channel first
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        bboxes = np.array(bboxes)

        # box area
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        # 多目标，还是不区分多目标
        if self.classes is None:
            labels = torch.ones((bboxes.shape[0],), dtype=torch.int64)
        else:
            labels = torch.tensor(self.classes[item], dtype=torch.int64)
        # 多检测框
        is_crowd = torch.zeros((bboxes.shape[0],), dtype=torch.int64)
        
        target = {
            "boxes": torch.as_tensor(bboxes.tolist(), dtype=torch.float32),
            "area": torch.as_tensor(area.tolist(), dtype=torch.float32),
            "iscrowd": is_crowd,
            "labels": labels,
        }

        if self.torchvision_format:
            return torch.tensor(image, dtype=torch.float), target
        else:
            target["image"] = torch.tensor(image, dtype=torch.float)
            return target
