import cv2
import torch
import numpy as np
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClsDa:
    def __init__(self,
                 filepaths, 
                 labels, 
                 augopts=None, 
                 backend="pil", 
                 channel_first=True, 
                 grayscale=False):
        """
        Arguments:
            filepaths {Iterable} -- 图片路径列表
            labels {Iterable} -- 图片对应标签

        Keyword Arguments:
            augopts {Callable} -- 数据增强函数 (default: {None})
            backend {str} -- cv2 或者 pil (default: {"pil"})
            channel_first {bool} -- 彩色图片设置 (default: {True})
            grayscale {bool} -- 灰度化 (default: {False})
        """
        labels, augopts=None, backend="pil", channel_first=True, grayscale=False):
        self.filepaths = filepaths
        self.labels = labels
        self.augopts = augopts
        self.backend = backend
        self.channel_first = channel_first
        self.grayscale = grayscale

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        if self.backend == "pil":
            image = Image.open(self.image_paths[idx])
            image = np.array(image)
        elif self.backend == "cv2":
            if self.grayscale is False:
                image = cv2.imread(self.image_paths[idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.augopts is not None:
            augmented = self.augopts(image=image)
            image = augmented["image"]

        if self.channel_first is True and self.grayscale is False:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image_tensor = torch.tensor(image)
        if self.grayscale:
            image_tensor = image_tensor.unsqueeze(0)
        labels = torch.tensor(self.labels[idx])

        return {
            "image": image_tensor,
            "labels": labels,
        }
