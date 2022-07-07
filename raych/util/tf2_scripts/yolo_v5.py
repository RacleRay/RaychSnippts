"""
https://github.com/ultralytics/yolov5


git clone https://github.com/ultralytics/yolov5
cd ./yolov5
pip install -qr requirements.txt
"""

import sys
sys.path.append(str("/path/yolov5"))

from yolov5 import utils
display = utils.notebook_init()

import sys
from PIL import Image

import torch
import numpy as np

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow,
                           check_requirements, colorstr, increment_path,
                           non_max_suppression, print_args, scale_coords,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# ============================================
img = Image.open('./test.jpg')

device = select_device('')
weights = '/path/yolov5/yolov5s.pt'
imgsz = [img.height, img.width]
original_size = imgsz
model = DetectMultiBackend(weights, device=device, dnn=False)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size
print(f"Original size: {original_size}")
print(f"YOLO input size: {imgsz}")

# ============================================
# https://stackoverflow.com/questions/50657449/convert-image-to-proper-dimension-pytorch
img2 = img.resize([imgsz[1], imgsz[0]], Image.ANTIALIAS)

half = False
img_raw = torch.from_numpy(np.asarray(img2)).to(device)
img_raw = img_raw.half() if half else img_raw.float()  # uint8 to fp16/32
img_raw /= 255  # 0 - 255 to 0.0 - 1.0
img_raw = img_raw.unsqueeze_(0)
img_raw = img_raw.permute(0, 3, 1, 2)
# print(img_raw.shape)

# ============================================
conf_thres = 0.25  # confidence threshold
iou_thres = 0.45  # NMS IOU threshold
classes = None
agnostic_nms = False,  # class-agnostic NMS
max_det = 1000

model.warmup(imgsz=(1, 3, *imgsz))  # warmup
dt, seen = [0.0, 0.0, 0.0], 0

pred = model(img_raw, augment=False, visualize=False)
pred = non_max_suppression(pred,
                           conf_thres,
                           iou_thres,
                           classes,
                           agnostic_nms,
                           max_det=max_det)

# ============================================
# convert these raw predictions into the bounding boxes, labels, and confidences
results = []
for i, det in enumerate(pred):  # per image
    gn = torch.tensor(img_raw.shape)[[1, 0, 1, 0]]

    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(original_size, det[:, :4], imgsz).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                    gn).view(-1).tolist()
            # Choose between xyxy and xywh as your desired format.
            results.append([names[int(cls)], float(conf), [*xyxy]])


# from PIL import Image, ImageDraw

# img3 = img.copy()
# draw = ImageDraw.Draw(img3)

# for itm in results:
#     b = itm[2]
#     # print(b)
#     draw.rectangle(b)
