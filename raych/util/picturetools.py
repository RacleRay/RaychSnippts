import os
import numpy as np
from PIL import Image


def check_wh(imgPath, show_static=True):
    w_list = []
    h_list = []
    hw_ratio = []
    for root, dirs, files in os.walk(imgPath):
        for file in files:
            img = Image.open(root + '\\' + file)
            w = img.width
            h = img.height
            hw = h / w
            w_list.append(w)
            h_list.append(h)
            hw_ratio.append(hw)
        # print(h_list)
        # print(w_list)
        # print(hw_ratio)
        if show_static:
            print(np.mean(h_list), np.mean(w_list), np.mean(hw_ratio))

    return h_list, w_list, hw_ratio