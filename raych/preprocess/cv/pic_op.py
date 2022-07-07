import numpy as np
from PIL import Image


def crop_square(image):
    """
    截取图片中心正方形区域

    image: Image.open(...) 的对象
    """
    width, height = image.size

    # Crop the image, centered
    new_width = min(width,height)
    new_height = new_width
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    return image.crop((left, top, right, bottom))


def add_noise(image, ratio, num_noise):
    "随机隐藏小块区域图像"
    a2 = image.copy()
    rows = a2.shape[0]
    cols = a2.shape[1]
    s = int(min(rows,cols)/ratio) # size of spot is 1/ratio of smallest dimension

    for i in range(num_noise):
        x = np.random.randint(cols-s)
        y = np.random.randint(rows-s)
        a2[y:(y+s), x:(x+s)] = 0

    return a2


def standardize(image):
    "将image转为 [width, height, channel] 格式数据"
    rgbimg = Image.new("RGB", image.size)
    rgbimg.paste(image)
    return rgbimg


def scale(img, scale_width, scale_height):
    # Scale the image
    img = img.resize((scale_width, scale_height), Image.ANTIALIAS)
    return img