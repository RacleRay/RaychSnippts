from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np


def visualize_generator(img_file, gen):
    # Load the requested image
    img = load_img(img_file)
    data = img_to_array(img)
    samples = expand_dims(data, 0)

    # Generat augumentations from the generator
    it = gen.flow(samples, batch_size=1)
    images = []
    for i in range(4):
        batch = it.next()
        image = batch[0].astype('uint8')
        images.append(image)

    images = np.array(images)

    # Create a grid of 4 images from the generator
    index, height, width, channels = images.shape
    nrows = index // 2

    grid = (images.reshape(nrows, 2, height, width, channels)
                  .swapaxes(1, 2)
                  .reshape(height * nrows, width * 2, 3))

    # fig = plt.figure(figsize=(15., 15.))
    plt.figure(figsize=(15., 15.))
    plt.axis('off')
    plt.imshow(grid)
