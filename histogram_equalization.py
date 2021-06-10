"""
9. Attēla līmeņu transformācijas, histogrammas operācijas
9.b) Izstrādāt datorprogrammu, kas realizē histogrammas vienmērīgošanu.

Implementation of cv.equalizeHist(), see https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


IMAGE_PATH = 'grayscale.jpg'
PIXEL_RANGE = 256


def histogram(channel):
    hist = np.zeros(PIXEL_RANGE, dtype=np.uint8)
    for pixel in channel.flatten():
        hist[pixel] += 1
    return hist


def equalize(hist):
    hist_cm = np.cumsum(hist)
    min_val, max_val = hist_cm.min(), hist_cm.max()
    # normalize
    norm = (hist_cm - min_val) / (max_val - min_val) * 255
    return norm.astype('int')


def subplot(fig, shape, idx, img, text):
    ax = fig.add_subplot(*shape, idx)
    ax.axis("off")
    ax.title.set_text(text)
    ax.imshow(img)


def main():
    # read image
    image = cv.imread(IMAGE_PATH)
    # configure plot
    fig = plt.figure()
    # equalize
    image_equalized = []
    for ch_idx in range(image.shape[-1]):
        channel = image[:, :, ch_idx]
        hist = histogram(channel)
        hist_equalized = equalize(hist)
        channel_equalized = hist_equalized[channel]
        channel_equalized = np.reshape(a=channel_equalized, newshape=channel.shape)
        image_equalized.append(channel_equalized)
        # subplot(axs[ch_idx][1], data=hist, data_type='hist', text='Original')
        # subplot(ax=, idx=1, data=hist, type='', text='Original')
    image_equalized = np.dstack(tup=tuple(image_equalized))
    subplot(fig=fig, shape=(1, 2), idx=1, img=image, text='Original')
    subplot(fig=fig, shape=(1, 2), idx=2, img=image_equalized, text='Equalized')
    plt.show()


if __name__ == '__main__':
    main()
