"""
10.a) Izstrādāt datorprogrammu, kas realizē izpludināšanas un asināšanas filtrus pie dažādu izmēru konvolūcijas maskām.
Gaussian blur and Sharpen kernel implemented

blur kernel
        1 2 1
1/16 *  1 4 1
        1 2 1

sharpen kernel
 0 -1  0
-1  5 -1
 0 -1  0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = 'grayscale.jpg'


def subplot(fig, shape, idx, img, text):
    ax = fig.add_subplot(*shape, idx)
    ax.axis("off")
    ax.title.set_text(text)
    ax.imshow(img, cmap='gray')


def min_max(arr):
    min_val, max_val = arr.min(), arr.max()
    return (arr - min_val) / (max_val - min_val)


def sharpen_kernel(shape, scale):
    # it is recommended that shape is of odd shapes (e.x. (3,3), (9,9) (3,5)...)
    # (3,3) -> e.x. np.array([[-1, -1, -1], [-1, -9, -1], [-1, -1, -1]])
    kernel = np.ones(shape)
    kernel *= -1
    H, W = shape
    kernel[H // 2, W // 2] = scale
    return kernel


def gaussian_kernel(shape, sigma=1.):
    H, W = shape
    lin_H = np.linspace(start=-(H - 1) / 2., stop=(H - 1) / 2., num=H)
    lin_W = np.linspace(start=-(W - 1) / 2., stop=(W - 1) / 2., num=W)
    xmesh, ymesh = np.meshgrid(lin_H, lin_W)
    kernel = np.exp(-0.5 * (np.square(xmesh) + np.square(ymesh)) / np.square(sigma))
    return kernel / np.sum(kernel)


def conv_2d(image, kernel, pad, stride):
    inH, inW, inC = image.shape
    kH, kW = kernel.shape

    outH = int(((inH + 2 * pad - kH) / stride) + 1)
    outW = int(((inW + 2 * pad - kW) / stride) + 1)
    outC = inC
    out = np.zeros((outH, outW, outC))

    # add padding to input image
    if pad != 0:
        image_pad = np.zeros((inH + pad * 2, inW + pad * 2, inC))
        image_pad[int(pad):int(-1 * pad), int(pad):int(-1 * pad)] = image.copy()
        image = image_pad

    # convolution
    # range(start, stop, step)
    for i in range(0, inH + 2 * pad - kH + 1, stride):
        for j in range(0, inW + 2 * pad - kW + 1, stride):
            for ch in range(inC):
                out[i, j, ch] = (kernel * image[i: i + kH, j: j + kW, ch]).sum()
    # if out is < 0 -> 0, if > 255 -> 255
    out = np.clip(out, 0, 255)
    return out.astype(np.uint8)


def main():
    image = cv2.imread(IMAGE_PATH)
    # padding = kernel // 2 + 1 - to make 'same_padding'

    # SHARPEN
    # try different combinations,
    # the best is odd square with scale=shape_1 x shape_2, e.x. (3,3) scale 9 or (5,5) scale 25 ...
    # skernel = sharpen_kernel(shape=(3, 3), scale=9)
    # skernel = sharpen_kernel(shape=(11, 6), scale=66)
    skernel = sharpen_kernel(shape=(5, 5), scale=25)
    out = conv_2d(image=image, kernel=skernel, pad=2, stride=1)

    # configure plot
    fig = plt.figure()
    subplot(fig=fig, shape=(1, 2), idx=1, img=image, text='Original')
    subplot(fig=fig, shape=(1, 2), idx=2, img=out, text=f'Sharpen {skernel.shape}')
    plt.show()

    # BLUR
    gkernel = gaussian_kernel(shape=(3, 3))
    out = conv_2d(image=image, kernel=gkernel, pad=0, stride=1)
    # configure plot
    fig = plt.figure()
    subplot(fig=fig, shape=(1, 2), idx=1, img=image, text='Original')
    subplot(fig=fig, shape=(1, 2), idx=2, img=out, text=f'Blur {gkernel.shape}')
    plt.show()


if __name__ == '__main__':
    main()
