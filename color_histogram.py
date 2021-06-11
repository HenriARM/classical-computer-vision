"""
Given image, visualize color value of each pixel as a histogram
Implementation is channel free, so as an input it can be colorful or grayscale image
Additionally, we give to each 3D Bar its pixel color
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('./rgb.jpg')
# img = cv.imread('./grayscale.jpg')
# resize image to process faster all pixels
img = cv.resize(img, (20, 40))
print(img.shape)

# save resized image
# cv.imwrite('tmp.jpg', img)

# loop through each channel
fig = plt.figure()
chs = img.shape[2]

# check for grayscale, R = G = B (.all need to comape all array values)
if (img[:, :, 0] == img[:, :, 1]).all() and (img[:, :, 1] == img[:, :, 2]).all():
    chs = 1

for ch in range(chs):
    # subplot positions: nrows, ncols, index
    ax = fig.add_subplot(1, chs, ch + 1, projection='3d')

    # take one channel of image
    channel = img[:, :, ch]
    # create coordinates for all points
    y, x = np.meshgrid(range(channel.shape[0]), range(channel.shape[1]))

    # color
    if chs == 1:
        # grayscale (copy all channels)
        colors = np.copy(img)
    else:
        # rgb (copy only one channel)
        colors = np.zeros(img.shape)
        colors[:, :, ch] = img[:, :, ch]
        # convert to rgba
    colors = colors / 255

    # draw histogram using bar3d
    z_data = channel.flatten()
    ax.bar3d(x=x.flatten(), y=y.flatten(), z=np.zeros(len(z_data)), dx=1, dy=1, dz=z_data, color=colors.reshape(-1, 3))

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Color')

plt.show()
