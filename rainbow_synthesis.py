"""
7.2a) Izstrādāt datorprogrammu, kas sintezē attēlu, kurš satur maksimāli reālistisku krāsu spektra attainojumu,
piemēram, varavīksni vai gaismas laušanu stikla prizmā.
Formulas are described here https://www.fourmilab.ch/documents/specrend/
"""

import numpy as np
import cv2 as cv

# SYNTHESIZED IMAGE PARAMETERS
SHORTEST_WAVE = 380  # nm
LONGEST_WAVE = 780  # nm

IMAGE_H = 500
BAR_W = 2  # px, each wave's column size
IMAGE_W = (LONGEST_WAVE - SHORTEST_WAVE) * BAR_W
IMAGE_C = 3
IMAGE_PATH = "rainbow.png"

# CONSTANTS
GAMMA = 0.8
PIXEL_INTENSITY = 255


def wavelength_to_rgb(nm):
    factor = 0
    color = ()  # RGB
    if SHORTEST_WAVE <= nm < 440:
        color = (-(nm - 440) / (440 - SHORTEST_WAVE), 0.0, 1.0)
    elif 440 <= nm < 490:
        color = (0.0, (nm - 440) / (490 - 440), 1.0)
    elif 490 <= nm < 510:
        color = (0.0, 1.0, -(nm - 510) / (510 - 490))
    elif 510 <= nm < 580:
        color = ((nm - 510) / (580 - 510), 1.0, 0.0)
    elif 580 <= nm < 645:
        color = (1.0, -(nm - 645) / (645 - 580), 0.0)
    elif 645 <= nm < LONGEST_WAVE:
        color = (1.0, 0.0, 0.0)

    if SHORTEST_WAVE <= nm < 420:
        factor = 0.3 + 0.7 * (nm - SHORTEST_WAVE) / (420 - SHORTEST_WAVE)
    elif 420 <= nm < 701:
        factor = 1.0
    elif 701 <= nm < LONGEST_WAVE + 1:
        factor = 0.3 + 0.7 * (LONGEST_WAVE - nm) / (LONGEST_WAVE - 700)
    arr = []
    for ch in color:
        arr.append(max(0, int(PIXEL_INTENSITY * ((ch * factor) ** GAMMA))))
    return tuple(arr)


def main():
    image = np.zeros((IMAGE_H, IMAGE_W, IMAGE_C))
    x = 0
    for nm in range(SHORTEST_WAVE, LONGEST_WAVE):
        color = wavelength_to_rgb(nm)
        image = cv.rectangle(image, (x, 0), (x + BAR_W, IMAGE_H), color, thickness=-1)
        x += BAR_W
    # convert RGB to BGR
    image = image.astype(np.uint8)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    cv.imwrite(IMAGE_PATH, image)


if __name__ == '__main__':
    main()
