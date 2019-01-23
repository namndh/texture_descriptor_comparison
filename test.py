import os
import cv2
import numpy

from scipy.ndimage import convolve

img = cv2.imread('/home/t3min4l/workspace/texture_descriptor_comparison/datas/KYLBERG/floor1/floor1-a-p001.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, (128, 128))

kernel = img.copy()

print(kernel.shape)

# dst = cv2.filter2D(img, -1, kernel)

res = convolve(img, kernel)

print(res)
cv2.imshow('n', res)
cv2.waitKey(0)