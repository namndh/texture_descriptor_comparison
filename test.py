import cv2
import sys
import os
from skimage.filters import gabor_kernel
import numpy as np
img_path = os.path.join('./data/Kylberg/blanket1/blanket1-a-p001.png')

print(os.path.isfile(img_path))

img = cv2.imread(img_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape)
# cv2.imshow('img', img)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

img = cv2.resize(img, (256, 256))

mean, stddev = cv2.meanStdDev(img,mask=None)

print(mean)
print(stddev)
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)

print(len(kernels))       

a = 0
for i in np.arange(0, np.pi*2, np.pi/4):
	print(i/np.pi)


# winSize = (64,64)
# blockSize = (16,16)
# blockStride = (8,8)
# cellSize = (8,8)
# nbins = 9
# derivAperture = 1
# winSigma = 4.
# histogramNormType = 0
# L2HysThreshold = 2.0000000000000001e-01
# gammaCorrection = 0
# nlevels = 64
# hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

# winStride = (8,8)
# padding = (8,8)
# img = cv2.resize(img,(256, 256))
# h = hog.compute(img,winStride, padding)
# print(hog.getDescriptorSize())
# print(len(hog.computeGradient(img, winStride, padding)))
# print(h.shape)