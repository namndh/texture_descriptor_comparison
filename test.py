import cv2
import sys
import os
from skimage.filters import gabor_kernel
import numpy as np
from utils import *
from torch.utils.data import Dataset, DataLoader
import datetime

# img_path = os.path.join('./data/Kylberg/blanket1/blanket1-a-p001.png')

# print(os.path.isfile(img_path))

# img = cv2.imread(img_path)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(img.shape)
# # cv2.imshow('img', img)
# # cv2.waitKey(10000)
# # cv2.destroyAllWindows()

# img = cv2.resize(img, (256, 256))

# mean, stddev = cv2.meanStdDev(img,mask=None)

# print(mean)
# print(stddev)
kernels = []
# for theta in range(4):
#     theta = theta / 4. * np.pi
#     for sigma in (1, 3):
#         for frequency in (0.05, 0.25):
#             kernel = np.real(gabor_kernel(frequency, theta=theta,
#                                           sigma_x=sigma, sigma_y=sigma))
#             kernels.append(kernel)

# print(len(kernels))       

# a = 0
# for i in np.arange(0, np.pi*2, np.pi/4):
# 	for j in np.arange(2,7,1):
# 		kernel=cv2.getGaborKernel((15, 15) , 3, i, j, 1, np.pi, cv2.CV_32F )
# 		kernels.append(kernel)

# print(len(kernels))
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
# lists = list()
# lists.extend([1,2,3,4])
# def returns():
# 	return 1, 3,4

# lists.extend(returns())
# print(lists)
# folders = [x[0] for x in os.walk(constants.KYLBERG_DATA_PATH)]
# # print(folders)
# labels = get_labels(folders, constants.KYLBERG_FOLDER)
# # print(labels)

# train_paths, test_paths = build_fns_labels(labels, dataset='kylberg')
# # print('{}.{}'.format(len(train_paths), len(test_paths)))
# # print(train_paths)
# print(train_paths)
# img = cv2.imread(train_paths[0][0])
# img = cv2.cvtColor(img,	 cv2.COLOR_BGR2GRAY)
# print(extractors['db4'](img))

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'datas')

GABOR_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_gabor.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_gabor.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_gabor.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_gabor.bin')
					}
HAAR_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_haar.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_haar.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_haar.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_haar.bin')
					}
DB4_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_db4.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_db4.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_db4.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_db4.bin')
					}

LBP_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_lbp.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_lbp.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_lbp.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_lbp.bin')
					}

GLCM_DATA_PATHS = {'kylberg_train':os.path.join(DATA_DIR, 'kylberg_train_glcm.bin'), 
					'kylberg_test':os.path.join(DATA_DIR, 'kylberg_test_glcm.bin'),
					'kth_train':os.path.join(DATA_DIR, 'kth_train_glcm.bin'),
					'kth_test':os.path.join(DATA_DIR, 'kth_test_glcm.bin')
					}

datas_paths = {'gabor':GABOR_DATA_PATHS, 'haar':HAAR_DATA_PATHS}

descriptor = 'gabor'
print(datas_paths[descriptor]['kylberg_train'])

x = datetime.datetime.now()

print(x.strftime("%H:%M-%d-%b-%Y"))

kth_folders = [x[0] for x in os.walk(constants.KTH_TIPS2_DATA_PATH)]
kth_labels = get_labels(kth_folders, constants.KTH_TIPS2_FOLDER)
kth_train_paths, kth_test_paths = build_fns_labels(kth_labels, 'kth')
print(len(kth_train_paths))

kylberg_folders = [x[0] for x in os.walk(constants.KYLBERG_DATA_PATH)]
kylberg_labels = get_labels(kylberg_folders, constants.KYLBERG_FOLDER)
kylberg_train_paths, kylberg_test_paths = build_fns_labels(kylberg_labels, 'kylberg')
print(len(kylberg_train_paths))