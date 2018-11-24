import sys
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from torch.utils.data import Dataset, DataLoader
import cv2


import constants
from utils import *

parser = argparse.ArgumentParser(description="Texture Descriptor Comparision")
parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate texture descriptors')
parser.add_argument('--process_data', '-pd', action='store_true', help='Processing data before evaluating')
args = parser.parse_args()

kth_config = configs('kth')
kylberg_config = configs('kylberg')
models = ['svm', 'nb', 'knn']
svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']




if args.process_data:
	kylberg_folders = [x[0] for x in os.walk(constants.KYLBERG_DATA_PATH)]
	kylberg_labels = get_labels(kylberg_folders, constants.KYLBERG_FOLDER)
	kylberg_train_paths, kylberg_test_paths = build_fns_labels(kylberg_labels, 'kylberg')

	kylberg_train_gabor_dataset = data_loader(kylberg_train_paths, transform=gaborFilters_featuresExtractor)
	with open(constants.GABOR_DATA_PATHS['kylberg_train'], 'wb') as f:
		pickle.dump(kylberg_train_gabor_dataset, f)

	kylberg_test_gabor_dataset = data_loader(kylberg_test_paths, transform=gaborFilters_featuresExtractor)
	with open(constants.GABOR_DATA_PATHS['kylberg_test'], 'wb') as f:
		pickle.dump(kylberg_test_gabor_dataset, f)

	kylberg_train_haar_dataset = data_loader(kylberg_train_paths, transform=Haar_featuresExtractor)
	with open(constants.HAAR_DATA_PATHS['kylberg_train'], 'wb') as f:
		pickle.dump(kylberg_train_haar_dataset, f)

	kylberg_test_haar_dataset = data_loader(kylberg_test_paths, transform=Haar_featuresExtractor)
	with open(constants.HAAR_DATA_PATHS['kylberg_test'], 'wb') as f:
		pickle.dump(kylberg_test_haar_dataset, f)

	kylberg_train_db4_dataset = data_loader(kylberg_train_paths, transform=DB4_featuresExtractor)
	with open(constants.DB4_DATA_PATHS['kylberg_train'], 'wb') as f:
		pickle.dump(kylberg_train_db4_dataset, f)

	kylberg_test_db4_dataset = data_loader(kylberg_test_paths, transform=DB4_featuresExtractor)
	with open(constants.DB4_DATA_PATHS['kylberg_test'], 'wb') as f:
		pickle.dump(kylberg_test_db4_dataset, f)

	kylberg_train_lbp_dataset = data_loader(kylberg_train_paths, transform=LBP_featuresExtractor)
	with open(constants.LBP_DATA_PATHS['kylberg_train'], 'wb') as f:
		pickle.dump(kylberg_train_lbp_dataset, f)

	kylberg_test_lbp_dataset = data_loader(kylberg_test_paths, transform=LBP_featuresExtractor)
	with open(constants.LBP_DATA_PATHS['kylberg_test'], 'wb') as f:
		pickle.dump(kylberg_test_lbp_dataset, f)

	kylberg_train_glcm_dataset = data_loader(kylberg_train_paths, transform=GLCM_featuresExtractor)
	with open(constants.LBP_DATA_PATHS['kylberg_train'], 'wb') as f:
		pickle.dump(kylberg_train_lbp_dataset, f)

	kylberg_test_lbp_dataset = data_loader(kylberg_test_paths, transform=GLCM_featuresExtractor)
	with open(constants.LBP_DATA_PATHS['kylberg_test'], 'wb') as f:
		pickle.dump(kylberg_test_lbp_dataset, f)


	kth_folders = [x[0] for x in os.walk(constants.KTH_TIPS2_DATA_PATH)]
	kth_labels = get_labels(kth_folders, constants.KTH_TIPS2_FOLDER)
	kth_train_paths, kth_test_paths = build_fns_labels(kth_labels, 'kth')

	kth_train_gabor_dataset = data_loader(kth_train_paths, transform=gaborFilters_featuresExtractor)
	with open(constants.GABOR_DATA_PATHS['kth_train'], 'wb') as f:
		pickle.dump(kth_train_gabor_dataset, f)

	kth_test_gabor_dataset = data_loader(kth_test_paths, transform=gaborFilters_featuresExtractor)
	with open(constants.GABOR_DATA_PATHS['kth_test'], 'wb') as f:
		pickle.dump(kth_test_gabor_dataset, f)

	kth_train_gabor_dataset = data_loader(kth_train_paths, transform=Haar_featuresExtractor)
	with open(constants.HAAR_DATA_PATHS['kth_train'], 'wb') as f:
		pickle.dump(kth_train_gabor_dataset, f)

	kth_test_gabor_dataset = data_loader(kth_test_paths, transform=Haar_featuresExtractor)
	with open(constants.HAAR_DATA_PATHS['kth_test'], 'wb') as f:
		pickle.dump(kth_test_gabor_dataset, f)

	kth_train_gabor_dataset = data_loader(kth_train_paths, transform=DB4_featuresExtractor)
	with open(constants.DB4_DATA_PATHS['kth_train'], 'wb') as f:
		pickle.dump(kth_train_gabor_dataset, f)

	kth_test_gabor_dataset = data_loader(kth_test_paths, transform=DB4_featuresExtractor)
	with open(constants.DB4_DATA_PATHS['kth_test'], 'wb') as f:
		pickle.dump(kth_test_gabor_dataset, f)

	kth_train_gabor_dataset = data_loader(kth_train_paths, transform=LBP_featuresExtractor)
	with open(constants.LBP_DATA_PATHS['kth_train'], 'wb') as f:
		pickle.dump(kth_train_gabor_dataset, f)

	kth_test_gabor_dataset = data_loader(kth_test_paths, transform=LBP_featuresExtractor)
	with open(constants.LBP_DATA_PATHS['kth_test'], 'wb') as f:
		pickle.dump(kth_test_gabor_dataset, f)