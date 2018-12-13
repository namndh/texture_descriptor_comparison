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
import csv
import datetime
from subprocess import call

import constants
from utils import *

parser = argparse.ArgumentParser(description="Texture Descriptor Comparision")
parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate texture descriptors')
parser.add_argument('--process_data', '-pd', action='store_true', help='Processing data before evaluating')
args = parser.parse_args()

kth_config = configs('kth')
kylberg_config = configs('kylberg')
models = ['svm', 'nb', 'knn']
svm_kernels = ['linear']
descriptors = ['glcm']
datasets = ['kylberg']


if args.process_data:
	kylberg_folders = [x[0] for x in os.walk(constants.KYLBERG_DATA_PATH)]
	kylberg_labels = get_labels(kylberg_folders, constants.KYLBERG_FOLDER)
	kylberg_train_paths, kylberg_test_paths = build_fns_labels(kylberg_labels, 'kylberg')

	kth_folders = [x[0] for x in os.walk(constants.KTH_TIPS2_DATA_PATH)]
	kth_labels = get_labels(kth_folders, constants.KTH_TIPS2_FOLDER)
	kth_train_paths, kth_test_paths = build_fns_labels(kth_labels, 'kth')

	for descriptor in descriptors:
		kylberg_train_data = data_loader(kylberg_train_paths, transform=extractors[descriptor])
		with open(constants.datas_paths[descriptor]['kylberg_train'], 'wb') as f:
			pickle.dump(kylberg_train_data, f)
		kylberg_test_data = data_loader(kylberg_test_paths, transform=extractors[descriptor])
		with open(constants.datas_paths[descriptor]['kylberg_test'], 'wb') as f:
			pickle.dump(kylberg_test_data, f)

		kth_train_data = data_loader(kth_train_paths, transform=extractors[descriptor])
		with open(constants.datas_paths[descriptor]['kth_train'], 'wb') as f:
			pickle.dump(kth_train_data, f)
		kth_test_data = data_loader(kth_test_paths, transform=extractors[descriptor])
		with open(constants.datas_paths[descriptor]['kth_test'], 'wb') as f:
			pickle.dump(kth_test_data, f)

	print('Finished processing and saving data!')

if args.evaluate:
	x = datetime.datetime.now()
	time = x.strftime("%H:%M-%d-%b-%Y")
	time_created = time
	log_path = './log_' + time +'_.txt'
	f=open(log_path, 'w+')
	f.write('dataset, descriptor, classifier, acc\n')
	f.close()
	for descriptor in descriptors:
		for dataset in datasets:
			with open(constants.datas_paths[descriptor][dataset+'_train'], 'rb') as f:
				train_data = pickle.load(f)
			with open(constants.datas_paths[descriptor][dataset+'_test'], 'rb') as f:
				test_data = pickle.load(f)

			x_train, y_train = zip(*train_data)
			x_train = list(x_train)
			y_train = list(y_train)

			x_test, y_test = zip(*test_data)
			x_test = list(x_test)
			y_test = list(y_test)

			# print(x_train[0].shape)
			# print(x_test[0].shape)
			config = configs(dataset)
			for model in models:
				model_args = {'model':model, 'kernels_svm':svm_kernels}
				clf = classifier(model_args, config)
				clf.fit(x_train, y_train)
				acc = clf.evaluate(x_test, y_test)
				strings = str(dataset) + ',' + str(descriptor) + ',' + str(model) + ',' + str(acc) + '\n'
				print(strings)
				f=open(log_path, 'a')
				f.write(str(dataset) + ',' + str(descriptor) + ',' + str(model) + ',' + str(acc) + '\n')
				f.close()

# dataset = 'kth'
# descriptor = 'db4'
# with open(constants.datas_paths[descriptor][dataset + '_train'], 'rb') as f:
# 	train_data = pickle.load(f)

# with open(constants.datas_paths[descriptor][dataset + '_test'], 'rb') as f:
# 	test_data = pickle.load(f)

# x_train, y_train = zip(*train_data)
# x_train = list(x_train)
# y_train = list(y_train)

# x_test, y_test = zip(*test_data)
# x_test = list(x_test)
# y_test = list(y_test)
# config = configs(dataset)
# for model in models:
# 	model_args = {'model':model, 'kernels_svm':svm_kernels}
# 	clf = classifier(model_args, config)
# 	clf.fit(x_train, y_train)
# 	acc = clf.evaluate(x_test, y_test)
# 	strings = str(dataset) + ',' + str(descriptor) + ',' + str(model) + ',' + str(acc) + '\n'
# 	print(strings)
# 	f=open('./log_18:57-27-Nov-2018_.txt', 'a')
# 	f.write(str(dataset) + ',' + str(descriptor) + ',' + str(model) + ',' + str(acc) + '\n')
# 	f.close()