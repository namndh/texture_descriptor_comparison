import os
import sys
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import cv2
import numpy as np
import pywt
from skimage import feature
from torch.utils.data import Dataset
import glob
import mahotas as mt
from random import shuffle
from PIL import Image
import torch

import constants


def configs(dataset):
	if dataset == 'kth':
		return constants.KTH_TIPS_2_CONFIGS
	elif dataset == 'kylberg':
		return constants.KYLBERG_CONFIGS

class classifier():
	def __init__(self, args, configs):
		if args['model'] == 'svm':
			self.param_grid = {'C': [1], 'gamma': [0.1, 0.001], 
					'kernel':args['kernels_svm']}
			self.clf = GridSearchCV(SVC(), self.param_grid, verbose=1, n_jobs=6)

		if args['model'] == 'knn':
			self.clf = KNeighborsClassifier(5, weights='distance', n_jobs=12)
		
		if args['model'] == 'nb':
			self.clf = GaussianNB()

	def fit(self, X_train, y_train):
		self.clf.fit(X_train,y_train)

	def predict(self, X_test):
		self.predicted = self.clf.predict(X_test)

	def evaluate(self,X_test, y_test):
		self.predict(X_test)
		self.acc = accuracy_score(y_test, self.predicted)
		return self.acc

	def save_model(self):
		path_key = args['model'] + '_model_path'
		print('Saving...')
		with open(configs[path_key]) as model_bin:
			pickle.dump(self.clf, model_bin)

def extract_mean_std(img):
	mean, stdev = cv2.meanStdDev(img, mask=None)
	mean = mean[0][0]
	stdev = stdev[0][0]
	return mean, stdev


def gaborFilters_featuresExtractor(img, size=40):
	kernels = list()
	for i in np.arange(0, np.pi*2, np.pi/4):
		for j in np.arange(2,size/8*1+2,1):
			kernel=cv2.getGaborKernel((15, 15) , 3, i, j, 1, np.pi, cv2.CV_32F )
			kernels.append(kernel)
	features_vector = list()
	for kernel in kernels:
		img_output = cv2.filter2D(img, cv2.CV_8UC3, kernel)
		features_vector.extend(extract_mean_std(img_output))

	return features_vector

def Haar_featuresExtractor(img, depth=3):
	features_vector = list()
	for i in range(depth):
		coeff2 = pywt.dwt2(img, 'haar')
		LL, (LH, HL, HH) = coeff2
		features_vector.extend(extract_mean_std(LH))
		features_vector.extend(extract_mean_std(HL))
		features_vector.extend(extract_mean_std(HH))
		img = LL

	return features_vector

def DB4_featuresExtractor(img, depth=3):
	features_vector = list()
	for i in range(depth):
		coeff2 = pywt.dwt2(img, 'db4')
		LL, (LH, HL, HH) = coeff2
		features_vector.extend(extract_mean_std(LH))
		features_vector.extend(extract_mean_std(HL))
		features_vector.extend(extract_mean_std(HH))
		img = LL

	return features_vector

def LBP_featuresExtractor(img, numPoints=24, radius=8, eps=1e-7):
	lbp = feature.local_binary_pattern(img, numPoints, radius, method='uniform')
	(hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))

	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)

	# return the histogram of Local Binary Patterns
	return hist

def GLCM_featuresExtractor(image):
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean = textures.mean(axis=0)

	return ht_mean


def get_labels(folders, substrings):
	substr1 = '/' + substrings
	interval = len(substr1)+1
	labels = list()
	for folder in folders:
		idx = folder.find(substr1)
		label = folder[idx+interval:len(folder)]
		labels.append(label)
	labels = labels[1:]
	return labels

def build_fns_labels(labels, dataset, test_ratio=0.2):
	train_paths = list()
	test_paths = list()
	train_ratio = 1 - test_ratio
	for idx, label in enumerate(labels):
		data_paths = list()
		if dataset == 'kth':
			label_dir = os.path.join(constants.KTH_TIPS2_DATA_PATH, label)
		if dataset == 'kylberg':
			label_dir = os.path.join(constants.KYLBERG_DATA_PATH, label)

		fns = glob.glob(label_dir + '/*.png')
		for fn in fns:
			data_paths.append([fn, idx])
		train_pivot = int(train_ratio*len(data_paths))

		for path in data_paths[:int(train_pivot)]:
			train_paths.append(path)
		for path in data_paths[int(train_pivot):]:
			test_paths.append(path)

	return train_paths, test_paths

def data_loader(paths, transform):
	dataset = list()
	shuffle(paths)
	for path, label in paths:
		if not os.path.isfile(path):
			print('{} is None'.format(path))
			continue
		img = cv2.imread(path)
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_gray = cv2.resize(img_gray, (128,  128))

		data = transform(img_gray)
		dataset.append([data, label])

	return dataset


class MyDataset(Dataset):
	"""Custom dataset loader"""
	def __init__(self, datas, transform=None):
		assert datas != None, 'ERRORL: Data is not available!'
		self.transform = transform
		self.paths, self.labels = zip(*datas)
		self.paths = list(self.paths)
		self.labels = list(self.labels)
	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		image = Image.open(self.paths[idx])
		image = image.resize((224, 224))
		label = self.labels[idx]

		if self.transform:
			image = self.transform(image)

		return image, label


def exp_lr_scheduler(args, optimizer, epoch):
	init_lr = args.init_lr
	lr_decay_epoch = 50
	weight_decay = args.weight_decay
	lr = init_lr * (0.6**(min(epoch, 200) // lr_decay_epoch))
	print(lr)
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
		param_group['weight_decay'] = weight_decay

	return optimizer, lr

def save_convergence_model(save_loss, model, epoch):
	print("\nSaving convergence model at epoch {} with loss {}\n".format(epoch, save_loss))
	state = {
		'model' : model.state_dict(),
		'loss' : save_loss,
		'epoch' : epoch
	}
	if not os.path.isdir('./checkpoints/convergence'):
		os.mkdir('./checkpoints/convergence')

	torch.save(state, './checkpoints/convergence/checkpoint.t7')



extractors = {'gabor':gaborFilters_featuresExtractor, 'haar':Haar_featuresExtractor, 'db4':DB4_featuresExtractor, 'lbp':LBP_featuresExtractor, 
				'glcm':GLCM_featuresExtractor}