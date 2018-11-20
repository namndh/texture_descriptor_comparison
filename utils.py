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

import constants


def configs(dataset):
	if dataset == 'kth':
		return constants.KTH_TIPS_2_CONFIGS
	elif dataset == 'kylberg':
		return constants.KYLBERG_CONFIGS

class classfier():
	def __init__(self, args, configs):
		if args.model == 'svm':
			if args.kernel_svm == 'linear':
				self.clf = SVC(args.kernel_svm, C=args.C)
			else:
				self.param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
				self.clf = GridSearchCV(SVC(), param_grid, verbose=1)

		if args.model == 'knn':
			self.clf = KNeighborsClassifier(configs['class_num'], weights='distance')
		
		if args.model == 'nb':
			self.clf = MultinomialNB(alpha=1, fit_prior=False, class_prior=False)

	def fit(self, X_train, y_train):
		self.clf.fit(X_train,y_train)

	def predict(self, X_test):
		self.predicted = self.clf.predict(X_test)

	def evaluate(self, y_test):
		self.acc = accuracy_score(y_test, self.predicted)
		return self.acc

	def save_model(self):
		path_key = args.model + '_model_path'
		print('Saving...')
		with open(configs[path_key]) as model_bin:
			pickle.dump(self.clf, model_bin)

def extract_mean_std(img):
	mean, stdev = cv2.meanStdDev(img, mask=None)
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
		mean, stdev = extract_mean_std(img)
		features_vector.append(mean)
		features_vector.append(stdev)

	return features_vector

def Haar_featuresExtractor(img, depth=3):
	features_vector = list()
	for i in range(depth):
		coeff2 = pywt.dwt2(img, 'haar')
		LL, (LH, HL, HH) = coeff2
		features_vector.extend((extract_mean_std(LH),extract_mean_std(HL), extract_mean_std(HH)))
		img = LL

	return features_vector

def DB4_featuresExtractor(img, depth=3):
	features_vector = list():
	for i in range(depth):
		coeff2 = pywt.dwt2(img, 'db4')
		LL, (LH, HL, HH) = coeff2
		features_vector.append((extract_mean_std(LH),extract_mean_std(HL), extract_mean_std(HH)))
		img = LL

	return features_vector

def LBP_featuresExtractor(img, numPoints=24, radius=8, eps=1e-7):
	lbp = feature.local_binary_pattern(img, numPoints, radius, method='uniform')
	(hist, _) = np.histogram(lbp.ravel(),
		bins=np.arange(0, self.numPoints + 3),
		range=(0, self.numPoints + 2))

	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)

	# return the histogram of Local Binary Patterns
	return hist

def GLCM_featuresExtractor(img):
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean = textures.mean(axis=0)

	return ht_mean


class Mydataset(Dataset):
	"""custom dataset to load and process image"""
    def __init__(self, data, transform=None):
    	self.fns = list()
    	self.labels = list()
    	for fn, label in data:
    		self.fns.append(fn)
    		self.labels.append(label)
    	self.transform = transform

    def __len__(self):
    	return len(self.fns)

    def __getitem__(self, idx):
    	image = cv2.imread(self.fns[idx])
    	
    	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    	if self.transform:
    		features_vector = self.transform(image)
        return features_vector, self.labels[idx]
		

def get_labels(folders):
    substr1 = 'subdataset/'
    interval = len(substr1)
    labels = list()
    for folder in folders:
        idx = folder.find(substr1)
        label = folder[idx+interval:len(folder)]
        labels.append(label)
    return labels

def build_fns_labels(labels, test_ratio = 0.2, val_ratio=0.2):
    train_paths = list()
    val_paths = list()
    test_paths = list()
    train_ratio = (1-val_ratio)*(1-test_ratio)
    val_ratio = (1-test_ratio)*val_ratio
    for idx, label in enumerate(labels):
        label_paths = list()
        label_dir = os.path.join(constants.DATASET_PATH, label)
        fns = glob.glob(label_dir+'/*.png')
        for fn in fns:
            label_paths.append([fn, idx])
        train_pivot = int(train_ratio*len(label_paths))
        val_pivot = train_pivot + int(val_ratio*len(label_paths))
        
        for path in label_paths[:int(train_ratio*len(label_paths))]:
            train_paths.append(path)
        for path in label_paths[train_pivot:val_pivot]:
            val_paths.append(path)
        for path in label_paths[val_pivot:]:
            test_paths.append(path)

    return train_paths, val_paths, test_paths

def split_datasets(dataset):
    fns, labels = list(zip(*dataset))
    fns = list(fns)
    idx = list(fns)
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(fns, labels, test_size=0.2, random_state=42, shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=0.25, random_state=42, 
        shuffle=True)
    trainset = list(zip(X_train, y_train))
    validateset = list(zip(X_validation, y_validation))
    testset = list(zip(X_test, y_test))
    return trainset, validateset, testset