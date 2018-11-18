import os
import sys
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import pickle

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

		if args.model = 'knn':
			self.clf = KNeighborsClassifier(configs['class_num'], weights='distance')
		
		if args.model = 'nb':
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

def gaborFilters_featuresExtractor(img, kernels):
	