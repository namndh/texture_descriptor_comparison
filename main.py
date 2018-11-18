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

import constants
from utils import *

parser = argparse.ArgumentParser(description="Texture Descriptor Comparision")
parser.add_argument('--train', '-t', action='store_true', help='Train the model')
parser.add_argument('--model', default='svm', choices=['svm', 'nb', 'knn'], help='Choose model the train')
parser.add_argument('--kernel_svm', default='linear', choices=['linear', 'poly', 'rbf', 'sigmoid'], help='Choose kernel for SVM classifier')
parser.add_argument('-C', default=1, type=int, help='C parameter for SVM')

parser.add_argument('--predict', '-p', action='store_true', help='Predict data and log')
args = parser.parse_args()


x_train_kylberg = None
y_train_kylberg = None
x_test_kylberg = None
y_test_kylberg = None

x_train_kth_tips_2 = None
y_train_kth_tips_2 = None
x_test_kth_tips_2 = None
y_test_kth_tips_2 =None

kth_config = configs('kth')
kylberg_config = configs('kylberg')


if args.train:
    if args.model == 'svm':
        if args.kernel_svm == 'linear':
            clf_kylberg = SVC(kernel=args.kernel, C=args.C)
            clf_kth_tips_2 = SVC(kernel=args.kernel, C=args.C)

            clf_kylberg.fit(x_train_kylberg, y_train_kylberg)
            with open(constants.KYLBERG_SVM_MODEL, 'wb') as kylberg_model_bin:
                pickle.dump(clf_kylberg, kylberg_model_bin)

            clf_kth_tips_2.fit(x_train_kth_tips_2, y_train_kth_tips_2)
            with open(constants.KTH_TIPS_2_SVM_MODEL, 'wb') as kth_tips_2_model_bin:
                pickle.dump(clf_kth_tips_2, kth_tips_2_model_bin)


        else:
            param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
            clf_kylberg_grid = GridSearchCV(SVC(), param_grid, verbose=1)
            clf_kth_tips_2_grid = GridSearchCV(SVC(), param_grid, verbose=1)

            clf_kylberg_grid.fit(x_train_kylberg, y_train_kylberg)
            with open(constants.KYLBERG_SVM_MODEL, 'wb') as kylberg_model_bin:
                pickle.dump(clf_kylberg_grid, kylberg_model_bin)

            clf_kth_tips_2_grid.fit(x_train_kth_tips_2, y_train_kth_tips_2)
            with open(constants.KTH_TIPS_2_SVM_MODEL, 'wb') as kth_tips_2_model_bin:
                pickle.dump(clf_kth_tips_2_grid, kth_tips_2_model_bin)
    if args.model == 'knn':
        clf_kylberg = KNeighborsClassifier(constants.KYLBERG_CLASS_NUM*constants.KNN_N_NEIGHBORS, weights='distance')
        clf_kth_tips_2 = KNeighborsClassifier(constants.KTH_TIPS_2_CLASS_NUM*constants.KNN_N_NEIGHBORS, weights='distance')

        clf_kylberg.fit(x_train_kylberg, y_train_kylberg)
        with open(constants.KYlBERG_KNN_MODEL, 'wb') as kylberg_model_bin:
            pickle.dump(clf_kylberg, kylberg_model_bin)

        clf_kth_tips_2.fit(x_train_kth_tips_2, y_train_kth_tips_2)
        with open(constants.KTH_TIPS_2_KNN_MODEL, 'wb') as kth_tips_2_model_bin:
            pickle.dump(clf_kth_tips_2, kth_tips_2_model_bin)

    if args.model == 'nb':
        clf_kylberg = MultinomialNB(alpha=1, fit_prior=False, class_prior=False)
        clf_kth_tips_2 = MultinomialNB(alpha=1, fit_prior=False, class_prior=False)

        clf_kylberg.fit(x_train_kylberg, y_train_kylberg)
        with open(constants.KYLBERG_NB_MODEL, 'wb') as kylberg_model_bin:
            pickle.dump(clf_kylberg, kylberg_model_bin)

        clf_kth_tips_2.fit(x_train_kth_tips_2, y_train_kth_tips_2)
        with open(constants.KTH_TIPS_2_NB_MODEL, 'wb') as kth_tips_2_model_bin:
            pickle.dump(clf_kth_tips_2, kth_tips_2_model_bin)

if args.predict:




