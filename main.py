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
parser.add_argument('--evaluate', '-e', action='store_true', help='Evaluate texture descriptors')
parser.add_argument('--process_data', '-pd', action='store_true', help='Processing data before evaluating')
args = parser.parse_args()

kth_config = configs('kth')
kylberg_config = configs('kylberg')
models = ['svm', 'nb', 'knn']
svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

if args.process_data:
    kylberg_folders = [x[0] for x in os.walk(constants.KYLBERG_DATA_PATH)]
    kylberg_labels = get_labels(folders, constants.KYLBERG_FOLDER)
    



