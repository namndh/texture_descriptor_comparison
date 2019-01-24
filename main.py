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
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from shuffle_net import ShuffleNet
from shuffle_net_v2 import ShuffleNetV2
import constants
from utils import *

parser = argparse.ArgumentParser(description="Texture Descriptor Comparision")
parser.add_argument('--evaluate_descriptors', '-ed', action='store_true', help='Evaluate texture descriptors')
parser.add_argument('--evaluate_shuffle_net', '-es', action='store_true', help='Evaluate classfication with Shuffle Net')
parser.add_argument('--process_data', '-pd', action='store_true', help='Processing data before evaluating')
parser.add_argument('--init_lr', default=1e-2, type=float, help='Learning rate - default: 1e-2')
parser.add_argument('--weight_decay', default=5e-6, type=float, help='Weight decay - default: 5e-6 ')
parser.add_argument('--optim', default='sgd', choices=['adam', 'sgd'])
parser.add_argument('--batch_size', default=128, type=int, help='Batch size - default: 128')
parser.add_argument('--num_epochs', default=300, type=int, help='Number of epochs in training - default : 300')
parser.add_argument('--drop_out', default=0.5, type=float)
parser.add_argument('--check_after', default=1, type=int, help='Validate the model after how many epoch - default : 1')
args = parser.parse_args()

kth_config = configs('kth')
kylberg_config = configs('kylberg')
models = ['svm', 'nb', 'knn']
svm_kernels = ['linear']
descriptors = ['glcm']
datasets = ['kth']


def save_best_acc_model(save_acc, model, epoch, device, dataset):
	print("\nSaving ...")
	state = {
		'model' : model.state_dict(),
		'acc'  : save_acc,
		'epoch' : epoch,
		'device' : device
	}
	if not os.path.isdir('./checkpoints/best_acc'):
		os.mkdir('./checkpoints')
		os.mkdir('./checkpoints/best_acc')

	torch.save(state, './checkpoints/best_acc/'+ dataset + '_checkpoint.t7')

def train_evaluate(args, epoch, optimizer, model, criterion, train_loader, validate_loader, device, dataset):
	optimizer, lr = exp_lr_scheduler(args, optimizer, epoch)
	# print(optimizer)
	print(lr)
	print("\n=================================================\n")
	print('Epoch {}.'.format(epoch+1))
	print('==> Training at LR {:.5}'.format(lr))
	model.train()
	train_loss = 0
	train_acc = 0
	total = 0

	for idx, (image, labels) in enumerate(train_loader):
		images, labels = image.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss
		_, predicted = outputs.max(1)

		total += labels.size(0)
		train_acc += predicted.eq(labels).sum().item()
	epoch_loss = train_loss/(idx+1)
	epoch_train_acc = train_acc/total

	print('\nTraining: Loss {} || Accuracy {:.5}%.\n'.format(epoch_loss, epoch_train_acc*100))

	if (epoch+1) % args.check_after == 0:
		print("\n===============================================\n")
		print('==> Validating')
		model.eval()
		validate_acc = 0

		total = 0
		for idx, (images, labels) in enumerate(validate_loader):
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)

			_, predicted  = outputs.max(1)
			total += labels.size(0)
			validate_acc += predicted.eq(labels).sum().item()
		epoch_validate_acc = validate_acc/total*100
		print('\nValidating: Accuracy {:.3}%.'.format(epoch_validate_acc))

		global save_acc
		if epoch_validate_acc > save_acc:
			save_acc = epoch_validate_acc
			save_best_acc_model(save_acc, model, epoch, device, dataset)

def predict(model, test_loader, dataset):
	assert os.path.isdir('./checkpoints'), 'ERROR: model is not available!'
	checkpoint = torch.load('./checkpoints/best_acc/'+dataset+'_checkpoint.t7')
	model.load_state_dict(checkpoint['model'])
	acc = checkpoint['acc']
	epoch = checkpoint['epoch']
	device = checkpoint['device']
	print('Model used to predict has best acc {:.3}% on validate set at epoch {}.'.format(acc, epoch))

	torch.set_grad_enabled(False)
	model.eval()

	test_correct = 0
	total = 0 

	for idx, (images, labels) in enumerate(test_loader):
		images, labels = images.to(device), labels.to(device)

		outputs = model(images)
		_, predicted = outputs.max(1)

		test_correct += predicted.eq(labels).sum().item()
		total += labels.size(0)


	print('\nAccuracy of model in predicting in dataset {} is {:.3}%.\n'.format(dataset, test_correct/total*100))
save_acc = 0

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

if args.evaluate_descriptors:
	x = datetime.datetime.now()
	time = x.strftime("%H:%M-%d-%b-%Y")
	time_created = time
	log_path = './log_' + time +'_.txt'
	f=open(log_path, 'w+')
	f.write('dataset, descriptor, classifier, acc, time\n')
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

			config = configs(dataset)
			for model in models:
				FMT = '%H:%M:%S'
				time1 = datetime.datetime.now().strftime('%H:%M:%S')
				model_args = {'model':model, 'kernels_svm':svm_kernels}
				clf = classifier(model_args, config)
				clf.fit(x_train, y_train)
				acc = clf.evaluate(x_test, y_test)
				
				time2 = datetime.datetime.now().strftime('%H:%M:%S')
				deltaTime = datetime.datetime.strptime(time2, FMT) - datetime.datetime.strptime(time1, FMT)
				strings = str(dataset) + ',' + str(descriptor) + ',' + str(model) + ',' + str(acc) + ',' +  str(deltaTime) +'\n'
				print(strings)
				f=open(log_path, 'a')
				f.write(strings)
				f.close()

if args.evaluate_shuffle_net:
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('availabel device:', device)

	transforms_train = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	])

	transforms_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
	])
	# start_epoch = 0
	# save_acc = 0

	print(args.init_lr)
	for dataset in datasets:
		if dataset == 'kylberg':
			num_class = constants.KYLBERG_CLASS_NUM
			kylberg_folders = [x[0] for x in os.walk(constants.KYLBERG_DATA_PATH)]
			kylberg_labels = get_labels(kylberg_folders, constants.KYLBERG_FOLDER)
			
			kylberg_train_paths, kylberg_test_paths = build_fns_labels(kylberg_labels, 'kylberg')
			shuffle(kylberg_train_paths)
			shuffle(kylberg_test_paths)
			
			kylberg_valid_paths = kylberg_train_paths[int(len(kylberg_train_paths)*0.8):]
			del kylberg_train_paths[int(len(kylberg_train_paths)*0.8):]
			train_data = MyDataset(kylberg_train_paths, transform=transforms_train)
			test_data = MyDataset(kylberg_test_paths, transform=transforms_test)
			validate_data = MyDataset(kylberg_valid_paths, transform=transforms_test)

		if dataset == 'kth':
			num_class = constants.KTH_TIPS_2_CLASS_NUM
			kth_folders = [x[0] for x in os.walk(constants.KTH_TIPS2_DATA_PATH)]
			kth_labels = get_labels(kth_folders, constants.KTH_TIPS2_FOLDER)
			
			kth_train_paths, kth_test_paths = build_fns_labels(kth_labels, 'kth')
			shuffle(kth_train_paths)
			shuffle(kth_test_paths)
	
			kth_valid_paths = kth_train_paths[int(len(kth_train_paths)*0.8):]
			del kth_train_paths[int(len(kth_train_paths)*0.8):]			

			train_data = MyDataset(kth_train_paths, transform=transforms_train)
			test_data = MyDataset(kth_test_paths, transform=transforms_test)
			validate_data = MyDataset(kth_valid_paths, transform=transforms_test)

		if train_data is not None and test_data is not None and validate_data is not None:
			train_set = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
			val_set = DataLoader(dataset=validate_data, batch_size=args.batch_size, shuffle=True)
			model = ShuffleNetV2(n_class=num_class)
			model.to(device)
			criterion = nn.CrossEntropyLoss()
			if args.optim == 'sgd':
				optimizer = optim.SGD(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay, momentum=0.9)
			if args.optim =='adam':
				optimizer = optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
				
			# for epoch in range(args.num_epochs):
			# 	train_evaluate(args, epoch, optimizer, model, criterion, train_set, val_set, device,dataset)	
			test_set = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True)
			print(dataset + '\n')
			predict(model, test_set, dataset)


			