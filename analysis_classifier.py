# %%
import sys, os

data_lib = os.path.abspath('data')
if data_lib not in sys.path:
    sys.path.append(data_lib)
import os
import os.path as osp
import numpy as np
import configparser
from models.classifiers import BinaryClassifier
from models.deep_neural_network import *
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from data.dataset_loader import *
from data.result_utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch

# %%
# %%
# -- Uncomment the dataset that you wanna load -- #
# dataset_name = 'AffectiveROAD'
dataset_name = 'WESAD'
# device = 'right'
device = 'wrist'
physiological_signal_type = 'BVP_EDA'
# physiological_signal_type = 'BVP'
# NORMALIZER = '_nonorm'
NORMALIZER = ''
# physiological_signal_type = 'EDA'
# dataset_name = 'DCU_NVT_EXP1'

# %%
WINDOW_SHIFT = 0.25
WINDOW_SIZE = 60

# %%
# %%
ds_loader = DatasetLoader(dataset_name, device, physiological_signal_type, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)

# %% [markdown]
# # Define stress detection strategies

# %%
# -- Uncomment the detection strategy that you wanna use to detect -- #
strategies = ['random_forest'] #, 'mlp']
# detection_strategy = 'logistic_regression'
# detection_strategy = 'random_forest'
# detection_strategy = 'svm'
# detection_strategy = 'mlp'
# detection_strategy = 'knn'


# %%
# SCORING = 'accuracy'
SCORING = 'balanced_accuracy'

# %%
def get_classifier(method: str):
    clf = None
    if method == 'random_forest':
        clf = RandomForestClassifier(n_estimators = 250, random_state = 0, n_jobs = -1, max_depth = 8, min_samples_leaf = 4, min_samples_split = 2, oob_score = True, bootstrap = True, class_weight = 'balanced')
    return clf
        

# %%
BATCH_SIZE = 50
input_size = ds_loader.dataset.shape[1]
output_size = 2
device = 'cuda'
num_epoches = 1000
learning_rate = 1e-4

for train_index, test_index in tqdm(LeaveOneGroupOut().split(ds_loader.dataset, ds_loader.ground_truth, ds_loader.groups)):
    X_train, X_test = ds_loader.dataset[train_index], ds_loader.dataset[test_index]
    y_train, y_test = ds_loader.ground_truth[train_index], ds_loader.ground_truth[test_index]
    groups_test = ds_loader.groups[test_index][0]

    _x_test = torch.FloatTensor(X_test).to(device)
    _y_test = torch.FloatTensor(y_test).to(device)

    model = DeepNeuralNetwork(input_size, output_size).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for i in tqdm(range(num_epoches)):
        tensor_train_loader, tensor_train_labels = create_train_loader(X_train, y_train, device, BATCH_SIZE)
        for j, x in enumerate(tensor_train_loader):  
            model.zero_grad()

            y = torch.Tensor(tensor_train_labels[j]).to(device)
            x = torch.Tensor(x).to(device)
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
        
        if i % 10 == 0:
            print(f'Epoch {i+1}: {loss.item()}')

            y_pred = model(_x_test).argmax(dim = 1).cpu().numpy()
            # y_pred = clf.predict(X_test)
            y_true = y_test
            acc = accuracy_score(y_true, y_pred)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            print(f'Test subject {groups_test} --- Accuracy: {acc} --- Balanced Accuracy: {balanced_acc}')

# %%
print(len)

# %%
# for train_index, test_index in tqdm(LeaveOneGroupOut().split(ds_loader.X, ds_loader.y, ds_loader.groups)):
#     X_train, X_test = ds_loader.X[train_index], ds_loader.X[test_index]
#     y_train, y_test = ds_loader.y[train_index], ds_loader.y[test_index]
#     groups_test = ds_loader.groups[test_index][0]
#     for strategy in strategies:
#         clf = get_classifier(strategy)
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         y_true = y_test
#         acc = accuracy_score(y_true, y_pred)
#         balanced_acc = balanced_accuracy_score(y_true, y_pred)
#         print(f'Test subject {groups_test} --- Accuracy: {acc} --- Balanced Accuracy: {balanced_acc}')

# %%
# for detection_strategy in strategies:
#     detector_type = 'General'
#     print(f'--- RUNNING {detector_type} {detection_strategy} ---')
#     clf = BinaryClassifier(ds_loader.dataset, ds_loader.ground_truth, detection_strategy, basic_logo_validation = True, groups = ds_loader.groups, scoring = SCORING)
#     results = clf.exec_classifier() # Build classifier and return prediction results
#     print('------------------------------------------------------')

    # # %%
    # # Save results
    # result_helper = ResultUtils(dataset_name, device, physiological_signal_type, NORMALIZER = NORMALIZER, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
    # result_helper.dump_result_to_csv(results, detection_strategy, detector_type)

# %%



