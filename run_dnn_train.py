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
# physiological_signal_type = 'BVP_EDA'
physiological_signal_type = 'BVP_EDA_TEMP'
# physiological_signal_type = 'BVP'
# NORMALIZER = '_nonorm'
NORMALIZER = ''
# physiological_signal_type = 'EDA'
# dataset_name = 'DCU_NVT_EXP1'

# %%
WINDOW_SHIFT = 0.25
WINDOW_SIZE = 60
dp_manager = get_data_path_manager()

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
BATCH_SIZE = 1000
input_size = ds_loader.dataset.shape[1]
output_size = 2
_device = 'cuda'
num_epoches = 500
learning_rate = 1e-3
valid_percentage = 0.2
accs = []
baccs = []
test_groups = []


for train_index, test_index in tqdm(LeaveOneGroupOut().split(ds_loader.dataset, ds_loader.ground_truth, ds_loader.groups)):
    X_train, X_test = ds_loader.dataset[train_index], ds_loader.dataset[test_index]
    y_train, y_test = ds_loader.ground_truth[train_index], ds_loader.ground_truth[test_index]
    X_train, X_valid, y_train, y_valid = split_train_test_cv(X_train, y_train, test_size=valid_percentage)
    group_test = ds_loader.groups[test_index][0]

    _x_valid = torch.FloatTensor(X_valid).to(_device)
    _x_test = torch.FloatTensor(X_test).to(_device)

    model = DeepNeuralNetwork(input_size, output_size).to(_device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    perfect_loss = 1e19
    perfect_acc = 0

    for i in tqdm(range(num_epoches)):
        tensor_train_loader, tensor_train_labels = create_train_loader(X_train, y_train, _device, BATCH_SIZE)
        num_train = len(tensor_train_loader)
        valid_index = int(valid_percentage * num_train)
        tensor_train_loader, tensor_train_labels = tensor_train_loader[valid_index:], tensor_train_labels[valid_index:]
        for j, x in enumerate(tensor_train_loader):  
            model.zero_grad()

            y = torch.Tensor(tensor_train_labels[j]).to(_device)
            x = torch.Tensor(x).to(_device)
            out = model(x)
            loss = criterion(out, y)

            loss.backward()
            optimizer.step()
         
        y_valid_pred = model(_x_valid).argmax(dim = 1).cpu().numpy()
        y_true = y_valid
        acc = accuracy_score(y_true, y_valid_pred)
        _loss = loss.item()
        if _loss < perfect_loss and acc > perfect_acc:
            model_path = osp.join(get_model_path(dp_manager, dataset_name), f'{dataset_name}_{device}_{group_test}_{physiological_signal_type}_{WINDOW_SIZE}_{WINDOW_SHIFT}.pth')
            torch.save(model, model_path)
            perfect_loss = _loss
            perfect_acc = acc
        
        # if i % 100 == 0 or i == num_epoches - 1:
        #     print(f'Epoch {i}: Loss {loss.item()} --- Valid Accuracy: {acc}')

        #     y_pred = model(_x_test).argmax(dim = 1).cpu().numpy()
        #     # y_pred = clf.predict(X_test)
        #     y_true = y_test
        #     acc = accuracy_score(y_true, y_pred)
        #     balanced_acc = balanced_accuracy_score(y_true, y_pred)
        #     print(f'Test subject {group_test} --- Accuracy: {acc} --- Balanced Accuracy: {balanced_acc}')
        

    model = torch.load(model_path)
    y_pred = model(_x_test).argmax(dim = 1).cpu().numpy()
    # y_pred = clf.predict(X_test)
    y_true = y_test
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    test_groups.append(group_test)
    accs.append(acc)
    baccs.append(balanced_acc)

    print(f'Test subject {group_test} --- Accuracy: {acc} --- Balanced Accuracy: {balanced_acc}')

results = { 'groups': test_groups, 'accuracy_score': accs, 'balanced_accuracy_score': baccs }
detector_type = 'General'
detection_strategy = 'dnn'
result_helper = ResultUtils(dataset_name, device, physiological_signal_type, NORMALIZER = NORMALIZER, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
result_helper.dump_result_to_csv(results, detection_strategy, detector_type)
# %%

