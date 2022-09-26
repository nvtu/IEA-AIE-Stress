# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from data.dataset_loader import *
from data.result_utils import *

# %% [markdown]
# # Load eda statistical features and ground-truth from datasets
# %%
# -- Uncomment the dataset that you wanna load -- #
# dataset_name = 'AffectiveROAD'
dataset_name = 'WESAD'
# device = 'right'
device = 'wrist'
physiological_signal_type = 'BVP_EDA_TEMP'
# physiological_signal_type = 'BVP_EDA'
physiological_signal_type = 'BVP'
# physiological_signal_type = 'EDA'
# NORMALIZER = '_nonorm'
NORMALIZER = ''
# NORMALIZER = '_stdnorm'
# physiological_signal_type = 'TEMP'
# dataset_name = 'DCU_NVT_EXP1'

# %%
WINDOW_SHIFT = 0.25
WINDOW_SIZE = 60
# WINDOW_SIZE = 120

# %%
ds_loader = DatasetLoader(dataset_name, device, physiological_signal_type, NORMALIZER = NORMALIZER, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)

# %% [markdown]
# # Define stress detection strategies

# %%
# -- Uncomment the detection strategy that you wanna use to detect -- #
strategies = ['svm'] #, 'mlp']
# detection_strategy = 'logistic_regression'
# detection_strategy = 'random_forest'
# detection_strategy = 'svm'
# detection_strategy = 'mlp'
# detection_strategy = 'knn'


# %%
# SCORING = 'accuracy'
SCORING = 'balanced_accuracy'


# %% [markdown]
# # Build General Cross-population Stress Detector

# %%
for detection_strategy in strategies:
    detector_type = 'General'
    print(f'--- RUNNING {detector_type} {detection_strategy} ---')
    clf = BinaryClassifier(ds_loader.dataset, ds_loader.ground_truth, detection_strategy, basic_logo_validation = True, groups = ds_loader.groups, scoring = SCORING)
    results = clf.exec_classifier() # Build classifier and return prediction results
    print('------------------------------------------------------')

    # # %%
    # # Save results
    result_helper = ResultUtils(dataset_name, device, physiological_signal_type, NORMALIZER = NORMALIZER, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
    result_helper.dump_result_to_csv(results, detection_strategy, detector_type)

# %% [markdown]
# # Build Person-specific Stress Detector

# %%
# for detection_strategy in strategies:
#     detector_type = 'Personal'

#     print(f'--- RUNNING {detector_type} {detection_strategy} ---')
#     clf = BinaryClassifier(ds_loader.dataset, ds_loader.ground_truth, detection_strategy, cross_validation = True, groups = ds_loader.groups, scoring = SCORING)
#     # clf = BinaryClassifier(ds_loader.hrv_dataset, ds_loader.ground_truth, detection_strategy, cross_validation = True, groups = ds_loader.groups, scoring = SCORING)
#     # clf = BinaryClassifier(ds_loader.eda_dataset, ds_loader.ground_truth, detection_strategy, cross_validation = True, groups = ds_loader.groups, scoring = SCORING)
#     results = clf.exec_classifier()
#     print('------------------------------------------------------')

    # %%
    # Save results
    # result_helper = ResultUtils(dataset_name, WINDOW_SHIFT = WINDOW_SHIFT, WINDOW_SIZE = WINDOW_SIZE)
    # result_helper.dump_result_to_csv(results, detection_strategy, detector_type, physiological_signal_type = '')
