import sys, os

from sklearn import preprocessing


data_lib = os.path.abspath('../data')
if data_lib not in sys.path:
    sys.path.append(data_lib)

import numpy as np
from data_utils import *
from tqdm import tqdm
import os.path as osp
import pandas as pd
import warnings
import pickle
warnings.filterwarnings('ignore')


EDA_SAMPLING_RATE = 4
# DATASET_NAME = 'AffectiveROAD'
DATASET_NAME = 'WESAD'
# DEVICE = 'right'
DEVICE = 'wrist'
SIGNAL_NAME = 'EDA'

WINDOW_SIZE = 60
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()

dataset_path = osp.join(dp_manager.dataset_path, f'{DATASET_NAME}_{DEVICE}_dataset.pkl')
stats_feature_path = get_dataset_stats_features_path(dp_manager, DATASET_NAME)
bvp_stats_features = np.load(osp.join(stats_feature_path, f'{DATASET_NAME}_{DEVICE}_include_meditation', f'{DATASET_NAME}_{DEVICE}_BVP_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy'))
eda_stats_features = np.load(osp.join(stats_feature_path, f'{DATASET_NAME}_{DEVICE}_include_meditation', f'{DATASET_NAME}_{DEVICE}_EDA_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy'))
dataset = pickle.load(open(dataset_path, 'rb'))


groups = []
labels = []
new_bvp_stats_features = []
new_eda_stats_features = []
it = 0
for user_id, data in dataset['ground_truth'].items():
    print(f"Extracting metadata features of user {user_id}")
    for task_id, ground_truth in data.items():
        len_ground_truth = len(ground_truth)
        step = int(WINDOW_SHIFT * EDA_SAMPLING_RATE) # The true step to slide along the time axis of the signal
        first_iter = int(WINDOW_SIZE * EDA_SAMPLING_RATE) # The true index of the signal at a time-point
        if 'meditation' in task_id:
            it += len_ground_truth - first_iter + 1
            continue
        for current_iter in tqdm(range(first_iter, len_ground_truth, step)): # current_iter is "second_iter"
            gt = ground_truth[current_iter-1]
            new_bvp_stats_features.append(bvp_stats_features[it])
            new_eda_stats_features.append(eda_stats_features[it])
            it += 1
            labels.append(gt)
            groups.append(user_id)

    
groups = np.array(groups)
labels = np.array(labels)
new_bvp_stats_features = np.array(new_bvp_stats_features)
new_eda_stats_features = np.array(new_eda_stats_features)
output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_groups_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
np.save(output_file_path, groups)
output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_labels_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
np.save(output_file_path, labels)
output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_EDA_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
np.save(output_file_path, new_eda_stats_features)
output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_BVP_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
np.save(output_file_path, new_bvp_stats_features)