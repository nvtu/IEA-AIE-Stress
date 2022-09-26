# Add additional library
import sys, os, os.path as osp
import numpy as np

data_lib = os.path.abspath('../data')
if data_lib not in sys.path:
    sys.path.append(data_lib)

from data_utils import *


DATASET_NAME = 'WESAD'
# DATASET_NAME = 'AffectiveROAD'
# DEVICE = 'right'
DEVICE = 'wrist'
# NORMALIZER = '_nonorm'
NORMALIZER = ''

WINDOW_SIZE = 60
WINDOW_SIZE = 120
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()

dataset_stats_feature_path = get_dataset_stats_features_path(dp_manager, DATASET_NAME)


bvp_file_path = osp.join(dataset_stats_feature_path, f'{DATASET_NAME}_{DEVICE}_BVP{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
eda_file_path = osp.join(dataset_stats_feature_path, f'{DATASET_NAME}_{DEVICE}_EDA{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
temp_file_path = osp.join(dataset_stats_feature_path, f'{DATASET_NAME}_{DEVICE}_TEMP{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
bvp_stats_feat = np.load(bvp_file_path)
eda_stats_feat = np.load(eda_file_path)
temp_stats_feat = np.load(temp_file_path)


combined_stats_feat = np.concatenate((bvp_stats_feat, eda_stats_feat, temp_stats_feat), axis=1)
output_file_path = osp.join(dataset_stats_feature_path, f'{DATASET_NAME}_{DEVICE}_BVP_EDA_TEMP{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
np.save(output_file_path, combined_stats_feat)