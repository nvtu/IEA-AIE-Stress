# Add additional library
import sys, os

data_lib = os.path.abspath('../data')
eda_sp_lib = os.path.abspath('../signal_processing')
if data_lib not in sys.path:
    sys.path.append(data_lib)
if eda_sp_lib not in sys.path:
    sys.path.append(eda_sp_lib)

import numpy as np
from data_utils import *
from eda_signal_processing import *
from tqdm import tqdm
import os.path as osp
import pickle
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress
warnings.filterwarnings('ignore')


TEMP_SAMPLING_RATE = 4
# DATASET_NAME = 'AffectiveROAD'
DATASET_NAME = 'WESAD'
DEVICE = 'wrist'
# DEVICE =        if 'meditation' in task_id: continue
SIGNAL_NAME = 'TEMP'
# NORMALIZER = '_nonorm'
NORMALIZER = ''

# WINDOW_SIZE = 60
WINDOW_SIZE = 120
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()


def temp_feature_extraction(signal, sampling_rate):
    len_signal = len(signal)
    t = np.linspace(0, len_signal, len_signal)
    mean_temp, std_temp = signal.mean(), signal.std()
    min_temp, max_temp = signal.min(), signal.max()
    range_temp = max_temp - min_temp
    res = linregress(t, signal)
    temp_slope = res.slope
    features = [mean_temp, std_temp, min_temp, max_temp, range_temp, temp_slope]
    return features


if __name__ == '__main__':
    temp_stats_features = []

    dataset_path = osp.join(dp_manager.dataset_path, f'{DATASET_NAME}_{DEVICE}_dataset.pkl')
    dataset = pickle.load(open(dataset_path, 'rb'))

    for user_id, data in dataset['temp'].items():
        print(f"Extracting TEMP Features of user {user_id}")
        for task_id, temp_signal in data.items():
            len_temp_signal = len(temp_signal)
            step = int(WINDOW_SHIFT * TEMP_SAMPLING_RATE) # The true step to slide along the time axis of the signal
            first_iter = int(WINDOW_SIZE * TEMP_SAMPLING_RATE) # The true index of the signal at a time-point 
            for current_iter in tqdm(range(first_iter, len_temp_signal, step)): # current_iter is "second_iter"
                previous_iter = current_iter - first_iter
                signal = temp_signal[previous_iter:current_iter]
                # if NORMALIZER != '_nonorm':
                    # signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1)).ravel()
                stats_feature = temp_feature_extraction(signal, TEMP_SAMPLING_RATE) # Extract statistical features from extracted EDA features
                temp_stats_features.append(stats_feature)


    temp_stats_features = np.array(temp_stats_features) # Transform to numpy array format
    output_file_path = osp.join(get_dataset_stats_features_path(dp_manager, DATASET_NAME), f'{DATASET_NAME}_{DEVICE}_{SIGNAL_NAME}{NORMALIZER}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
    np.save(output_file_path, temp_stats_features)
