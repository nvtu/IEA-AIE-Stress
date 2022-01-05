# Add additional library
import sys, os

from numpy.ma.core import _MaskedUnaryOperation
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
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings('ignore')


EDA_SAMPLING_RATE = 4
DATASET_NAME = 'WESAD'
DEVICE = 'wrist'
SIGNAL_NAME = 'EDA'

WINDOW_SIZE = 60
WINDOW_SHIFT = 0.25

dp_manager = get_data_path_manager()
wesad_wrist_metadata = load_metadata(dp_manager.WESAD_wrist_metadata_path)
user_ids = sorted(os.listdir(dp_manager.WESAD_dataset_path))

swt_denoiser = SWT_Threshold_Denoiser()
eda_processor = EDA_Signal_Processor()

eda_stats_features = []
for user_id in user_ids:
    print(f"Extracting EDA features of user {user_id}")
    raw_signal = load_raw_signal(dp_manager, DATASET_NAME, user_id, DEVICE, SIGNAL_NAME)
    user_metadata = wesad_wrist_metadata[user_id]
    for i, label in enumerate(user_metadata['labels']):
        starting_index = user_metadata['eda']['starting_indices'][i]
        ending_index = user_metadata['eda']['ending_indices'][i]
        trim_indices = (starting_index, ending_index)
        trimmed_signal = get_trimmed_signal(raw_signal, trim_indices, lag = WINDOW_SIZE, sampling_rate=EDA_SAMPLING_RATE)
        len_signal = len(trimmed_signal)
        for j in tqdm(range(int(WINDOW_SIZE * EDA_SAMPLING_RATE), len_signal, int(WINDOW_SHIFT * EDA_SAMPLING_RATE))):
            k = j - WINDOW_SIZE * EDA_SAMPLING_RATE
            signal = swt_denoiser.denoise(trimmed_signal[k:j])
            signal = MinMaxScaler().fit_transform(signal.reshape(-1, 1)).ravel()
            stats_feature = eda_processor.eda_feature_extraction(signal, EDA_SAMPLING_RATE)
            eda_stats_features.append(stats_feature)

eda_stats_features = np.array(eda_stats_features)
output_file_path = osp.join(dp_manager.WESAD_stats_feature_path, f'{DATASET_NAME}_{DEVICE}_{SIGNAL_NAME}_{WINDOW_SIZE}_{WINDOW_SHIFT}.npy')
np.save(output_file_path, eda_stats_features)