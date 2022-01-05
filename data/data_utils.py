from data_path_manager import DataPathManager
import configparser
from typing import List
import os, os.path as osp
from pathlib import Path
import json
import pickle


def get_data_path_manager():
    """
    Read data path config.ini file and create a DataPathManager for later usage
    """

    config_file_path = str(Path(__file__).parent.parent / 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    data_path_obj = config['DATA_PATH']
    dataset_path = data_path_obj['dataset_path']
    WESAD_dataset_path = data_path_obj['wesad_dataset_path']
    WESAD_wrist_metadata_path = osp.join(dataset_path, 'wesad_wrist_metadata.json')
    WESAD_stats_feature_path = data_path_obj['WESAD_stats_feature_path']
    AffectiveROAD_dataset_path = data_path_obj['affectiveroad_dataset_path']

    dp_manager = DataPathManager(dataset_path, WESAD_dataset_path, WESAD_wrist_metadata_path, WESAD_stats_feature_path, AffectiveROAD_dataset_path)
    return dp_manager


def load_metadata(metadata_file_path: str):
    """
    Load json metadata which contains information about the labels, 
    starting indices, and ending indices of each session in the experiment
    """

    metadata = json.load(open(metadata_file_path, 'r'))
    return metadata


def load_raw_signal(dp_manager: DataPathManager, dataset_name: str, user_id: str, device: str, signal_name: str):
    raw_signal = None
    if dataset_name == 'WESAD':
        raw_signal_file_path = osp.join(dp_manager.WESAD_dataset_path, user_id, f'{user_id}.pkl')
        data = pickle.load(open(raw_signal_file_path, 'rb'), encoding = 'bytes')
        raw_signal = data[b'signal'][device.encode('ascii')][signal_name.encode('ascii')].ravel()
    return raw_signal


def get_trimmed_signal(raw_signal, trim_indices: List[int], lag: int, sampling_rate: int):
    if len(trim_indices) < 2: 
        raise ValueError('trim_indices must contains two values: starting index and ending index!!!')
    starting_index, ending_index = trim_indices
    starting_index = max(0, starting_index - lag * sampling_rate)
    trimmed_signal = raw_signal[starting_index:ending_index]
    return trimmed_signal 


def get_dataset_path(dp_manager: DataPathManager, name: str):
    if name == 'WESAD':
        return dp_manager.WESAD_dataset_path
    elif name == 'AffectiveROAD':
        return dp_manager.AffectiveROAD_dataset_path
    return None
