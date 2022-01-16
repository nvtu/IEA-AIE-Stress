import os.path as osp
import numpy as np

from feature_extraction.combine_stats_features import NORMALIZER
from .data_utils import *


class DatasetLoader:


    def __init__(self, dataset_name, device, physiological_signal_type, NORMALIZER = '', WINDOW_SIZE = 60, WINDOW_SHIFT = 1):
        self.dp_manager = get_data_path_manager()
        self.WINDOW_SIZE = WINDOW_SIZE 
        self.WINDOW_SHIFT = WINDOW_SHIFT
        self.NORMALIZER = NORMALIZER
        self.dataset, self.ground_truth, self.groups = self.load_dataset(dataset_name, device, physiological_signal_type=physiological_signal_type)


    def get_dataset_folder_path(self, dataset_name: str) -> str:
        dataset_folder_path = None
        if dataset_name == 'AffectiveROAD':
            dataset_folder_path = self.dp_manager.AffectiveROAD_stats_feature_path
        elif dataset_name == 'WESAD':
            dataset_folder_path = self.dp_manager.WESAD_stats_feature_path
        return dataset_folder_path


    def load_dataset(self, dataset_name: str, device: str, physiological_signal_type: str = ""):
        dataset = None
        ground_truth = None
        # Initialize dataset folder path
        dataset_folder_path = self.get_dataset_folder_path(dataset_name)
        # Initialize dataset file path
        dataset_file_path = osp.join(dataset_folder_path, f'{dataset_name}_{device}_{physiological_signal_type}{NORMALIZER}_{self.WINDOW_SIZE}_{self.WINDOW_SHIFT}.npy')
        # Initialize ground-truth file path
        ground_truth_file_path = osp.join(dataset_folder_path, f'{dataset_name}_{device}_labels_{self.WINDOW_SIZE}_{self.WINDOW_SHIFT}.npy')
        # Initialize group file path
        group_file_path = osp.join(dataset_folder_path, f'{dataset_name}_{device}_groups_{self.WINDOW_SIZE}_{self.WINDOW_SHIFT}.npy')

        # Load dataset, ground-truth, and groups
        dataset = np.load(dataset_file_path) # Load dataset
        ground_truth = np.load(ground_truth_file_path) # Load corresponding ground-truth
        groups = np.load(group_file_path) # Load corresponding user_id labels

        return dataset, ground_truth, groups  