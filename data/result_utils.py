import os
import os.path as osp
import pandas as pd

from feature_extraction.combine_stats_features import NORMALIZER
from .data_utils import get_data_path_manager


class ResultUtils:

    def __init__(self, dataset_name: str, device: str, physiological_signal_type: str, NORMALIZER = '', WINDOW_SHIFT: int = 1, WINDOW_SIZE: int = 60):
        self.dp_manager = get_data_path_manager()
        self.dataset_name = dataset_name
        self.device = device
        self.physiological_signal_type = physiological_signal_type
        self.WINDOW_SHIFT = WINDOW_SHIFT
        self.WINDOW_SIZE = WINDOW_SIZE
        self.NORMALIZER = NORMALIZER


    def get_output_folder_path(self) -> str:
        # Get output_folder_path for a specific dataset
        if self.dataset_name == 'WESAD':
            output_folder_path = osp.join(self.dp_manager.WESAD_result_path, f'{self.dataset_name}_{self.device}')
        elif self.dataset_name == 'AffectiveROAD':
            output_folder_path = osp.join(self.dp_manager.AffectiveROAD_result_path, f'{self.dataset_name}_{self.device}')
        # Create the output folder if it does not exist
        if not osp.exists(output_folder_path):
            os.makedirs(output_folder_path)
        return output_folder_path


    def dump_result_to_csv(self, results, detection_strategy: str, detector_type: str):
        output_folder_path = osp.join(self.get_output_folder_path(), detector_type)
        # Create the folder if it does not exist
        if not osp.exists(output_folder_path):
            os.makedirs(output_folder_path)
        # Get output_file_path
        output_file_path = osp.join(output_folder_path, f'{self.dataset_name}_{self.device}_{detection_strategy}_{self.physiological_signal_type}{self.NORMALIZER}_{self.WINDOW_SIZE}_{self.WINDOW_SHIFT}.csv')
        # Generate DataFrame to save to csv format
        df = pd.DataFrame.from_dict(results)
        df.to_csv(output_file_path, index=False)   

