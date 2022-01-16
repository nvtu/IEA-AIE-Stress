from dataclasses import dataclass


@dataclass
class DataPathManager:

    dataset_path: str
    WESAD_dataset_path: str
    WESAD_wrist_metadata_path: str
    WESAD_stats_feature_path: str
    AffectiveROAD_dataset_path: str
    AffectiveROAD_stats_feature_path: str
    WESAD_result_path: str
    AffectiveROAD_result_path: str
    WESAD_model_path: str
    AffectiveROAD_model_path: str
    