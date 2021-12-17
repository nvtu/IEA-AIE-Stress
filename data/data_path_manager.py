from dataclasses import dataclass


@dataclass
class DataPathManager:

    dataset_path: str
    WESAD_dataset_path: str
    WESAD_wrist_metadata_path: str
    AffectiveROAD_dataset_path: str
    