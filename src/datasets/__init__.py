from .base import BaseDataset, ClassificationDataset, VQADataset
from .lag import LAGDataset
from .ddr import DDRDataset
from .chexpert import CheXpertDataset
from .breakhis import BreakHisDataset
from .medical_cxr_vqa import MedicalCXRVQADataset
from .vqa_rad import VQARADDataset
from .pathvqa import PathVQADataset
from .pmc_vqa import PMCVQADataset

CLASSIFICATION_DATASETS = {
    "lag": LAGDataset,
    "ddr": DDRDataset,
    "chexpert": CheXpertDataset,
    "breakhis": BreakHisDataset,
}

VQA_DATASETS = {
    "medical_cxr_vqa": MedicalCXRVQADataset,
    "vqa_rad": VQARADDataset,
    "pathvqa": PathVQADataset,
    "pmc_vqa": PMCVQADataset,
}

ALL_DATASETS = {**CLASSIFICATION_DATASETS, **VQA_DATASETS}


def get_dataset(name: str, data_root: str, split: str = "test", **kwargs):
    if name not in ALL_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(ALL_DATASETS.keys())}")
    return ALL_DATASETS[name](data_root=data_root, split=split, **kwargs)
