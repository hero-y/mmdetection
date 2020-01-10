from .coco import CocoDataset
from .registry import DATASETS

@DATASETS.register_module
class CancerFinalDataset(CocoDataset):
    CLASSES = ('ASC-H','ASC-US','HSIL','LSIL','Candida','Trichomonas')