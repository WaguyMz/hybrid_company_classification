from enum import Enum


class DatasetType(str,Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ALL = "all"
