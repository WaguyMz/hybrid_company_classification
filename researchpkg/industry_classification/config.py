import os

GLOBAL_SEED=42
import torch
torch.manual_seed(GLOBAL_SEED)
import numpy as np
np.random.seed(GLOBAL_SEED)

# 1. Global constants.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project's root directory

TAGS_PERSHEET_COUNT_THRESHOLD = 30  # Min number of tags per sheet to consider it
GLOBAL_EXP_INDEX = {
    "count30_sic1agg_2023": "Balance sheet tags only few shot :2023 only",
}

# 2 Dataset constants
DATA_START_QTR = "2023q1"
DATA_END_QTR = "2024q1"

SEC_ROOT_DATA_DIR = os.path.join(ROOT_DIR, "data", "sec_data_v2")
SEC_RAW_DATA_DIR = os.path.join(SEC_ROOT_DATA_DIR, "raw")
SEC_TAX_DATA_DIR = os.path.join(SEC_ROOT_DATA_DIR, "taxonomies")

SEC_FILENAMES = ["sub", "pre", "tag", "num"]  # Names of the files in the SEC rawdataset

# 3. Logs directory
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# 4. Taxonomy
SEC_TAX = "us-gaap"
SEC_TAX_VERSION = "2023"
SEC_TAX_MIN_TAGS_DEPTH: int = 0
SEC_TAX_MAX_TAGS_DEPTH: int = 100000

# 5. Datasets settings
SEC_MIN_YEAR = 2008
SEC_MAX_YEAR = 2024
MAX_SUB_PER_CIK = (
    1e10  # Max number of submissions for a single company (10000= infinite)
)
MIN_SUB_PER_CIK = 1  # min number of submissions to accept a cik
TOP_K_TAGS = 1e5  # Top k tags to consider

SIC1_EXCLUDED = [0, 9]  # Sics 1 to exclude
# SIC1_EXCLUDED = [0,10, 15, 50, 52,9]

# 6. Sics
RAW_SIC_FILE = os.path.join(SEC_ROOT_DATA_DIR, "sics.csv")

# 6. TRAIN VAL
SEC_TRAIN_RATIO = 70
SEC_VALIDATION_RATIO = 10
SEC_TEST_RATIO = 20

MAX_CORE_USAGE = 32

# 7. Tensorboard setting1
CONFUSION_MATRIX_PLOT_RATE = 1  # Plot the confusion matrix each  nth epoch

# 8 Adding income statement tags.
TOP_K_INCOME_STATEMENT_TAGS = 1000

RANDOM_SEED = 42
