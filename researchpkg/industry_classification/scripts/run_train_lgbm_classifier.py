import argparse
import multiprocessing
import os
import shutil
import sys

import numpy as np

from researchpkg.industry_classification.preprocessing.sec_preprocessing_utils import (
    COMMON_PERCENTAGES_15,
)

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(dir_path)

import torch.cuda

from researchpkg.industry_classification.config import (
    LOGS_DIR,
    MAX_CORE_USAGE,
    SEC_ROOT_DATA_DIR,
)
from researchpkg.industry_classification.dataset.sec_datamodule import (
    NormalizationType,
    SecDataset,
)
from researchpkg.industry_classification.dataset.sec_gbdt_dataset import (
    SecGBDTDataset,
)
from researchpkg.industry_classification.dataset.utils import DatasetType
from researchpkg.industry_classification.models.decision_trees.lgbm import (
    LgbmForSicClassification,
)
from researchpkg.industry_classification.utils.experiment_utils import (
    ExperimentUtils,
)
from researchpkg.industry_classification.utils.sics_loader import load_sic_codes

parser = argparse.ArgumentParser(
    prog="LGBM SIC classifier", description="Run something"
)
parser.add_argument(
    "-g",
    "--global_exp_name",
    help="The global experiment name to use. Should match one of existing generated datasets.\
        Example : _count30_sic1agg, _count30_sic1agg_with12ratios",
)
parser.add_argument(
    "-e", "--n_estimators", type=int, default=100, help="Number of lgbm estimators"
)
parser.add_argument(
    "-nl",
    "--num_leaves",
    type=int,
    default=60,
    help="Number of leaves for the base learner",
)

parser.add_argument(
    "-md",
    "--max_depth",
    type=int,
    default=60,
    help="Max depth of the DT. -1 means no limit",
)
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=0.5, help="The learning rate"
)

parser.add_argument(
    "-a",
    "--accelerator",
    default="cpu" if not torch.cuda.is_available() else "cuda",
    help="The accelerator to use",
)
parser.add_argument("-x", "--experiment_name", default="sic_prediction")

parser.add_argument(
    "-j",
    "--n_jobs",
    type=int,
    default=max(1, min(multiprocessing.cpu_count() - 2, MAX_CORE_USAGE)),
    help="The number of jobs of the GBDT algorithm",
)
parser.add_argument(
    "-r", "--reset", action="store_true", help="Reset the experiment from the begining"
)
parser.add_argument("--no_gaap", action="store_true", help="Use dataset without gaap")
parser.add_argument("-b", "--boosting_type", default="gbdt", help="Boosting algo")
parser.add_argument(
    "--sic_digits",
    default=1,
    type=int,
    help="The target features sic{sic_digits}. Ex: sic1 , sci2",
)
parser.add_argument("--seed", default=42, type=int, help="The seed for the experiment")
parser.add_argument("--normalization", default="local", help="The normalization type")


def train(
    global_exp_name: str,
    n_estimators: int,
    num_leaves: int,
    max_depth: int,
    learning_rate: float,
    accelerator: str,
    experiment_name: str,
    reset: bool,
    n_jobs: int,
    sic_digits: int,
    no_gaap: bool,
    boosting_type: str,
    seed: int,
    normalization: str,
):
    """
    Run the training of the balance sheet classification model.
    """
    ExperimentUtils.check_global_experiment_name(global_exp_name)
    dataset_dir = os.path.join(SEC_ROOT_DATA_DIR, f"{global_exp_name}")
    experiments_dir = os.path.join(LOGS_DIR, f"experiments_{global_exp_name}")

    assert (
        normalization in SecDataset.NORMALIZATION_TYPES
    ), f"Normalization type {normalization} not supported"

    # 1. Load the dataset.
    train_dataset = SecGBDTDataset(
        dataset_dir,
        DatasetType.TRAIN,
        sic_digits=sic_digits,
        normalization_type=normalization,
        max_tag_depth=max_depth
    )
    val_dataset = SecGBDTDataset(
        dataset_dir,
        DatasetType.VAL,
        sic_digits=sic_digits,
        normalization_type=normalization,
        max_tag_depth=max_depth
    )
    test_dataset = SecGBDTDataset(
        dataset_dir,
        DatasetType.TEST,
        sic_digits=sic_digits,
        normalization_type=normalization,
        max_tag_depth=max_depth
    )
    # dataset = SecGBDTDataset(dataset_dir, DatasetType.ALL, sic_digits=sic_digits)

    accounts_index = train_dataset.accounts_index

    sic_reverse_index = {v: k for k, v in train_dataset.sic_id_index.items()}
    labels = np.unique(train_dataset.Y).tolist()
    labels = [sic_reverse_index[l] for l in labels]
    print("labels", labels)
    sic_code_df = load_sic_codes()[["sic", "industry_title"]]
    labels = [
        sic_code_df[sic_code_df["sic"] == l]["industry_title"].values[0] for l in labels
    ]
    n_labels = len(labels)

    if normalization == NormalizationType.COMMON_PERCENTAGES:
        features_name = ["cp_" + x for x in COMMON_PERCENTAGES_15]
    elif normalization == NormalizationType.COMMON_PERCENTAGES_WITH_BASE_FEATURES:
        features_name = accounts_index["tag"].tolist() + [
            "cp_" + x for x in COMMON_PERCENTAGES_15
        ]
    elif normalization == NormalizationType.LOCAL_WITH_RAW:
        features_name = accounts_index["tag"].tolist()
        features_name = features_name + [f"{x}_norm" for x in features_name]
    else:
        features_name = accounts_index["tag"].tolist()

    print("Number of accounts:", len(features_name))
    print(f"Number of labels (sic{sic_digits}):", n_labels)

    X_train, Y_train = train_dataset.X, train_dataset.Y
    X_val, Y_val = val_dataset.X, val_dataset.Y
    X_test, Y_test = test_dataset.X, test_dataset.Y

    # Ensure all y_val labels appears at least once in Y_train
    all_y_train_labels = np.unique(Y_train)
    index = np.isin(Y_val, all_y_train_labels)
    X_val = X_val[index]
    Y_val = Y_val[index]

    index_test = np.isin(Y_test, all_y_train_labels)
    X_test = X_test[index_test]
    Y_test = Y_test[index_test]


    # Compute class weights
    # class_weights = SecDataset.calculate_class_weights(Y_train.tolist(),beta=0.1)
    # class_weights = SecDataset.calculate_class_weights(Y_train.tolist())

    # 2. Load the model.
    model = LgbmForSicClassification(
        n_accounts=len(features_name),
        n_classes=n_labels,
        num_leaves=num_leaves,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        n_jobs=n_jobs,
        features_name=features_name,
        class_names=labels,
        boosting_type=boosting_type,
        seed=seed,
        class_weight="balanced",
    )
    
        
    experiment_name = f"{model.__name__}{'_no_gaap' if no_gaap else ''}_{experiment_name}_sic{sic_digits}_balanced"
    experiment_name = f"{experiment_name}_scaling.{normalization}"
    
    if max_depth:
        experiment_name = f"{experiment_name}_max_depth{max_depth}"

    # 3. Trainer
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    if reset and os.path.exists(experiment_dir):
        print("Reset experiment")
        shutil.rmtree(experiment_dir)

    if not ExperimentUtils.check_experiment(experiment_dir):
        # 4. Initialize the experiment
        ExperimentUtils.initialize_experiment(
            experiment_dir,
            dataset_dir,
            model.hparams,
            training_config={
                "num_jobs": n_jobs,
                "learning_rate": learning_rate,
                "seed": seed,
                "device": accelerator,
                "ngpus": torch.cuda.device_count() if accelerator == "cuda" else 0,
            },
        )

    # model.train_top_k(X_train, Y_train, X_val, Y_val, experiment_dir=experiment_dir,
    #             accelerator=accelerator,top_k=3
    #             )

    model.train(
        X_train,
        Y_train,
        X_val,
        Y_val,
        experiment_dir=experiment_dir,
        accelerator=accelerator,
    )
    
    model.test(X_test, Y_test, experiment_dir=experiment_dir)

    # X, Y = dataset.X, dataset.Y
    # all_ciks = dataset.get_all_ciks()

    # model.train_kfold(X,Y,all_ciks, experiment_dir=experiment_dir,\
    #                    accelerator=accelerator,num_folds=N_FOLDS,seed=seed)


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    train(**vars(args))
