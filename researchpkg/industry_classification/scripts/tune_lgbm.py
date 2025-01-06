import argparse
import datetime
import os
import sys

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))  # noqa
sys.path.append(dir_path)  # noqa
import numpy as np  # noqa
import ray  # noqa
import sklearn.metrics  # noqa
from ray import train, tune  # noqa
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback  # noqa
from ray.tune.search.hyperopt import HyperOptSearch  # noqa
from sklearn.model_selection import KFold  # noqa

from researchpkg.industry_classification.config import (  # noqa
    ROOT_DIR,
    SEC_CLEAN_DATA_DIR,
)
from researchpkg.industry_classification.dataset.sec_gbdt_dataset import (  # noqa
    SecGBDTDataset,
)
from researchpkg.industry_classification.dataset.utils import DatasetType  # noqa
from researchpkg.industry_classification.models.decision_trees.lgbm import (  # noqa; noqa
    LgbmForSicClassification,
)

EXPERIMENTS_DIR = os.path.join(
    ROOT_DIR,
    "logs/hyperparams_tuning_lgbm_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)
if not os.path.exists(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)

parser = argparse.ArgumentParser(
    prog="LGBM SIC classifier", description="Find  lgbm hyperparams"
)
parser.add_argument(
    "--sic_digits",
    default=1,
    type=int,
    help="The target features sic{sic_digits}. Ex: sic1 , sci2",
)


def train_kfold(config, dataset, sic_digits: int, n_jobs: int, num_folds=5):
    # 1. First instantiate the model with the hyperparams

    n_accounts = len(dataset.accounts_index)
    labels = list(
        sorted(dataset.registrants_index[f"sic{sic_digits}"].unique().tolist())
    )
    n_labels = len(labels)

    n_estimators = config["n_estimators"]
    num_leaves = config["num_leaves"]
    max_depth = config["max_depth"]
    learning_rate = config["learning_rate"]
    boosting_type = config["boosting_type"]
    reg_lambda = config["reg_lambda"]
    reg_alpha = config["reg_alpha"]
    seed = config["seed"]
    X, Y = dataset.X, dataset.Y
    all_ciks = dataset.get_all_ciks()

    model = LgbmForSicClassification(
        n_accounts=n_accounts,
        n_classes=n_labels,
        num_leaves=num_leaves,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        boosting_type=boosting_type,
        device="cpu",
        n_jobs=n_jobs,
        features_name=dataset.accounts_index["tag"].tolist(),
        class_names=labels,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        seed=seed,
    )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    unique_ciks = np.unique(all_ciks)

    scores = []
    for k, (train_ciks_index, val_ciks_index) in enumerate(kf.split(unique_ciks)):
        print(f"Training fold {k + 1}/{num_folds}")
        train_ciks = unique_ciks[train_ciks_index]
        val_ciks = unique_ciks[val_ciks_index]

        train_index = np.isin(all_ciks, train_ciks)
        val_index = np.isin(all_ciks, val_ciks)

        X_train = X[train_index]
        Y_train = Y[train_index]

        X_val = X[val_index]
        Y_val = Y[val_index]

        _, best_iteration = model.fit(X_train, Y_train, X_val, Y_val)
        y_pred_val = model.predict(X_val)

        best_score = model.metrics["mcc"](Y_val, y_pred_val)
        scores.append(best_score)

    # Report the average score
    average_score = np.mean(scores)
    train.report({"mcc": average_score})


# Define a function to train the model
def train_nokfold(config, train_dataset, val_dataset, sic_digits: int, n_jobs: int):
    """
    Train the lgm model on a given dataset.
    """

    n_accounts = len(train_dataset.accounts_index)
    labels = list(
        sorted(train_dataset.registrants_index[f"sic{sic_digits}"].unique().tolist())
    )
    n_labels = len(labels)

    n_estimators = config["n_estimators"]
    num_leaves = config["num_leaves"]
    max_depth = config["max_depth"]
    learning_rate = config["learning_rate"]
    boosting_type = config["boosting_type"]
    reg_lambda = config["reg_lambda"]
    reg_alpha = config["reg_alpha"]

    model = LgbmForSicClassification(
        n_accounts=n_accounts,
        n_classes=n_labels,
        num_leaves=num_leaves,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        boosting_type=boosting_type,
        device="cpu",
        n_jobs=n_jobs,
        features_name=train_dataset.accounts_index["tag"].tolist(),
        class_names=labels,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
    )

    experiment_name = f"{model.__name__}_no_gaap_sic{sic_digits}"
    experiment_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    x_val = val_dataset.X
    y_val = val_dataset.Y

    model.model.fit(
        X=train_dataset.X,
        y=train_dataset.Y,
        eval_set=[(x_val, y_val)],
        eval_names=["eval"],
        eval_metric=lambda y_true, y_pred: (
            "mcc",
            sklearn.metrics.matthews_corrcoef(y_true, y_pred.argmax(axis=1)),
            True,
        ),
        callbacks=[TuneReportCheckpointCallback({"mcc": "eval-mcc"}, "lightgbm.mdl")],
    )


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()

    sic_digits = args.sic_digits
    # Load the dataset

    dataset_dir = SEC_CLEAN_DATA_DIR
    # train_dataset = SecGBDTDataset(dataset_dir, DatasetType.TRAIN,sic_digits=sic_digits)
    # val_dataset = SecGBDTDataset(dataset_dir,DatasetType.VAL,sic_digits=sic_digits)

    # Entire dataset for kfold
    dataset = SecGBDTDataset(dataset_dir, DatasetType.ALL, sic_digits=sic_digits)

    # Initialize Ray
    ray.init()

    # Define the search space for Bayesian optimization
    search_space = {
        "n_estimators": 30,  # Adjust the range as needed
        "num_leaves": tune.randint(8, 1000),  # Adjust the range as needed
        "max_depth": tune.choice([-1, 2, 10, 20, 30]),  # Adjust the range as needed
        "boosting_type": tune.choice(["gbdt", "dart"]),
        "learning_rate": tune.loguniform(1e-8, 1e-1),
        "sic_digits": sic_digits,
        "reg_lambda": tune.loguniform(0.1, 1),
        "reg_alpha": tune.loguniform(0.1, 1),
        "seed": tune.randint(0, 1000),
    }

    # Configure the Ray Tune experiment with HyperOptSearch
    analysis = tune.run(
        # tune.with_parameters(train_no_kfold, train_dataset=train_dataset,val_dataset=val_dataset,sic_digits=sic_digits, n_jobs=60),
        tune.with_parameters(
            train_kfold, dataset=dataset, sic_digits=sic_digits, n_jobs=60
        ),
        config=search_space,
        num_samples=150,  # Number of hyperpar"gpu":0.1ameter combinations to try
        local_dir=EXPERIMENTS_DIR,  # Directory to store results
        resources_per_trial={"cpu": 60},  # Adjust resources as needed
        max_concurrent_trials=1,
        verbose=1,  # Set verbosity levelSearc
        search_alg=HyperOptSearch(metric="mcc", mode="max"),
    )

    # Print the best hyperparameters
    best_config = analysis.get_best_config(mode="max", metric="mcc")
    # Save the results to a pandas DataFrame
    analysis.results_df.to_csv(os.path.join(EXPERIMENTS_DIR, "results.csv"))

    print("Best Hyperparameters:", best_config)

    # Optionally, you can train the best model on the entire dataset here.

    # Shutdown Ray
    ray.shutdown()
