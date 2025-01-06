import os
from abc import ABC
from typing import List

import numpy as np
import pyaml
import sklearn.metrics
from torch.utils.tensorboard import SummaryWriter

from researchpkg.industry_classification.models.utils import NN_Utils
from researchpkg.industry_classification.utils.experiment_utils import (
    ExperimentUtils,
)


class AbstractDecisionTreeForSicClassification(ABC):
    """
    Abstract decision tree model for SIC classification.

    """

    def __init__(
        self,
        n_accounts: int,
        n_classes: int,
        max_depth: int,
        n_estimators: int,
        learning_rate: float,
        n_jobs: int,
        features_name: List,
        class_names=None,
        device="cpu",
    ):
        self.n_accounts = n_accounts
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.features_name = features_name
        self.class_names = class_names
        self.device = device

        if self.class_names is None:
            self.class_names = [str(i + 1) for i in range(self.n_classes)]
        else:
            assert len(self.class_names) == self.n_classes

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        # Metrics
        self.metrics = {
            "accuracy": sklearn.metrics.accuracy_score,
            "recall_macro": lambda y_true, y_pred: sklearn.metrics.recall_score(
                y_true, y_pred, average="macro"
            ),
            "precision_macro": lambda y_true, y_pred: sklearn.metrics.precision_score(
                y_true, y_pred, average="macro"
            ),
            "f1_macro": lambda y_true, y_pred: sklearn.metrics.f1_score(
                y_true, y_pred, average="macro"
            ),
            "recall_weighted": lambda y_true, y_pred: sklearn.metrics.recall_score(
                y_true, y_pred, average="weighted"
            ),
            "precision_weigthed": lambda y_true, y_pred: sklearn.metrics.precision_score(
                y_true, y_pred, average="weighted"
            ),
            "f1_weighted": lambda y_true, y_pred: sklearn.metrics.f1_score(
                y_true, y_pred, average="weighted"
            ),
            "mcc": sklearn.metrics.matthews_corrcoef,
            "mrr": sklearn.metrics.label_ranking_average_precision_score,
        }

        self.validation_step_outputs = []
        self.best_val_loss = 1e10

        self.datamodule_for_precomputing = None

        self.model = None

        self.hparams = {}

    def fit(self, X_train, Y_train, X_val, Y_val):
        raise NotImplementedError

    def predict(self, X_val):
        # Run prediction on validation dataset
        y_pred = self.model.predict(X_val)
        return y_pred

    def predict_proba(self, X_val):
        # Run prediction on validation dataset
        y_pred = self.model.predict(X_val, raw_score=True)
        return y_pred

    def compute_confusion_matrix(
        self, all_y_true, all_y_pred, experiment_dir, suffix="", step=None
    ):
        # Compute and plot the confusion matrix at the end of the training.

        cm_plot = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, self.class_names
        )

        cm_plot_normalized = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, self.class_names, normalize=True
        )

        cm_writer = SummaryWriter(log_dir=experiment_dir)
        cm_writer.add_figure(f"Confusion Matrix - Validation - {suffix}", cm_plot, step)
        cm_writer.add_figure(
            f"Confusion Matrix Normalized - Validation-{suffix}",
            cm_plot_normalized,
            step,
        )

    def train(self, X_train, Y_train, X_val, Y_val, experiment_dir, accelerator="cpu"):
        # 1. First save hyperparams
        with open(os.path.join(experiment_dir, "hparams.yaml"), "w") as f:
            pyaml.dump(self.hparams, f)

        # 2. Run training
        _, best_iteration = self.fit(X_train, Y_train, X_val, Y_val)

        y_pred_train_prob = self.predict_proba(
            X_train
        )  # TODO remove this asap. not efficient
        y_pred_train = np.argmax(y_pred_train_prob, axis=1)

        y_pred_val_prob = self.predict_proba(X_val)
        y_pred_val = np.argmax(y_pred_val_prob, axis=1)

        # 4. Compute confusion matrix on validation dataset
        self.compute_confusion_matrix(Y_val, y_pred_val, experiment_dir)

        # 5. Compute metrics.
        results = {}

        val_results = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "mrr":
                Y_val_indicators = np.zeros((Y_val.shape[0], self.n_classes))
                Y_val_indicators[np.arange(Y_val.shape[0]), Y_val] = 1

                print("Shapes : ", Y_val_indicators.shape, y_pred_val_prob.shape)

                val_results[f"val_{metric_name}"] = metric(
                    Y_val_indicators, y_pred_val_prob
                ).item()
            else:
                val_results[f"val_{metric_name}"] = metric(y_pred_val, Y_val).item()
        results = dict(results, **val_results)

        best_score = val_results["val_mcc"]

        train_results = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "mrr":
                Y_train_indicators = np.zeros((Y_train.shape[0], self.n_classes))
                Y_train_indicators[np.arange(Y_train.shape[0]), Y_train] = 1

                train_results[f"train_{metric_name}"] = metric(
                    Y_train_indicators, y_pred_train_prob
                ).item()
            else:
                train_results[f"train_{metric_name}"] = metric(y_pred_train, Y_train)
        results = dict(results, **train_results)

        # 6. Save results
        with open(os.path.join(experiment_dir, "results.yaml"), "w") as f:
            pyaml.dump(results, f)

        # 7. Save the model.
        self.save_model(experiment_dir)

        ExperimentUtils.uptate_experiment_best_model(
            experiment_dir, "val_mcc", val_results["val_mcc"], best_iteration, None
        )

        print("Best mcc score :", best_score)

        return results["val_mcc"]

    def train_top_k(
        self, X_train, Y_train, X_val, Y_val, experiment_dir, accelerator="cpu", top_k=3
    ):
        # 1. First save hyperparams
        with open(os.path.join(experiment_dir, "hparams.yaml"), "w") as f:
            pyaml.dump(self.hparams, f)

        # 2. Run training
        _, best_iteration = self.fit(X_train, Y_train, X_val, Y_val)

        y_pred_train = self.predict(X_train)  # TODO remove this asap. not efficient

        y_pred_proba_val = self.predict_proba(X_val)

        y_pred_val = np.argsort(y_pred_proba_val, axis=1)[:, -1:]

        # 3. Compute metrics.
        results = {}

        val_results = {}
        for metric_name, metric in self.metrics.items():
            val_results[f"val_{metric_name}"] = metric(y_pred_val, Y_val)
        results = dict(results, **val_results)

        best_score = val_results["val_mcc"]

        train_results = {}
        for metric_name, metric in self.metrics.items():
            train_results[f"train_{metric_name}"] = metric(y_pred_train, Y_train)
        results = dict(results, **train_results)

        # 4. Compute confusion matrix on validation dataset
        self.compute_confusion_matrix(Y_val, y_pred_val, experiment_dir, suffix="_top1")

        # 5. Compute top_k prediction confusion matrix on validation dataset
        for k in range(2, top_k + 1):
            y_pred_val = np.argsort(y_pred_proba_val, axis=1)[:, -k]
            self.compute_confusion_matrix(
                Y_val, y_pred_val, experiment_dir, suffix=f"_top{k}"
            )

        # 6. Save results
        with open(os.path.join(experiment_dir, "results.yaml"), "w") as f:
            pyaml.dump(results, f)

        # 7. Save the model.
        self.save_model(experiment_dir)

        ExperimentUtils.uptate_experiment_best_model(
            experiment_dir, "val_mcc", val_results["val_mcc"], best_iteration, None
        )

        print("Best mcc score :", best_score)

        return results["val_mcc"]

    def train_kfold(
        self, X, Y, all_ciks, experiment_dir, num_folds=5, accelerator="cpu", seed=42
    ):
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

        unique_ciks = np.unique(all_ciks)
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

            _, best_iteration = self.fit(X_train, Y_train, X_val, Y_val)

            # TODO remove this asap. not efficient
            y_pred_train = self.predict(X_train)

            y_pred_val = self.predict(X_val)

            # 4. Compute confusion matrix on validation dataset
            self.compute_confusion_matrix(
                Y_val, y_pred_val, experiment_dir, suffix=f"_fold{k}"
            )

            # 5. Compute metrics.
            results = {}

            val_results = {}
            for metric_name, metric in self.metrics.items():
                val_results[f"val_{metric_name}"] = metric(y_pred_val, Y_val)
            results = dict(results, **val_results)

            best_score = val_results["val_mcc"]

            train_results = {}
            for metric_name, metric in self.metrics.items():
                train_results[f"train_{metric_name}"] = metric(y_pred_train, Y_train)
            results = dict(results, **train_results)

            # 6. Save results
            with open(os.path.join(experiment_dir, f"results_fold_{k}.yaml"), "w") as f:
                pyaml.dump(results, f)

            ExperimentUtils.uptate_experiment_best_model(
                experiment_dir,
                f"val_mcc_fold_{k}",
                val_results["val_mcc"],
                best_iteration,
                None,
            )

            print(f"Best mcc score fold {k}:", best_score, end="\n\n")

    def save_model(self, experiment_dir: str):
        raise NotImplementedError

    def test(self, X_test, Y_test, experiment_dir, suffix=""):
        y_pred_test = self.predict(X_test)

        # 4. Compute confusion matrix on validation dataset
        self.compute_confusion_matrix(
            Y_test, y_pred_test, experiment_dir, suffix="test"
        )

        # 5. Compute metrics.
        results = {}

        test_results = {}
        for metric_name, metric in self.metrics.items():
            if metric_name == "mrr":
                #print("Shape of Y_test", Y_test.shape, "Min y test", Y_test.min)
                y_pred_test_prob = self.predict_proba(X_test)
                Y_test_indicators = np.zeros((Y_test.shape[0], self.n_classes))
                Y_test_indicators[np.arange(Y_test.shape[0]), Y_test] = 1

                test_results[f"test_{metric_name}"] = metric(
                    Y_test_indicators, y_pred_test_prob
                ).item()
            else:
                test_results[f"test_{metric_name}"] = metric(y_pred_test, Y_test)

        results = dict(results, **test_results)

        # 6. Save results
        with open(os.path.join(experiment_dir, f"results_test{suffix}.yaml"), "w") as f:
            pyaml.dump(results, f)

        print("Test results : ", results)

        return results
