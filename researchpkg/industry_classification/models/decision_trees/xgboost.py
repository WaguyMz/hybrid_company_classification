import json
import os
from typing import List

import sklearn.metrics
import xgboost as xgb

from researchpkg.industry_classification.models.decision_trees.abstract_model import (
    AbstractDecisionTreeForSicClassification,
)


class XGBoostForSicClassification(AbstractDecisionTreeForSicClassification):
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
        seed=42,
    ):
        super().__init__(
            n_accounts=n_accounts,
            n_classes=n_classes,
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            features_name=features_name,
            class_names=class_names,
            device=device,
        )
        self.seed = seed
        self.build_model()

        self.hparams = {
            "n_accounts": n_accounts,
            "n_classes": n_classes,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "n_jobs": n_jobs,
            "class_names": class_names,
            "seed": seed,
        }

    def build_model(self):
        self.model = xgb.XGBClassifier(
            objective="multi:softprob",
            max_depth=self.max_depth,
            num_class=self.n_classes,
            enable_categorical=True,
            device=self.device,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )

        hparams_text = (
            f"rows{self.n_accounts}_"
            f"dpth{self.max_depth}_"
            f"est{self.n_estimators}_"
            f"j{self.n_jobs}_"
            f"lr{self.learning_rate}"
            f"seed{self.seed}"
        )

        self.__name__ = f"xgboost_classifier_{hparams_text}"

    def save_model(self, experiment_dir: str):
        # 4. Save the model.

        self.model.save_model(os.path.join(experiment_dir, "model.xgboost"))

    def fit(self, x_train, y_train, x_val, y_val):
        # Fit the train dataset.
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_names=["eval"],
            eval_metric=lambda y_true, y_pred: (
                "mcc",
                sklearn.metrics.matthews_corrcoef(y_true, y_pred.argmax(axis=1)),
                True,
            ),
            feature_name=self.features_name,
        )

        # COmpute  the best mcc score on train_dataset

        best_score = self.model.best_score_["eval"]["mcc"].item()
        best_iteration = self.model.best_iteration_
        return best_score, best_iteration
