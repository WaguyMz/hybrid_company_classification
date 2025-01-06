import os
from typing import List

import sklearn.metrics
from lightgbm import LGBMClassifier

from researchpkg.industry_classification.models.decision_trees.abstract_model import (
    AbstractDecisionTreeForSicClassification,
)


class LgbmForSicClassification(AbstractDecisionTreeForSicClassification):
    def __init__(
        self,
        n_accounts: int,
        n_classes: int,
        num_leaves: int,
        max_depth: int,
        n_estimators: int,
        learning_rate: float,
        n_jobs: int,
        features_name: List,
        class_names=None,
        device="cpu",
        boosting_type="gbdt",
        seed=42,
        reg_lambda=0.8,
        reg_alpha=0.8,
        class_weight=None,
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
        self.num_leaves = num_leaves
        self.boosting_type = boosting_type
        self.seed = seed
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.class_weight = class_weight or "balanced"
        self.build_model()

        self.hparams = {
            "n_accounts": n_accounts,
            "boosting_type": self.boosting_type,
            "n_classes": n_classes,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "class_names": class_names,
            "seed": seed,
            "n_jobs": n_jobs,
        }

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

        best_score = self.model.best_score_["eval"]["mcc"]

        # Check if best_score is nan
        if best_score != best_score:
            best_score = -1  # Set best_score to -1 if nan
            best_iteration = -1
        else:
            best_score = best_score.item()
            best_iteration = self.model.best_iteration_
        return best_score, best_iteration

    def build_model(self):
        self.model: LGBMClassifier = LGBMClassifier(
            boosting_type=self.boosting_type,
            objective="multiclass",
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            class_weight=self.class_weight,
            early_stopping_rounds=10,
            device=self.device,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            num_classes=self.n_classes,
            random_state=self.seed,
            metric="multi_logloss",
        )

        hparams_text = (
            f"rows{self.n_accounts}_"
            f"boost.{self.boosting_type}_"
            f"l{self.num_leaves}_"
            f"dpth{self.max_depth}_"
            f"est{self.n_estimators}_"
            f"j{self.n_jobs}_"
            f"lr{self.learning_rate}"
            f"seed{self.seed}"
        )

        self.__name__ = f"lgbm_classifier_{hparams_text}"

    def save_model(self, experiment_dir: str, step=None):
        """
        Saving the model
        :param experiment_dir: The experiment directory
        """
        # 3. Save the model.

        model_name = "model" + f"_{step}" if step is not None else "" + ".lgbm"

        model_path = os.path.join(experiment_dir, f"{model_name}")
        self.model.booster_.save_model(model_path)

        return model_path
