import os
from typing import Dict

import pytorch_lightning as pl
import torch
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from researchpkg.industry_classification.models.transformers.text_llm_sft import (
    TextLLMForInstructionSFT,
)
from researchpkg.industry_classification.utils.experiment_utils import (
    ExperimentUtils,
)
import gc

class SftLoggerCallback(TrainerCallback):
    """
    Custom Model Checkpoint Callback that allows to save the model
    according to a custom metric.
    """

    def __init__(
        self,
        model: TextLLMForInstructionSFT,
        experiment_dir,
        train_dataset,
        val_dataset,
        batch_size=1,
    ):
        self.experiment_dir = experiment_dir
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        current_exp = ExperimentUtils.get_best_model(
            os.path.basename(self.experiment_dir), os.path.dirname(self.experiment_dir)
        )
        if current_exp is None:
            best_mcc = 0
        else:
            best_mcc = current_exp["mcc"]
        metrics = self.model.run_prediction(
            self.val_dataset, self.experiment_dir, mode="val", epoch=state.epoch,
            batch_size=self.batch_size
        )
        mcc = metrics["val_mcc"]
        if mcc >= best_mcc:
            ExperimentUtils.uptate_experiment_best_model(
                self.experiment_dir, "mcc", mcc, state.epoch, None)
            

            ExperimentUtils.update_experiment_results(self.experiment_dir, metrics)

        # Update last epoch number
        experiment = ExperimentUtils.load_experiment_data(self.experiment_dir)
        experiment["last_epoch"] = state.epoch
        ExperimentUtils.save_experiment_data(self.experiment_dir, experiment)