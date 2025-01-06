from typing import Dict

import pytorch_lightning as pl
import torch

from researchpkg.industry_classification.utils.experiment_utils import (
    ExperimentUtils,
)


class CustomModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
    """
    Custom Model Checkpoint Callback that allows to save the model
    according to a custom metric.
    """

    def __init__(
        self,
        experiment_dir,
        monitor,
        save_top_k,
        mode,
        auto_insert_metric_name,
        save_last,
        verbose=True,
        save_every_epoch_percentage=0.1,
    ):
        self.experiment_dir = experiment_dir
        self.save_every_epoch_percentage = save_every_epoch_percentage

        super().__init__(
            monitor=monitor,
            save_top_k=save_top_k,
            mode=mode,
            dirpath=experiment_dir,
            auto_insert_metric_name=auto_insert_metric_name,
            save_last=save_last,
            verbose=verbose,
        )

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.save_every_epoch_percentage == 0:
            return

        progress_percent = trainer.global_step / trainer.max_steps

        if (
            (progress_percent % self.save_every_epoch_percentage == 0)
            and (progress_percent != 0)
            and (progress_percent != 1)
        ):
            print(f"Saving model with progress percentage: {progress_percent}")
            trainer.save_checkpoint(self.experiment_dir / "last.ckpt")

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: "pl.Trainer",
        monitor_candidates: Dict[str, torch.Tensor],
    ) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)

        # Update the experiment
        # If multiple devices are used, the best model is saved in the last device
        if trainer.num_devices > 1:
            if trainer.model.device.index != trainer.num_devices - 1:
                return

        if self.best_model_path is not None and self.best_model_score == current:
            ExperimentUtils.uptate_experiment_best_model(
                self.experiment_dir,
                self.monitor,
                self.best_model_score.item(),
                trainer.current_epoch,
                self.best_model_path,
            )

        #Update the last epoch
        experiment_data = ExperimentUtils.load_experiment_data(self.experiment_dir)
        experiment_data["last_epoch"] = trainer.current_epoch
        ExperimentUtils.save_experiment_data(self.experiment_dir, experiment_data)
        