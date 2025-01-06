from typing import Dict, Union

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    Accuracy,
    F1Score,
    MatthewsCorrCoef,
    Precision,
    Recall,
)
from transformers import BertModel, DistilBertModel, GPT2Model

from researchpkg.industry_classification.models.transformers.pretrained_index import (
    PretrainedTrf,
)
from researchpkg.industry_classification.models.utils import NN_Utils
from researchpkg.industry_classification.utils.experiment_utils import (
    ExperimentUtils,
)


class TextTransformerForClassification(pl.LightningModule):
    def __init__(
        self,
        n_accounts: int,
        pretrained_model=PretrainedTrf.BERT_TINY,
        mlp_hidden_dim=1024,
        n_classes=10,
        class_names=None,
        trf_trainable_layers=1,
        learning_rate=1e-3,
        dropout_rate=0.1,
        build_on_init=True,
    ):
        """
        Initialize the model.
        :param n_accounts: Number of accounts in the dataset.
        :param pretrained_model: Pretrained model to use. See PretrainedTrf enum.
        :param dim_net_change: Dimension of the net change.Default to 1.
        """

        super().__init__()
        self.n_accounts = n_accounts
        self.n_classes = n_classes
        self.class_names = class_names
        self.is_encoder = True

        if PretrainedTrf.is_gpt(pretrained_model):
            # GPT2 is the only decoder model used for now
            # Eos token will be used for classification
            self.is_encoder = False
        else:
            # All other models are encoders
            # CLS token will be used for classification
            self.is_encoder = True

        if self.class_names is None:
            self.class_names = [str(i + 1) for i in range(self.n_classes)]
        else:
            assert len(self.class_names) == self.n_classes

        self.pretrained_model = pretrained_model
        self.trf_trainable_layers = trf_trainable_layers
        self.learning_rate = learning_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_rate = dropout_rate
        # Loss
        self.criterion = nn.NLLLoss(reduction="none")

        # Metrics
        self.metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=self.n_classes),
            "recall": Recall(task="multiclass", num_classes=self.n_classes),
            "precision": Precision(task="multiclass", num_classes=self.n_classes),
            "f1": F1Score(task="multiclass", num_classes=self.n_classes),
            "f1_macro": F1Score(
                task="multiclass", num_classes=self.n_classes, average="macro"
            ),
            "mcc": MatthewsCorrCoef(task="multiclass", num_classes=self.n_classes),
            "mrr": sklearn.metrics.label_ranking_average_precision_score,
        }

        # Save hyperparams
        self.save_hyperparameters()

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_val_mcc = 0

        if build_on_init:
            self.build_model()

    def build_model(self):
        # 1. Text Transformer to encode description
        self.text_transformer, self.tokenizer = PretrainedTrf.load(
            self.pretrained_model
        )

        # Freeze transformer layers

        # Bert or distilbert

        if (
            type(self.text_transformer) == BertModel
            or type(self.text_transformer) == DistilBertModel
        ):
            # . Freeze all embeddings.
            for param in self.text_transformer.embeddings.parameters():
                param.requires_grad = False

            if type(self.text_transformer) == BertModel:
                layers = self.text_transformer.encoder.layer
            else:
                # Distilbert
                layers = self.text_transformer.transformer.layer  # type: ignore

            n_layers = len(layers)  # type: ignore
            if self.trf_trainable_layers < n_layers and self.trf_trainable_layers > 0:
                for layer in layers[
                    : n_layers - self.trf_trainable_layers
                ]:  # type: ignore
                    for param in layer.parameters():
                        param.requires_grad = False

        elif PretrainedTrf.is_gpt(self.pretrained_model):
            # Freeze non trainable layers
            n_layers = self.text_transformer.config.n_layer
            if self.trf_trainable_layers < n_layers and self.trf_trainable_layers > 0:
                for layer in self.text_transformer.h[
                    : n_layers - self.trf_trainable_layers
                ]:
                    for param in layer.parameters():
                        param.requires_grad = False

            # Make embedding trainable
            # for name, param in self.text_transformer.named_parameters():
            #     if "wte" in name or "wpe" in name:
            #         param.requires_grad = True

        # 2. Classification MLP
        self.classifier = nn.Sequential(
            # nn.Linear(self.text_transformer.config.hidden_size,
            #             self.mlp_hidden_dim),
            #             nn.GELU(),
            #             nn.Dropout(self.dropout_rate),
            # nn.Linear(self.mlp_hidden_dim, self.n_classes),
            nn.Linear(self.text_transformer.config.hidden_size, self.n_classes),
            nn.LogSoftmax(dim=1),
        )

        self.hparams_text = (
            f"{self.pretrained_model}_nt_{self.trf_trainable_layers}"
            f"_dp_{self.dropout_rate}"
        )

        self.__name__ = f"texttransformer_encoder_{self.hparams_text}"

    def forward(
        self,
        input: torch.Tensor,
        input_attn_mask: torch.Tensor,
        sample_idx=None,
        type="train",
        return_dict=False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        :param input_desc: Tokenized accounts descriptions. Shape [bs, n_accounts, seq_len]
        :param input_net_change: Accounts net changes Shape [bs, n_accounts, 1]
        :param input_attn_mask: Mask for account spositions. Shape [bs, n_accounts, seq_len]
        :param sample_idx: List of sample idx for using precomputed text embeddings.
        :param mode: train, val or text
        :param return_dict: Whether to return a dict with all outputs or just the logits.
        :return  : logits if return_dict is False, else a dict with all outputs.
        """

        # 1. Encode text
        x_enc = self.text_transformer(input_ids=input, attention_mask=input_attn_mask)[
            0
        ]

        # 2. Take the embedding for classification
        if self.is_encoder:
            # CLS token
            x_enc = x_enc[:, 0, :]
        else:
            # EOS token
            x_enc = x_enc[:, -1, :]

        # 3. Classify
        yhat = self.classifier(x_enc)

        if return_dict:
            return {"logits": yhat, "x_enc": x_enc}

        return yhat

    def compute_loss(self, y_pred, y_true):
        """
        Compute negative likelihood criterion.
        :param  y_pred log probabilities of classes: model's output.
        :param  y_true True labels
        """
        return self.criterion(y_pred, y_true.long())

    def compute_metrics(self, y_pred, y_true):
        """
        Compute all metrics
        :param  y_pred log probabilities of classes: model's output.
        :param  y_true True labels
        """
        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            if metric_name == "mrr":
                try:
                    metrics[metric_name] = torch.tensor(
                        self.compute_mrr_score(
                            y_pred.cpu().detach(), y_true.cpu().detach()
                        )
                    )
                except:
                    metrics[metric_name] = torch.tensor(0.0)
            else:
                metrics[metric_name] = torch.tensor(
                    metric_fn.to(self.device)(y_pred, y_true).item(), dtype=torch.float
                )

        return metrics

    def compute_mrr_score(self, y_pred_probs, y_true):

        y_indicators = np.zeros((y_pred_probs.shape[0], self.n_classes))
        y_indicators[np.arange(y_pred_probs.shape[0]), y_true] = 1
        return sklearn.metrics.label_ranking_average_precision_score(
            y_indicators, y_pred_probs
        )

    def training_step(self, batch, batch_idx):
        text, attn_mask, y, sample_idx = (
            batch["input"],
            batch["input_attn_mask"],
            batch["target"],
            batch["sample_idx"],
        )
        ajusted_padding_size = batch["length"].max()
        text = text[:, :ajusted_padding_size]
        attn_mask = attn_mask[:, :ajusted_padding_size]

        yhat = self.forward(text, attn_mask, type="train", sample_idx=sample_idx)

        loss = self.compute_loss(yhat, y)
        if "class_weights" in batch:
            class_weights = batch["class_weights"]
            weighted_loss = loss * class_weights
            weighted_loss = weighted_loss.mean()
        loss = loss.mean()

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            on_step=True,
        )

        if "class_weights" in batch:
            self.log(
                "train_loss_weighted",
                weighted_loss,
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                on_step=True,
            )
            loss = weighted_loss

        metrics = self.compute_metrics(yhat, y)

        for metric_name, metric_value in metrics.items():
            self.log(
                f"train_{metric_name}",
                metric_value,
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                on_step=True,
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        text, attn_mask, y, sample_idx = (
            batch["input"],
            batch["input_attn_mask"],
            batch["target"],
            batch["sample_idx"],
        )
        ajusted_padding_size = batch["length"].max()
        text = text[:, :ajusted_padding_size]
        attn_mask = attn_mask[:, :ajusted_padding_size]

        yhat = self.forward(text, attn_mask, type="val", sample_idx=sample_idx)

        loss = self.compute_loss(yhat, y.type_as(yhat))
        if "class_weights" in batch:
            class_weights = batch["class_weights"]
            weighted_loss = loss * class_weights
            weighted_loss = weighted_loss.mean()
        loss = loss.mean()

        self.validation_step_outputs.append(
            {
                "y_true": y.cpu(),
                "y_pred": yhat.cpu(),
                "val_loss": loss.item(),
                "weighted_loss": (
                    weighted_loss.item() if "class_weights" in batch else None
                ),
            }
        )

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            logger=True,
            on_step=True,
            sync_dist=True,
            prog_bar=True,
        )

        if "class_weights" in batch:
            self.log(
                "val_loss_weighted",
                weighted_loss,
                on_epoch=True,
                logger=True,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )

        metrics = self.compute_metrics(yhat, y)
        for metric_name, metric_value in metrics.items():
            if metric_name in ["mcc", "f1", "f1_macro"]:continue
            self.log(
                f"val_{metric_name}",
                metric_value,
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                on_step=True,
            )

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        text, attn_mask, y, sample_idx = (
            batch["input"],
            batch["input_attn_mask"],
            batch["target"],
            batch["sample_idx"],
        )

        ajusted_padding_size = batch["length"].max()
        text = text[:, :ajusted_padding_size]
        attn_mask = attn_mask[:, :ajusted_padding_size]

        yhat = self.forward(text, attn_mask, type="test", sample_idx=sample_idx)

        loss = self.compute_loss(yhat, y.type_as(yhat))
        if "class_weights" in batch:
            class_weights = batch["class_weights"]
            weighted_loss = loss * class_weights
            weighted_loss = weighted_loss.mean()
        loss = loss.mean()

        self.test_step_outputs.append(
            {
                "y_true": y.cpu(),
                "y_pred": yhat.cpu(),
                "test_loss": loss.item(),
                "weighted_loss": (
                    weighted_loss.item() if "class_weights" in batch else None
                ),
            }
        )

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            logger=True,
            on_step=True,
            sync_dist=True,
            prog_bar=True,
        )

        if "class_weights" in batch:
            self.log(
                "test_loss_weighted",
                weighted_loss,
                on_epoch=True,
                logger=True,
                on_step=True,
                sync_dist=True,
                prog_bar=True,
            )

        metrics = self.compute_metrics(yhat, y)
        for metric_name, metric_value in metrics.items():
            if metric_name in ["mcc", "f1", "f1_macro"]:continue
            self.log(
                f"test_{metric_name}",
                metric_value,
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                on_step=True,
            )

        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-6
        )
        # Plateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            cooldown=2,
            verbose=True,
            min_lr=1e-6,
            eps=1e-08,
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
        }

        return [optimizer], [scheduler_dict]

    def on_train_start(self) -> None:
        """
        Printing the network graph to tensorboard at the begining of
        the training.
        """
        super().on_train_start()

    def on_validation_epoch_end(self):
        cm_writer = SummaryWriter(log_dir=self.logger.log_dir)
        outputs = self.validation_step_outputs
        # Compute the overall confusion matrix based on all validation batches
        all_y_pred = []
        all_y_pred_probs = []
        all_y_true = []
        all_val_loss = []
        for output in outputs:
            all_y_pred.extend(torch.argmax(output["y_pred"], dim=1).numpy().tolist())
            all_y_pred_probs.extend(output["y_pred"].numpy())
            all_y_true.extend(output["y_true"].numpy())
            all_val_loss.append(output["val_loss"])
            
            
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_pred_probs = np.array(all_y_pred_probs)
        # val_loss = torch.mean(torch.tensor(all_val_loss))
      
        cm_plot = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, self.class_names
        )
        cm_plot_normalized = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, self.class_names, normalize=True
        )

        suffix = ""
        cm_writer.add_figure(
            f"Confusion Matrix - Validation - {suffix}",
            cm_plot,
            global_step=self.current_epoch,
        )
        cm_writer.add_figure(
            f"Confusion Matrix Normalized - Validation-{suffix}",
            cm_plot_normalized,
            global_step=self.current_epoch,
        )

        val_mcc = sklearn.metrics.matthews_corrcoef(all_y_true, all_y_pred)
        val_f1_macro  = sklearn.metrics.f1_score(all_y_true, all_y_pred,average="macro")
        val_f1_weighted= sklearn.metrics.f1_score(all_y_true,all_y_pred,average="weighted")
        
        
        
        self.log("val_mcc_epoch",val_mcc,on_step=False,on_epoch=True)
        self.log("val_f1_epoch",val_f1_weighted,on_step=False, on_epoch=True)
        self.log("val_f1_macro_epoch",val_f1_macro,on_step=False, on_epoch=True)
                                 
        if val_mcc > self.best_val_mcc:
            self.best_val_mcc = val_mcc
            suffix = "best"
            cm_writer.add_figure(
                f"Confusion Matrix - Validation - {suffix}",
                cm_plot,
                global_step=self.current_epoch,
            )
            cm_writer.add_figure(
                f"Confusion Matrix Normalized - Validation-{suffix}",
                cm_plot_normalized,
                global_step=self.current_epoch,
            )

            # Log the yhat and ytrue for the best model
            output_file = f"{self.logger.log_dir}/best_model_outputs.pt"
            torch.save(
                {"y_true": all_y_true,
                 "y_pred": all_y_pred,
                " y_pred_probs": all_y_pred_probs 
                },
                output_file,
            )

        self.validation_step_outputs.clear()
        super().on_validation_epoch_end()



    def on_test_epoch_end(self):
        cm_writer = SummaryWriter(log_dir=self.logger.log_dir)
        outputs = self.test_step_outputs
        # Compute the overall confusion matrix based on all validation batches
        all_y_pred = []
        all_y_pred_probs = []
        all_y_true = []
        all_test_loss = []
        for output in outputs:
            all_y_pred.extend(torch.argmax(output["y_pred"], dim=1).numpy().tolist())
            all_y_pred_probs.extend(output["y_pred"].numpy())
            all_y_true.extend(output["y_true"].numpy())
            all_test_loss.append(output["test_loss"])
            
            
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_pred_probs = np.array(all_y_pred_probs)
        
        cm_plot = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, self.class_names
        )
        cm_plot_normalized = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, self.class_names, normalize=True
        )

        suffix = ""
        cm_writer.add_figure(
            f"Confusion Matrix - Test - {suffix}",
            cm_plot,
            global_step=self.current_epoch,
        )
        cm_writer.add_figure(
            f"Confusion Matrix Normalized - Test-{suffix}",
            cm_plot_normalized,
            global_step=self.current_epoch,
        )

        test_mcc = sklearn.metrics.matthews_corrcoef(all_y_true, all_y_pred)
        test_f1_macro  = sklearn.metrics.f1_score(all_y_true, all_y_pred,average="macro")
        test_f1_weighted= sklearn.metrics.f1_score(all_y_true,all_y_pred,average="weighted")
        
        
        
        self.log("test_mcc_epoch",test_mcc,on_step=False,on_epoch=True)
        self.log("test_f1_epoch",test_f1_weighted,on_step=False, on_epoch=True)
        self.log("test_f1_macro_epoch",test_f1_macro,on_step=False, on_epoch=True)
                                 
        
        # Log the yhat and ytrue for the best model
        self.test_step_outputs.clear()
        super().on_test_epoch_end()



    @staticmethod
    def load_from_experiment(experiment_dir: str) -> "TextTransformerForClassification":
        """
        Load a model from a given experiment directory.
        :param experiment_dir: Directory of the experiment.
        :param checkpoint_name: Name of the checkpoint to load.
        :return: The loaded model.
        """
        # Load the model

        experiment_config = ExperimentUtils.load_experiment_data(experiment_dir)

        n_accounts = experiment_config["model_config"]["n_accounts"]
        pretrained_model = experiment_config["model_config"]["pretrained_model"]
        mlp_hidden_dim = experiment_config["model_config"]["mlp_hidden_dim"]
        n_classes = experiment_config["model_config"]["n_classes"]

        model = TextTransformerForClassification(
            n_accounts=n_accounts,
            pretrained_model=pretrained_model,
            mlp_hidden_dim=mlp_hidden_dim,
            n_classes=n_classes,
            class_names=None,
            dropout_rate=0,
            trf_trainable_layers=0,
            learning_rate=1e-5,
        )

        # Load the weights
        ckpt_path = experiment_config["best_model"]["path"]
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])

        return model

    def get_hidden_dim(self):
        return self.text_transformer.config.hidden_size

    
    
    @staticmethod
    def calculate_metrics(experiment_dir: str):
        """
        Compute and update the metrics for a given experiment directory.
        :param experiment_dir: Directory of the experiment.
        """

        # Mock model (Don't build it)
        model = TextTransformerForClassification(n_accounts=0, build_on_init=False)

        # Load best model outputs
        best_model_outputs = torch.load(f"{experiment_dir}/tb/best_model_outputs.pt")
        y_true = torch.from_numpy(best_model_outputs["y_true"])
        y_pred = torch.from_numpy(best_model_outputs["y_pred"])

        # Compute all the metrics based on the best torch.smodel outputs
        metrics = model.compute_metrics(y_pred, y_true)
        print(f"Metrics: {metrics}")
        