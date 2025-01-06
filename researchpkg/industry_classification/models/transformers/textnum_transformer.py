import os
from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
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
from tqdm import tqdm
from transformers import BertModel

from researchpkg.industry_classification.models.transformers.bert_explainability_modules.BERT_explainability.modules.layers_lrp_new import (
    Linear,
    Sequential,
    Tanh,
)
from researchpkg.industry_classification.models.transformers.pretrained_index import (
    PretrainedTrf,
)
from researchpkg.industry_classification.models.utils import NN_Utils


class TextNumTransformerForClassification(pl.LightningModule):
    def __init__(
        self,
        n_accounts: int,
        pretrained_model=PretrainedTrf.BERT_TINY,
        n_head=8,
        n_layers=4,
        emb_dim=512,
        ffw_dim=1024,
        n_classes=10,
        dropout_rate=0.1,
        trf_trainable_layers=1,
        learning_rate=5e-3,
        class_names=None,
        build_on_init=True,
        text_project_dim=4,
        mlp_head_dim=128,
        dim_net_change=1,
        use_lrp_modules=False,
    ):
        """
        Initialize the model.
        :param n_accounts: Number of accounts in the dataset.
        :param pretrained_model: Pretrained model to use. See PretrainedTrf enum.
        :param n_head: Number of heads in the transformer.
        :param n_layers: Number of layers in the transformer.
        :param emb_dim: Embedding dimension.
        :param ffw_dim: Feed forward dimension.
        :param n_classes: Number of classes.
        :param dropout_rate: Dropout rate.
        :param trf_trainable_layers: Number of trainable layers in the transformer.
        :param learning_rate: Learning rate.
        :param class_names: List of class names.
        :paraù build_on_init: Build the model on init.
        :param text_project_dim: Dimension of the text projection.
        :param mlp_head_dim: Dimension of the head of the MLP.
        :param dim_net_change: Dimension of the net change.Default to 1.
        :param use_lrp_modules: Use the LRP modules for explainability.
        """

        super().__init__()
        self.n_accounts = n_accounts
        self.n_head = n_head
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.ffw_dim = ffw_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.dim_net_change = dim_net_change

        self.text_project_dim = text_project_dim

        self.n_classes = n_classes
        self.class_names = class_names
        if self.class_names is None:
            self.class_names = [str(i + 1) for i in range(self.n_classes)]
        else:
            assert len(self.class_names) == self.n_classes

        self.pretrained_model = pretrained_model
        self.trf_trainable_layers = trf_trainable_layers
        self.mlp_head_dim = mlp_head_dim
        # Loss
        # self.criterion = nn.NLLLoss(reduction="none")
        self.criterion = nn.CrossEntropyLoss(reduction="none")

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
        self.use_lrp_modules = use_lrp_modules
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

        self.set_text_transformer_trainable(False)

        # Make embedding trainable
        for param in self.text_transformer.embeddings.parameters():
            param.requires_grad = False

        self.text_project_dim = self.text_transformer.config.hidden_size
        self.text_transformer_projector = nn.Identity()

        # 2. Text Num encoder(Embedding)
        self.text_num_encoder = nn.Sequential(
            nn.Linear(self.text_project_dim, self.emb_dim),
            nn.GELU(),
        )

        # 3. TextNumTransformer
        
        if not self.use_lrp_modules:
            self.text_num_transformer = nn.TransformerEncoder(
                num_layers=self.n_layers,
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=self.n_head,
                    dim_feedforward=self.ffw_dim,
                    dropout=self.dropout_rate,
                    activation="gelu",
                    batch_first=True,
                ),
            )
        else:
            #Use moules lrp-explainability
            from transformers import BertConfig
            from researchpkg.industry_classification.models.transformers.bert_explainability_modules.BERT_explainability.modules.BERT.BERT import (  # noqa
                BertEncoder,
            )

            bert_config = BertConfig(
                hidden_size=self.emb_dim,
                num_hidden_layers=self.n_layers,
                num_attention_heads=self.n_head,
                intermediate_size = self.ffw_dim,
                hidden_act="gelu",
                hidden_dropout_prob=self.dropout_rate,
                attention_probs_dropout_prob=self.dropout_rate,
            )
            self.text_num_transformer = BertEncoder(bert_config)    
            
        
        # The cls is  equivalent to the cls token of Bert for example
        self.text_num_cls = torch.nn.Parameter(torch.randn(self.emb_dim))
        

        # 4. Fully Connected Layers
        self.classifier = Sequential(
            Linear(self.emb_dim, self.n_classes)
        )

        self.hparams_text = (
            f"{self.pretrained_model.lower()}l{self.trf_trainable_layers}_row{self.n_accounts}"
            f"_h{self.n_head}_l{self.n_layers}_e{self.emb_dim}_f{self.ffw_dim}_d{self.dropout_rate}"
            f"_proj{self.text_project_dim}"
        )

        self.__name__ = f"transformer_encoder_{self.hparams_text}"
        
        if self.use_lrp_modules:
            self.__name__ ="lrp_"+self.__name__

    def set_text_transformer_trainable(self, trainable: bool = False):
        if type(self.text_transformer) == BertModel:
            layers = self.text_transformer.encoder.layer
        else:
            # Distilbert
            layers = self.text_transformer.transformer.layer
        n_layers = len(layers)
        if self.trf_trainable_layers < n_layers:
            for layer in layers[: n_layers - self.trf_trainable_layers]:
                for param in layer.parameters():
                    param.requires_grad = trainable

    
    def __textnumtransformer_call__(self, input_enc, input_attn_mask):
        if not self.use_lrp_modules:
            text_num_enc = self.text_num_transformer(
                input_enc, src_key_padding_mask=input_attn_mask
            )
        else:
            text_num_enc = self.text_num_transformer(
                input_enc, encoder_attention_mask=input_attn_mask)[0]
            
            
        return text_num_enc
 
    def forward(
        self,
        input_desc: torch.Tensor,
        input_net_change: torch.Tensor,
        input_attn_mask: torch.Tensor,
        sample_idx=None,
        type="train",
        return_attn_weights=False,
        return_emb=False,
        return_text_num_enc=False,
    ) -> torch.Tensor:
        # TODO : GW Replace of those options by a "return_dict" option.
        """
        :param input_desc: Tokenized accounts descriptions. Shape [bs, n_accounts, seq_len]
        :param input_net_change: Accounts net changes Shape [bs, n_accounts, 1]
        :param input_attn_mask: Mask for account spositions. Shape [bs, n_accounts, seq_len]
        :param sample_idx: List of sample idx for using precomputed text embeddings.
        :param mode: train, val or text
        :return:
        """

        
        # Encode the description using the text_transformer
        bs, n_seq, seq_len = input_desc.shape
        batch_input_desc = input_desc.view(bs * n_seq, -1)
        batch_input_attn_mask = input_attn_mask.view(bs * n_seq, -1)

        text_outputs = torch.zeros(
            bs * n_seq,
            self.text_transformer.config.hidden_size,
            device=input_desc.device,
        )

        masked_seq = (
            batch_input_attn_mask.sum(1) == 0
        )  # Full masked sequenced (to no proceed)emb
        
        text_transformer_full_output = self.text_transformer(
            batch_input_desc[~masked_seq],
            attention_mask=batch_input_attn_mask[~masked_seq],
        )[0]
    
        text_outputs[~masked_seq] = self.text_transformer.pooler(text_transformer_full_output)
        #Apply pooler
        text_outputs[masked_seq] = 0
        text_outputs = text_outputs.view(bs, n_seq, -1)
        text_outputs = self.text_transformer_projector(text_outputs)


        # Combine with numerical input (input_net_change) if needed
        if input_net_change is not None:
            text_num_enc = text_outputs * input_net_change
        else:
            text_num_enc = text_outputs

        # Add the cls token.The attention_weight from the last layer of the first transformer
        text_num_enc = self.text_num_encoder(text_num_enc)  # Encode Text and Net cha

        if return_text_num_enc:
            return text_num_enc

        # Apply the transformer encoder
        text_num_padding_mask = input_attn_mask.sum(-1) == 0  # src mask

        text_num_enc = torch.cat(
            (
                self.text_num_cls.unsqueeze(0).repeat(text_num_enc.shape[0], 1, 1),
                text_num_enc,
            ),
            dim=1,
        )

        text_num_padding_mask = torch.cat(
            (
                torch.zeros(
                    text_num_enc.shape[0], 1, device=text_num_enc.device, dtype=torch.bool
                ),
                text_num_padding_mask,
            ),
            dim=1,
        )

        attn_weights = None

        if not return_attn_weights:
            text_num_enc = self.__textnumtransformer_call__(text_num_enc, text_num_padding_mask)
        else:
            x = text_num_enc
            attn_weights = []
            padding_mask = text_num_padding_mask
            for i in range(self.text_num_transformer.num_layers):
                trf = self.text_num_transformer.layers[i]
                if trf.norm_first:
                    x = trf.norm1(x)
                    att_values, att_weights = trf.self_attn(
                        x, x, x, key_padding_mask=padding_mask, need_weights=True
                    )
                    x = x + att_values
                    x = x + trf.dropout(
                        trf.linear2(
                            trf.dropout(trf.activation(trf.linear1(trf.norm2(x))))
                        )
                    )
                else:
                    att_values, att_weights = trf.self_attn(
                        x, x, x, key_padding_mask=padding_mask, need_weights=True
                    )
                    x = trf.norm1(x + att_values)
                    x = trf.norm2(
                        x
                        + trf.dropout(
                            trf.linear2(trf.dropout(trf.activation(trf.linear1(x))))
                        )
                    )
                attn_weights.append(att_weights)

            text_num_enc = x

        # Ensure combined outputs has the same shatext_num_enc.shape[1]pe as text_num_enc
        text_num_enc = torch.nn.functional.pad(
            text_num_enc,
            (0, 0, 0, text_num_enc.shape[1] - text_num_enc.shape[1]),
            "constant",
            0,
        )

        # Compute the cls from the text num transformer output
        text_num_enc_cls = text_num_enc[:, 0, :]

        # Fully connected layer
        probs = self.classifier(text_num_enc_cls)

        if return_emb:
            return text_num_enc, probs

        if not return_attn_weights:
            return probs
        else:
            return probs, attn_weights

    def compute_loss(self, y_pred, y_true):
        """
        Compute negative likelihood criterion.
        :param  y_pred log probabilities of classes: model's output.
        :param  y_true True labels
        """
        return self.criterion(y_pred, y_true.long())

    def compute_mrr_score(self, y_pred_probs, y_true):

        y_indicators = np.zeros((y_pred_probs.shape[0], self.n_classes))
        y_indicators[np.arange(y_pred_probs.shape[0]), y_true] = 1
        return sklearn.metrics.label_ranking_average_precision_score(
            y_indicators, y_pred_probs
        )

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

    def training_step(self, batch, batch_idx):
        text, net_change, attn_mask, y, sample_idx = (
            batch["input_desc"],
            batch["input_net_change"],
            batch["input_attn_mask"],
            batch["target"],
            batch["sample_idx"],
        )
        yhat = self.forward(
            text, net_change, attn_mask, type="train", sample_idx=sample_idx
        )

        loss = self.compute_loss(yhat, y)

        if "class_weights" in batch:
            self.log(
                "train_loss",
                loss.mean(),
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                on_step=True,
            )
            loss = loss * batch["class_weights"]
            self.log(
                "weigthed_train_loss",
                loss.mean(),
                prog_bar=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
                on_step=True,
            )

        else:
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
        text, net_change, attn_mask, y, sample_idx = (
            batch["input_desc"],
            batch["input_net_change"],
            batch["input_attn_mask"],
            batch["target"],
            batch["sample_idx"],
        )

        yhat = self.forward(
            text, net_change, attn_mask, type="val", sample_idx=sample_idx
        )

        loss = self.compute_loss(yhat, y.type_as(yhat)).mean()

        self.validation_step_outputs.append(
            {"y_true": y.cpu(), "y_pred": yhat.cpu(), "val_loss": loss.item()}
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

        metrics = self.compute_metrics(yhat, y)
        for metric_name, metric_value in metrics.items():
            if metric_name in ["mcc", "f1", "f1_macro"]:
                continue
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
        text, net_change, attn_mask, y, sample_idx = (
            batch["input_desc"],
            batch["input_net_change"],
            batch["input_attn_mask"],
            batch["target"],
            batch["sample_idx"],
        )

        yhat = self.forward(
            text, net_change, attn_mask, type="val", sample_idx=sample_idx
        )

        loss = self.compute_loss(yhat, y.type_as(yhat)).mean()

        self.test_step_outputs.append(
            {"y_true": y.cpu(), "y_pred": yhat.cpu(), "test_loss": loss.item()}
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

        metrics = self.compute_metrics(yhat, y)
        for metric_name, metric_value in metrics.items():
            if metric_name in ["mcc", "f1", "f1_macro"]:
                continue
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
            min_lr=5e-6,
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
        # TODO : Enable graph plotting
        # sample_desc =  torch.randint(1,128,(1,self.n_accounts,MAX_DESC_LENGTH))
        # sample_net_change = torch.randn(1,self.n_accounts,1)
        # self.logger.experiment.add_graph(self,(sample_desc, sample_net_change))
        super().on_train_start()
        # if self.datamodule_for_precomputing  is not None:
        #     self.compute_all_text_embeddings(self.datamodule_for_precomputing.train_dataloader(shuffle=False),
        #                                      self.datamodule_for_precomputing.val_dataloader()
        #                                      )

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
        val_f1_macro = sklearn.metrics.f1_score(all_y_true, all_y_pred, average="macro")
        val_f1_weighted = sklearn.metrics.f1_score(
            all_y_true, all_y_pred, average="weighted"
        )

        self.log("val_mcc_epoch", val_mcc, on_step=False, on_epoch=True)
        self.log("val_f1_epoch", val_f1_weighted, on_step=False, on_epoch=True)
        self.log("val_f1_macro_epoch", val_f1_macro, on_step=False, on_epoch=True)

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
                {
                    "y_true": all_y_true,
                    "y_pred": all_y_pred,
                    " y_pred_probs": all_y_pred_probs,
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
        test_f1_macro = sklearn.metrics.f1_score(
            all_y_true, all_y_pred, average="macro"
        )
        test_f1_weighted = sklearn.metrics.f1_score(
            all_y_true, all_y_pred, average="weighted"
        )
        test_mrr = self.compute_mrr_score(all_y_pred_probs, all_y_true)

        self.log("test_mcc_epoch", test_mcc, on_step=False, on_epoch=True)
        self.log("test_f1_epoch", test_f1_weighted, on_step=False, on_epoch=True)
        self.log("test_f1_macro_epoch", test_f1_macro, on_step=False, on_epoch=True)
        self.log("test_mrr_epoch", test_mrr, on_step=False, on_epoch=True)

        # Write results_test.yaml
        output_file = f"{self.logger.log_dir}/results_test.yaml"
        with open(output_file, "w") as f:
            f.write(f"test_mcc: {test_mcc}\n")
            f.write(f"test_f1_macro: {test_f1_macro}\n")
            f.write(f"test_f1_weighted: {test_f1_weighted}\n")
            f.write(f"test_mrr: {test_mrr}\n")
            f.write(f"test_loss: {np.mean(all_test_loss)}\n")
            f.write(f"best_val_mcc: {self.best_val_mcc}\n")
            f.write(f"best_val_mcc_epoch: {self.current_epoch}\n")
            f.write(f"best_val_mcc: {self.best_val_mcc}\n")
            f.write(f"best_val_mcc_epoch: {self.current_epoch}\n")

        # Log the yhat and ytrue for the best model
        self.test_step_outputs.clear()
        super().on_test_epoch_end()

    def compute_all_text_num_cls(
        self, dataloader, output_dir
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the embeddings of samples in dataloader
        Embedding actually corresponds to the cls ouput of the transformer
        """
        all_embeddings = []
        all_y_true = []
        all_y_preds = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with torch.no_grad():
            for batch in tqdm(
                dataloader, "Computing financial statements' embeddings :"
            ):
                input_desc, net_change, attn_mask, y, samples_idx, tags = (
                    batch["input_desc"],
                    batch["input_net_change"],
                    batch["input_attn_mask"],
                    batch["target"],
                    batch["sample_idx"],
                    batch["tags"],
                )

                embeddings, y_probs = self.forward(
                    input_desc.to(self.device),
                    net_change.to(self.device),
                    attn_mask.to(self.device),
                    type="val",
                    sample_idx=samples_idx,
                    return_emb=True,
                )

                embeddings_cls = embeddings[:, 0, :]

                all_embeddings.extend(embeddings_cls.cpu().detach().numpy())
                all_y_true.extend(y.cpu().detach().numpy())
                all_y_preds.extend(torch.argmax(y_probs, dim=1).cpu().detach().numpy())

                # Save separetelely the embeddings for each sample per tag
                for idx in range(input_desc.shape[0]):
                    sample_idx = samples_idx[idx]

                    # Create a separate directory for the filee
                    sample_output_dir = os.path.join(output_dir, f"{sample_idx}")
                    os.makedirs(sample_output_dir, exist_ok=True)

                    tag_list = tags[idx].split(";")

                    # Save the tag_list to csv
                    tag_list_file = os.path.join(sample_output_dir, "tags.csv")
                    with open(tag_list_file, "w") as f:
                        f.write("\n".join(tag_list))

                    # Stack the embeddings and save npy file
                    text_num_embeddings = embeddings[idx].cpu().detach().numpy()
                    text_num_embeddings = text_num_embeddings[0 : len(tag_list) + 1]
                    emb_output_file_name = os.path.join(
                        sample_output_dir, "text_num_embeddings.npy"
                    )
                    np.save(emb_output_file_name, text_num_embeddings)

        all_embeddings = np.stack(all_embeddings)
        all_y_true = np.array(all_y_true)
        all_y_preds = np.array(all_y_preds)

        emb_output_file_name = os.path.join(output_dir, "embeddings.npy")
        np.save(emb_output_file_name, all_embeddings)
        print(f"Embeddings saved to {emb_output_file_name}")

        # Save y_true and y_pred
        pred_out_put_df = pd.DataFrame({"y_true": all_y_true, "y_pred": all_y_preds})
        pred_output_file_name = os.path.join(output_dir, "predictions.csv")

        pred_out_put_df.to_csv(pred_output_file_name)

        return embeddings, all_y_true, all_y_preds

    def forward_only_textnum(self, text_num_enc, input_attn_mask):
        # Apply the transformer encoder
        text_num_padding_mask = input_attn_mask.sum(-1) == 0  # src mask

        text_num_enc = torch.cat(
            (
                self.text_num_cls.unsqueeze(0).repeat(text_num_enc.shape[0], 1, 1),
                text_num_enc,
            ),
            dim=1,
        )

        text_num_padding_mask = torch.cat(
            (
                torch.zeros(
                    text_num_enc.shape[0], 1, device=text_num_enc.device, dtype=torch.bool
                ),
                text_num_padding_mask,
            ),
            dim=1,
        )
        text_num_enc = self.__textnumtransformer_call__(text_num_enc, text_num_padding_mask)
        # Compute the cls from the text num transformer output
        text_num_enc_cls = text_num_enc[:, 0, :]
        

        # Fully connected layer
        probs = self.classifier(text_num_enc_cls)

        return probs

    def relprop(self, cam, **kwargs):
        cam = self.classifier.relprop(cam, **kwargs)
        cam = self.text_num_transformer.relprop(cam, **kwargs)
        return cam
