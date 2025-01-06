""""
Run Training Text num transfomer
"""

import argparse
import os
import shutil
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from researchpkg.industry_classification.utils.sics_loader import load_sic_codes

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(dir_path)
from researchpkg.industry_classification.callbacks.custom_model_checkpoint_callback import (
    CustomModelCheckpointCallback,
)
from researchpkg.industry_classification.config import LOGS_DIR, SEC_ROOT_DATA_DIR
from researchpkg.industry_classification.dataset.sec_datamodule import SecDataset
from researchpkg.industry_classification.dataset.sec_textnum_transformer_datamodule import (
    SecTrfClassificationDataModule,
)
from researchpkg.industry_classification.models.transformers.pretrained_index import (
    PretrainedTrf,
)
from researchpkg.industry_classification.models.transformers.textnum_transformer import (
    TextNumTransformerForClassification,
)
from researchpkg.industry_classification.utils.experiment_utils import (
    ExperimentUtils,
)

parser = argparse.ArgumentParser(
    prog="Run Training of TextNum Transformer Model",
    description="Training TextNumTransformer model.",
)
parser.add_argument(
    "-g",
    "--global_exp_name",
    default="count30_sic1agg_including_is",
    help="The global experiment name to use. Should match one of existing generated datasets.\
        Example : _count30_sic1agg, _count30_sic1agg_with12ratios",
)

parser.add_argument(
    "-lr", "--learning_rate", type=float, default=1e-5, help="The learning rate"
)
parser.add_argument(
    "-bs", "--batch_size", type=int, default=16, help="The batch size of the model"
)
parser.add_argument(
    "-a",
    "--accelerator",
    default="cpu" if not torch.cuda.is_available() else "cuda",
    help="The accelerator to use",
)
parser.add_argument("-x", "--experiment_name", default="")

parser.add_argument(
    "-nh",
    "--n_head",
    default=4,
    type=int,
    help="The number of head of the textnum transfomrer",
)
parser.add_argument(
    "-nl",
    "--n_layers",
    default=6,
    type=int,
    help="The number of lyers of the textnum trasnfomrer",
)

parser.add_argument(
    "-ffw",
    "--feed_forward-dim",
    default=1024,
    type=int,
    help="Feed forward dim of the textnum transformer",
)

parser.add_argument(
    "-he",
    "--emb_dim",
    default=256,
    type=int,
    help="The embedding dim of the transformer ",
)
parser.add_argument(
    "-nt",
    "--n_trainable_layers",
    default=2,
    type=int,
    help="The number of trainable layers of the textnecoder transformer",
)

parser.add_argument(
    "-ptr",
    "--pretrained_text_transformer",
    default=PretrainedTrf.BERT_TINY,
    help="The pretrained transformer model to use"
)

parser.add_argument(
    "-r", "--reset", action="store_true", help="Reset the experiment from the begining"
)
parser.add_argument(
    "-w",
    "--num_workers",
    type=int,
    default=7,
    help="Dataloaders num workers",
)
parser.add_argument(
    "-dp", "--dropout_rate", type=float, default=0.5, help="The dropout rate"
)
parser.add_argument(
    "-m", "--load_in_memory", default=0, type=int, help="Load thedropout dataset in memory"
)
parser.add_argument(
    "--sic_digits",
    default=1,
    type=int,
    help="The target features sic{sic_digits}. Ex: sic1 , sci2",
)
parser.add_argument(
    "-e", "--epochs", type=int, default=40, help="The number of training epochs"
)
parser.add_argument(
    "--no_change",
    action="store_true",
    help="If true make prediction only using textual descriptions.",
)
parser.add_argument("--no_gaap", action="store_true", help="Use dataset without gaap")

parser.add_argument(
    "--precompute_text_embeddings",
    action="store_true",
    help="If true make prediction only using textual descriptions.",
)
parser.add_argument(
    "--max_desc_len",
    type=int,
    default=32,
    help="The maximum number of tokens per account description",
)
parser.add_argument(
    "--balance_sampling", action="store_true", help="Use balance sampling"
)
parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss")

parser.add_argument("--only_test", action="store_true",
                    help="Test mode only run dataloader.test")
parser.add_argument(
    "--max_tag_depth",
    type=int,
    default=None,
    help="The maximum depth of tags in the taxonomy to considerer.By default all tags are considered.",
    # noqa
)
parser.add_argument(
    "--compute_embeddings",
    action="store_true")
parser.add_argument(
    "--use_lrp_modules",
    action="store_true",
    help="Use LRP modules in the model for explainability")
def train(
    global_exp_name: str,
    learning_rate: float,
    batch_size: int,
    accelerator: str,
    experiment_name: str,
    n_head: int,
    n_layers: int,
    emb_dim: int,
    n_trainable_layers: int,
    pretrained_text_transformer: str,
    reset: bool,
    num_workers: int,
    load_in_memory: bool,
    sic_digits: int,
    feed_forward_dim: int,
    epochs: int,
    no_change: bool,
    dropout_rate: float,
    precompute_text_embeddings: bool,
    no_gaap: bool,
    max_desc_len: int,
    balance_sampling: bool,
    weighted_loss: bool,
    only_test:bool,
    max_tag_depth: int = None,
    compute_embeddings: bool = False,
    use_lrp_modules: bool = False,
):
    """
    Run the training of the balance sheet classification model.
    """

    ExperimentUtils.check_global_experiment_name(global_exp_name)
    dataset_dir = os.path.join(SEC_ROOT_DATA_DIR, global_exp_name)
    experiments_dir = os.path.join(LOGS_DIR, f"experiments_{global_exp_name}")

    accounts_index, registrants_index, _ = SecDataset.load_index(
        dataset_dir, sic_digits=sic_digits
    )

    labels = list(sorted(registrants_index[f"sic{sic_digits}"].unique().tolist()))
    n_labels = len(labels)

    sic_code_df = load_sic_codes()[["sic", "industry_title"]]

    labels = [
        sic_code_df[sic_code_df["sic"] == l]["industry_title"].values[0] for l in labels
    ]
    n_labels = len(labels)

    n_accounts = len(accounts_index)

    model = TextNumTransformerForClassification(
        n_accounts=n_accounts,
        pretrained_model=pretrained_text_transformer,
        n_head=n_head,
        n_layers=n_layers,
        n_classes=n_labels,
        emb_dim=emb_dim,
        ffw_dim=feed_forward_dim,
        learning_rate=learning_rate,
        class_names=labels,
        dropout_rate=dropout_rate,
        trf_trainable_layers=n_trainable_layers,
        use_lrp_modules=use_lrp_modules,
    )
    experiment_name = (
        f"{model.__name__}_{'_nochange' if no_change else ''}"
        f"_{experiment_name}_sic{sic_digits}"
    )
    if weighted_loss:
        experiment_name += "_weighted_loss"
    if balance_sampling:
        experiment_name += "_balance_sampling"
    if max_tag_depth is not None:
        experiment_name += f"_max_tag_depth_{max_tag_depth}"
    
    experiment_dir = os.path.join(experiments_dir, experiment_name)
    # 3. Trainer
    checkpoint_cb = CustomModelCheckpointCallback(
        experiment_dir,
        monitor="val_mcc_epoch",
        mode="max",
        save_top_k=2,
        auto_insert_metric_name=True,
        save_last=True,
    )

    if reset and os.path.exists(experiment_dir):
        print("Reset experiment")
        shutil.rmtree(experiment_dir)

    if not ExperimentUtils.check_experiment(experiment_dir):
        # 4. Initialize the experiment
        from researchpkg.industry_classification.models.utils import NN_Utils

        model.hparams["num_params"] = NN_Utils.get_num_params(model)
        model.hparams["num_trainable_params"] = NN_Utils.get_num_trainable_params(model)
        model.hparams["size"] = f"{NN_Utils.get_model_size(model)} MB"
        ExperimentUtils.initialize_experiment(
            experiment_dir,
            dataset_dir,
            model.hparams,
            training_config={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "learning_rate": learning_rate,
                "device": accelerator,
                "max_desc_len": max_desc_len,
                "ngpus": torch.cuda.device_count() if accelerator == "cuda" else 0,
                "epochs": epochs,
            },
        )

    dataset_config = ExperimentUtils.load_experiment_data(experiment_dir)[
        "dataset_config"
    ]

    datamodule = SecTrfClassificationDataModule(
        dataset_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        sic_digits=sic_digits,
        load_in_memory=load_in_memory,
        tokenizer=model.tokenizer,
        use_change=not no_change,
        max_desc_len=max_desc_len,
        max_tags=dataset_config["max_nb_tag_per_sheet"],
        balance_sampling=balance_sampling,
        weighted_loss=weighted_loss,
        max_tag_depth=max_tag_depth,
    )
    

    if n_trainable_layers == 0 and precompute_text_embeddings:
        model.set_datamodule_for_precomputing(datamodule=datamodule)

    logger = pl.loggers.TensorBoardLogger(experiment_dir, name="", version="tb")

    
    if compute_embeddings:
        #Load the best model
        ckpt_file = ExperimentUtils.get_best_model( os.path.basename(experiment_dir),os.path.dirname(experiment_dir),)["path"]
        assert os.path.exists(ckpt_file), f"{ckpt_file} do not exists"
        model  = model.load_from_checkpoint(ckpt_file)
        output_dir = os.path.join(experiment_dir,"text_embeddings")
        model.compute_all_textnum_cls(datamodule.test_dataloader(),output_dir)   
        return 
    
    
    
    
    lr_monitor_cb = LearningRateMonitor(logging_interval="step")
    # Check where last.ckpt exists
    last_ckpt_path = os.path.join(experiment_dir, "last.ckpt")
    ckpt_exists = os.path.exists(last_ckpt_path)
    trainer = pl.Trainer(
        devices=-1,
        callbacks=[
            # early_stop_cb,
            lr_monitor_cb,
            checkpoint_cb,
        ],
        logger=logger,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        max_epochs=epochs,
        num_sanity_val_steps=0 if ckpt_exists else 2,
        strategy="ddp_find_unused_parameters_true",
    )

    if not only_test:
        trainer.fit(model, datamodule, ckpt_path="last")
    
  
    ckpt_file = ExperimentUtils.get_best_model( os.path.basename(experiment_dir),os.path.dirname(experiment_dir),)["path"]
    assert os.path.exists(ckpt_file), f"{ckpt_file} do not exists"
    trainer.test(model,datamodule,ckpt_path=ckpt_file)


if __name__ == "__main__":
    args: argparse.Namespace = parser.parse_args()
    train(**vars(args))
