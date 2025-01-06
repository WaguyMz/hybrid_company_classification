import random
from typing import Dict, Tuple, Union
import os
import huggingface_hub
import pandas as pd
import pytorch_lightning as pl
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from trl import SFTTrainer

from researchpkg.industry_classification.models.transformers.pretrained_index import (
    PretrainedTrf,
)
from researchpkg.industry_classification.models.transformers.text_transformer import (
    TextTransformerForClassification,
)
from researchpkg.industry_classification.models.utils import NN_Utils
import numpy as np 
import sklearn.metrics

from researchpkg.industry_classification.utils.experiment_utils import ExperimentUtils
class TextLLMForInstructionSFT(TextTransformerForClassification):
    """
    A specilized version of TextTransformerForClassification for LLM models.
    Using PEft
    """

    def __init__(
        self,
        n_accounts: int,
        pretrained_model=PretrainedTrf.BERT_TINY,
        n_classes=10,
        class_names=None,
        learning_rate=1e-3,
        build_on_init=True,
        dropout_rate=0.1,
        use_ia3=False):
        super().__init__(
            n_accounts=n_accounts,
            pretrained_model=pretrained_model,
            mlp_hidden_dim=0,
            n_classes=n_classes,
            class_names=class_names,
            trf_trainable_layers=0,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            build_on_init=False,
        )
        
        self.is_encoder = False
        self.use_ia3 = use_ia3

        if build_on_init:
            self.build_model()

    def build_llama_gptq(self, model_config: dict) -> Tuple:
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        from transformers import AutoModel, GPTQConfig

        from accelerate import PartialState
        quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
        llm = AutoModelForCausalLM.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            config=quantization_config_loading,
            device_map={'':PartialState().process_index},
            torch_dtype=torch.float16,
        )


        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        llm.config.pad_token_id = tokenizer.pad_token_id
        return llm, tokenizer


    def build_gemma_gptq(self, model_config: dict) -> Tuple:
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        from transformers import AutoModel, GPTQConfig

        quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
        from accelerate import Accelerator
        llm = AutoModelForCausalLM.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            config=quantization_config_loading,
            device_map={'':Accelerator().process_index},
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        )

        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        llm.config.pad_token_id = tokenizer.pad_token_id
        return llm, tokenizer

    def build_llm_qlora(self, model_config: dict) -> Tuple:
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
          load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

        from accelerate import Accelerator
        llm = AutoModelForCausalLM.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            quantization_config=quantization_config,
            device_map={'':Accelerator().process_index},
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        )

        tokenizer = AutoTokenizer.from_pretrained(repo_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        llm.config.pad_token_id = tokenizer.pad_token_id
        return llm, tokenizer
    

    def build_finance_llm_gptq(self, model_config: dict) -> Tuple:
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]
        from accelerate import Accelerator

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        # from accelerate import infer_afuto_device_map
        # QLoRa fine-tuning:

        model = AutoModelForCausalLM.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            device_map={'':Accelerator().process_index},
            torch_dtype=torch.float16,
        )
       
        
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    
    def build_gemma_unsloth(self,model_config:dict) -> Tuple:
        """
        Unsloth basd gemma model.
        Requirements: Flash attention 2 should be installed.
        Flash Attention 2 requires cuda 11.6 (currently cuda 11.2)
        
        """
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_config["link"],
            max_seq_length = 2048,
            dtype = None,
            load_in_4bit = True
            # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
        model = FastLanguageModel.get_peft_model(model,r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
        )
        return model, tokenizer
        
    
    
    def build_model(self):
        """
        Build the model.
        """

        if (
            self.pretrained_model == PretrainedTrf.GEMMA_2B_GPTQ
            or self.pretrained_model == PretrainedTrf.GEMMA_2B_INSTRUCT_GPTQ
            or self.pretrained_model == PretrainedTrf.GEMMA_9B_INSTRUCT_GPTQ
            or self.pretrained_model == PretrainedTrf.GEMMA_9B_GPTQ
        ):
            self.model, self.tokenizer = self.build_gemma_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        
        elif self.pretrained_model == PretrainedTrf.GEMMA_9B_QLORA:
            self.model, self.tokenizer = self.build_llm_qlora(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        elif (
            self.pretrained_model in [
                PretrainedTrf.LLAMA3_8B_GPTQ,
                PretrainedTrf.LLAMA3_8B_INSTRUCT_GPTQ,
                PretrainedTrf.LLAMA_31_8B_GPTQ,
                PretrainedTrf.LLAMA_31_8B_INSTRUCT_GPTQ,
            ]
        ):
            self.model, self.tokenizer = self.build_llama_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        elif self.pretrained_model == PretrainedTrf.FIN_LLAMA:
            self.model, self.tokenizer = self.build_finance_llm_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        elif self.pretrained_model in [PretrainedTrf.FIN_LLAMA3]:
            self.model, self.tokenizer = self.build_llm_qlora(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )


        else:
            raise Exception(f"Non supported model : {self.pretrained_model}")

        self.tokenizer.padding_side = "left"

        self.hparams_text = f"{self.pretrained_model}" f"_lora_dp.{self.dropout_rate}"
        self.__name__ = f"textllm_sft_onlycplt_{self.hparams_text}"
        # self.__name__ = f"textllm_sft_{self.hparams_text}"


    def get_collator(self):
        """
        Get a collator obejct to allow to train only on completions tokens.
        """
        
        if "gemma" in self.pretrained_model :
            response_template = "<start_of_turn>model"
        elif self.pretrained_model == PretrainedTrf.FIN_LLAMA3:
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
        elif "llama" in self.pretrained_model:
            response_template = "<|start_header_id|>assistant<|end_header_id|>"
        else:
            
            raise Exception(f"Unspported model for sft training {self.pretrained_model}")
        
        from trl import DataCollatorForCompletionOnlyLM
        
        return DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        


    def get_sft_trainer(
        self,
        train_dataset,
        val_dataset,
        max_seq_length,
        training_args: TrainingArguments,
        callbacks=[],
    ) -> SFTTrainer:
        """
        Return an instance of the STFTrainer
        """
        
        
        
        

        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
              peft_config = LoraConfig(
                    lora_alpha=16, 
                    lora_dropout=self.dropout_rate,
                    r=4,
                    bias="none",
                     target_modules=[
                    "q_proj",
                    "v_proj",
                    #  "k_proj",
                    #  "o_proj",
                ],
                    task_type="CAUSAL_LM"),
            dataset_kwargs={
                "add_special_tokens": self.pretrained_model != PretrainedTrf.FIN_LLAMA3,
                "append_concat_token": False,
            },
            
            data_collator = self.get_collator(),
            args=training_args
        )
        
    

    def run_prediction(
        self, dataset, experiment_dir: str, mode: str = "val", epoch=0,
        batch_size=1,
        explanation_instruction = None,
        verbose=False
    ):
        """
        Run prediction on the dataset.
        """
        labels =  list(dataset.sic_to_title.values())
        labels_to_ids = {label: i for i, label in enumerate(labels)}
        labels_to_ids['none'] = len(labels)
        
        
        ids_to_labels = {i: label for label, i in labels_to_ids.items()}
        all_y_pred = []
        all_y_output = []
        all_y_true = []

        def getlabel(answer):
            for label in labels_to_ids.keys():
                if label in answer:
                    return label
            return "none"

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=8,
            temperature=0.01,
            batch_size = batch_size,
            return_full_text=False)
        
   
        
        
        sft_dataset = dataset.get_sft_dataset()    
        all_y_true = [labels_to_ids[d["label"]] for d in sft_dataset]
        all_y_true_text  = [d["label"] for d in sft_dataset]
        all_y_pred_text = []
        all_y_pred_explanation = []
        i = 0
        from transformers.pipelines.pt_utils import KeyDataset
        with torch.no_grad():
            from transformers.pipelines.pt_utils import KeyDataset
            
            all_outputs=pipe(KeyDataset(sft_dataset, "prompt_text"), 
                                 num_return_sequences=1,
                                batch_size=batch_size, 
                                )
            for out in tqdm(all_outputs,
                            desc= f"Running prediction on {mode} dataset"):
                all_y_output.append(out[0]["generated_text"])
                all_y_pred.append(labels_to_ids[getlabel(out[0]["generated_text"])])
                all_y_pred_text.append(getlabel(out[0]["generated_text"]))
                
                if verbose:
                    print(f"y_true: {ids_to_labels[all_y_true[i]]}, "
                          f"y_output: {all_y_output[i]}, "
                          f"y_pred: {ids_to_labels[all_y_pred[i]]}")
                i+=1
        
            
        all_y_true = np.array(all_y_true)    
        all_y_pred = np.array(all_y_pred)

        # Computing precision, recall, f1, mcc_score.
        metrics = self.compute_metrics(all_y_pred, all_y_true)
        metrics = {f"{mode}_{k}": v for k, v in metrics.items()}

        cm_writer = SummaryWriter(log_dir=experiment_dir)
        logger = pl.loggers.TensorBoardLogger(experiment_dir, name="", version="tb")

        class_names = self.class_names
        if "none" in all_y_pred_text:
            
            class_names = class_names + ["none"]


        cm_plot = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, class_names
        )
        cm_plot_normalized = NN_Utils.compute_confusion_matrix(
            all_y_true, all_y_pred, class_names, normalize=True
        )
        cm_writer.add_figure(
            f"Confusion Matrix  - {mode}",
            cm_plot,
            global_step=self.current_epoch,
        )
        cm_writer.add_figure(
            f"Confusion Matrix Normalized - {mode}", cm_plot_normalized
        )

        # Logs all the metrics :
        logger.log_metrics(metrics, step=epoch)
        

        print(f"Epoch {epoch}")
        print(metrics)
        
        
        if mode=="test":
            #Save results_test.yaml with metrics
            import pyaml
            with open(os.path.join(experiment_dir, "results_test.yaml"), "w") as f:
                pyaml.dump(metrics, f)
        
            #Save the dataframe with the results
            df = pd.DataFrame({"y_true": all_y_true_text, "y_pred": all_y_pred_text, "y_output": all_y_output})
            df.to_csv(os.path.join(experiment_dir, f"df_results_{mode}.csv"), index=False)
            
            if explanation_instruction is not None:
                del pipe
                pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                temperature=0.01,
                batch_size = batch_size,
                return_full_text=False)
                
                
                sft_dataset_explanation  = dataset.get_sft_dataset_with_explanation_prompt(explanation_instruction, all_y_pred_text)
                
                with torch.no_grad():
                    all_outputs_with_explanation=pipe(KeyDataset(sft_dataset_explanation, "text_with_explanation_prompt"), 
                                            num_return_sequences=1,
                                            batch_size=batch_size, 
                                            )
                    for out in tqdm(all_outputs_with_explanation, "Getting answer explanations"):
                        all_y_pred_explanation.append(out[0]["generated_text"])
                        if verbose:
                                print(f"y_pred_explanation: {all_y_pred_explanation[-1]}")    
                    
                    
                    #TO set the value with explanation
                    df["y_pred_explanation"] = all_y_pred_explanation
                    df.to_csv(os.path.join(experiment_dir, f"df_results_{mode}_with_explanation.csv"), index=False)
        
                
        return metrics

    def compute_metrics(self, y_pred, y_true):
        """
        Compute all metrics
        :param  y_pred log probabilities of classes: model's output.
        :param  y_true True labels
        """
        metrics_fn_dict = {
            "accuracy": 
                lambda y_pred, y_true: sklearn.metrics.accuracy_score(y_true, y_pred),
                
            "recall": lambda y_pred, y_true: sklearn.metrics.recall_score(
                y_true, y_pred,average="weighted"
            ),
            "precision": lambda y_pred, y_true: sklearn.metrics.precision_score(
                y_true, y_pred,average="weighted"
            ),
            
            "f1_macro": lambda y_pred, y_true: sklearn.metrics.f1_score(
                y_true, y_pred, average="macro"
            ),
            "f1_weighted": lambda y_pred, y_true: sklearn.metrics.f1_score(
                y_true, y_pred, average="weighted"
            ),
            "mcc":  lambda y_pred, y_true: sklearn.metrics.matthews_corrcoef(
                y_true, y_pred
            )
        }
        
        metrics = {}
        for metric_name, metric_fn  in metrics_fn_dict.items():
            metric_value = metric_fn(y_pred, y_true)
            metrics[metric_name] = metric_value.item() if (isinstance(metric_value, torch.Tensor) or isinstance(metric_value, np.ndarray)) else metric_value
        
        return metrics


    @staticmethod
    def load_best_model(experiment_dir):
        """
        Load the best model checkpoint for a given experiment.
        """
        from transformers.trainer_callback import TrainerState
        from peft import PeftModel   
        import accelerate
        accelerator = accelerate.Accelerator()
        model = TextLLMForInstructionSFT(n_accounts=-1, build_on_init=False)
        save_dir = experiment_dir
        experiments_dir = os.path.dirname(save_dir)
        experiment_name = os.path.basename(save_dir)
        ckpt_dirs = [f for f in os.listdir(save_dir) if "checkpoint" in f]

        # Get epoch of each checkpoint
        ckpt_per_epoch = {}
        for ckpt in ckpt_dirs:
            state = TrainerState.load_from_json(os.path.join(save_dir, ckpt, "trainer_state.json"))
            epoch = state.log_history[-1]["epoch"]
            ckpt_per_epoch[epoch] = ckpt

        # Get best checkpoint epoch
        best_model_epoch = ExperimentUtils.get_best_model(experiment_name, experiments_dir)["epoch"]
        
        # Return the first checkpoint with epoch >= best_model_epoch
        epoch = list(sorted([epoch for epoch in ckpt_per_epoch.keys() if epoch >= best_model_epoch]))[0]
        best_ckpt = ckpt_per_epoch[epoch]
        
        print("Loading model from checkpoint", os.path.basename(best_ckpt))
        best_ckpt = os.path.join(save_dir, best_ckpt)
        model.model = AutoModelForCausalLM.from_pretrained(
            best_ckpt,
            device_map={'': accelerator.process_index},
            torch_dtype=torch.float16
        )
        model.tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
        model.tokenizer.padding_side = "left"
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.model.config.pad_token_id = model.tokenizer.pad_token_id

        # Load lora model
        model.model = PeftModel.from_pretrained(model.model, best_ckpt)
        accelerator.prepare(model.model)
        return model
