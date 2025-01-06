from typing import Dict, Tuple, Union

import huggingface_hub
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from researchpkg.industry_classification.dataset.numero_logic_formatter import update_model_tokenizer_and_embedding
from researchpkg.industry_classification.models.transformers.pretrained_index import (
    PretrainedTrf,
)
from researchpkg.industry_classification.models.transformers.text_transformer import (
    TextTransformerForClassification,
)


class TextLlmForClassification(TextTransformerForClassification):
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
        use_ia3=False,
        **kwargs,
    ):
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
        self.model : AutoModelForSequenceClassification = None
        self.tokenizer : AutoTokenizer = None
        self.kwargs = kwargs
        if build_on_init:
            self.build_model()


    def build_llama(self, model_config: dict, layers_to_transform=[31]) -> Tuple: 
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name, trust_remote_code=True, num_labels=self.n_classes
        )
        # modules_to_save = []
        # for name, _ in llm.named_parameters():
        #     if "embed" in name:
        #         modules_to_save.append(name)

        if self.use_ia3:
            from peft import IA3Config

            peft_config = IA3Config(task_type=TaskType.SEQ_CLS)

        else:
            peft_config = LoraConfig(
                TaskType.SEQ_CLS,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    #  "k_proj",
                    #  "o_proj",
                ],
                # modules_to_save=modules_to_save,
                layers_to_transform=layers_to_transform,
                inference_mode=False,
                r=4,
                lora_alpha=16,
                lora_dropout=self.dropout_rate,
            )

        model = get_peft_model(llm, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def build_gemma(self, model_config: dict) -> Tuple: 
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name, trust_remote_code=True, num_labels=self.n_classes
        )
        # modules_to_save = []
        # for name, _ in llm.named_parameters():
        #     if "embed" in name:
        #         modules_to_save.append(name)

        # print(llm)
        if self.use_ia3:
            from peft import IA3Config

            peft_config = IA3Config(task_type=TaskType.SEQ_CLS)

        else:
            peft_config = LoraConfig(
                TaskType.SEQ_CLS,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    #  "k_proj",
                    #  "o_proj",
                ],
                inference_mode=False,
                layers_to_transform= [k for k in range(30,41)],
                r=4,
                lora_alpha=16,
                lora_dropout=self.dropout_rate,
            )

        model = get_peft_model(llm, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

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


        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        

        quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            config=quantization_config_loading,
            device_map="auto",
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
        )
        
        target_modules = ["q_proj", "v_proj"]
        use_numero_logic_ = self.kwargs.get("use_numero_logic", False)
        if use_numero_logic_:
            update_model_tokenizer_and_embedding(llm,tokenizer)
            modules_to_save = []
            target_modules.append("embed_tokens")
            target_modules.append("lm_head")
            print("Trainable embeddings")
        else:
            modules_to_save = None
        
        if self.use_ia3:
            from peft import IA3Config

            peft_config = IA3Config(task_type=TaskType.SEQ_CLS)
            model = get_peft_model(llm, peft_config)
        else:
            peft_config = LoraConfig(
                TaskType.SEQ_CLS,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
                inference_mode=False,
                r=4,
                lora_alpha=16,
                lora_dropout=self.dropout_rate,
            )
            model = get_peft_model(llm, peft_config)
        
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
        # from peft import prepare_model_for_kbit_training
        # model = prepare_model_for_kbit_training(model)
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

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

        quantization_config_loading = GPTQConfig(bits=4, disable_exllama=True)
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            config=quantization_config_loading,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        if self.use_ia3:
            from peft import IA3Config

            peft_config = IA3Config(task_type=TaskType.SEQ_CLS)
            model = get_peft_model(llm, peft_config)
        else:
            peft_config = LoraConfig(
                TaskType.SEQ_CLS,
                target_modules=[
                    "q_proj",
                    "v_proj",
                ],
                inference_mode=False,
                r=4,
                lora_alpha=16,
                lora_dropout=self.dropout_rate,
            )
            model = get_peft_model(llm, peft_config)
        # from peft import prepare_model_for_kbit_training
        # model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def build_finance_llm_gptq(self, model_config: dict) -> Tuple:
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]

        if "hf_token" in model_config:
            # login to huggingface hub
            hf_token = model_config["hf_token"]
            huggingface_hub.login(token=hf_token)

        # from accelerate import infer_auto_device_map
        # QLoRa fine-tuning:

        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            device_map="auto",
        )
        peft_config = LoraConfig(
            TaskType.SEQ_CLS,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            inference_mode=False,
            r=4,
            lora_alpha=16,
            lora_dropout=self.dropout_rate,
        )

        model = get_peft_model(llm, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

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
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name,
            trust_remote_code=True,
            num_labels=self.n_classes,
            quantization_config=quantization_config,

        )
        
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        llm.config.pad_token_id = tokenizer.pad_token_id
        
        
        peft_config = LoraConfig(
                TaskType.SEQ_CLS,
                target_modules=[
                    "q_proj",
                    "v_proj",
                ],
                inference_mode=False,
                r=4,
                lora_alpha=16,
                lora_dropout=self.dropout_rate,
            )
        model = get_peft_model(llm, peft_config)
        # from peft import prepare_model_for_kbit_training
        # model = prepare_model_for_kbit_training(model)
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer
        
    
    def build_falcon(self, model_config: dict) -> Tuple:
        """
        Build the Llama model.
        """
        repo_name = model_config["link"]
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name, trust_remote_code=True, num_labels=self.n_classes
        )
        peft_config = LoraConfig(
            TaskType.SEQ_CLS,
            target_modules=[
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h",
            ],
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
        )
        model = get_peft_model(llm, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def build_gpt2_llm(self, model_config: dict) -> Tuple:
        repo_name = model_config["link"]
        llm = AutoModelForSequenceClassification.from_pretrained(
            repo_name, trust_remote_code=True, num_labels=self.n_classes
        )
        peft_config = LoraConfig(
            TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=self.dropout_rate,
        )
        model = get_peft_model(llm, peft_config)
        tokenizer = AutoTokenizer.from_pretrained(repo_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def build_model(self):
        """
        Build the model.
        """

        if self.pretrained_model in [PretrainedTrf.LLAMA2_7B, PretrainedTrf.LLAMA3_8B,]:
            self.model, self.tokenizer = self.build_llama(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        
        elif self.pretrained_model in [
            PretrainedTrf.VICUNA_1B,
        ]:
            self.model, self.tokenizer = self.build_llama(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model],
                layers_to_transform=[18, 19, 20, 21],
            )
        elif self.pretrained_model in [
            PretrainedTrf.LLAMA2_7B_GPTQ,
            PretrainedTrf.LLAMA3_8B_GPTQ,
            PretrainedTrf.LLAMA_31_8B_GPTQ,
            PretrainedTrf.LLAMA_31_8B_INSTRUCT_GPTQ,
        ]:

            self.model, self.tokenizer = self.build_llama_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )

        elif self.pretrained_model in [
            PretrainedTrf.VICUNA_1B_GPTQ,
        ]:

            self.model, self.tokenizer = self.build_llama_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model],
                layers_to_transform=[20, 21],
            )

        elif self.pretrained_model == PretrainedTrf.FALCON_1B:
            self.model, self.tokenizer = self.build_falcon(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        elif self.pretrained_model == PretrainedTrf.FIN_LLAMA:
            self.model, self.tokenizer = self.build_finance_llm_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        elif (
            self.pretrained_model == PretrainedTrf.GEMMA_2B_GPTQ
            or self.pretrained_model == PretrainedTrf.GEMMA_2B_INSTRUCT_GPTQ
            or self.pretrained_model == PretrainedTrf.GEMMA_9B_GPTQ
        ):
            self.model, self.tokenizer = self.build_gemma_gptq(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
        
        elif     self.pretrained_model == PretrainedTrf.FIN_LLAMA3:
            self.model, self.tokenizer = self.build_llm_qlora(
                PretrainedTrf.MODEL_INDEX[self.pretrained_model]
            )
    
       
        else:
            raise Exception(f"Non supported model : {self.pretrained_model}")

        self.tokenizer.padding_side = "left"

        self.hparams_text = f"{self.pretrained_model}" f"_lora_dp.{self.dropout_rate}"
        self.__name__ = f"textllm_{self.hparams_text}"

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

        # The llm directly outputs the logits
        logits = self.model(input_ids=input, attention_mask=input_attn_mask).logits
        probs = torch.log_softmax(logits, dim=-1)

        return probs
