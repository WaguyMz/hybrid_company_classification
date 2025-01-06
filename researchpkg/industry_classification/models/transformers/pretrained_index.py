from transformers import AutoTokenizer, BertModel, DistilBertModel, GPT2Model, AutoModel

class PretrainedTrf:
    BERT_BASE = "bert_base"
    BGE_BASE = "bge_base"
    DISTILBERT_BASE = "distilbert_base"
    SEC_BERT = "sec_bert"
    SBERT = "sbert"
    FIN_LANG = "finlang"
    GPT2 = "gpt2"
    DISTIL_GPT2 = "distil_gpt2"
    GPT2_LLM = "gt2_llm"
    FALCON_1B = "falcon_1b"
    LLAMA2_7B = "llama2_7b"
    VICUNA_1B = "vicuna_1b"
    VICUNA_1B_GPTQ = "vicuna_1b_gptq"
    LLAMA2_7B_GPTQ = "llama2_7b_gptq"
    LLAMA2_7B_AWQ = "llama2_7b_awq"
    LLAMA3_8B_GPTQ = "llama3_8b_gptq"
    LLAMA3_8B_INSTRUCT_GPTQ = "llama3_8b_instruct_gptq"
    LLAMA2_8B = "llama2_8b"
    LLAMA_31_8B_GPTQ = "llama31_8b_gptq"
    LLAMA_31_8B_INSTRUCT_GPTQ = "llama31_8b_instruct_gptq"
    FIN_LLAMA = "fin_llama"
    FIN_LLAMA3 = "fin_llama3"
    PHI = "microsoft/phi-1_5"
    LLAMA3_8B = "llama3_8b"
    GEMMA_2B_GPTQ = "gemma_2b_gptq"
    GEMMA_2B_INSTRUCT_GPTQ = "gemma_2b_instruct_gptq"
    GEMMA_9B_QLORA = "gemma_9b_qlora"
    GEMMA_9B_GPTQ = "gemma_9b_gptq"
    GEMMA_9B_INSTRUCT_GPTQ = "gemma_9b_instruct_gptq"
    GEMMA_9B_INSTRUCT_UNSLOTH = "gemma_9b_instruct_unsloth"
    MODEL_INDEX = {
        BERT_BASE: {
            "link": "bert-base-uncased",
            "class": BertModel,
        },
        BGE_BASE : {
            "link": "BAAI/bge-base-en",
            "class": AutoModel,
        },
        SBERT: {
            "link": "sentence-transformers/all-MiniLM-L6-v2",
            "class": BertModel},
        DISTILBERT_BASE: {"link": "distilbert-base-uncased", "class": DistilBertModel},
        SEC_BERT: {"link": "nlpaueb/sec-bert-shape", "class": BertModel},
        FIN_LANG: {"link": "FinLang/finance-embeddings-investopedia", "class": BertModel},
        GPT2: {"link": "gpt2", "class": GPT2Model},
        DISTIL_GPT2: {"link": "gpt2", "class": GPT2Model},
        FALCON_1B: {"link": "tiiuae/falcon-rw-1b"},
        LLAMA2_7B: {
            "link": "meta-llama/Llama-2-7b-hf",
        },
        LLAMA3_8B: {
            "link": "meta-llama/Meta-Llama-3-8B",
        },
        FIN_LLAMA3: {
            "link":"instruction-pretrain/finance-Llama3-8B",
        },
        VICUNA_1B: {
            "link": "Jiayi-Pan/Tiny-Vicuna-1B",
        },
        VICUNA_1B_GPTQ: {
            "link": "TheBloke/vicuna-7B-v0-GPTQ",
        },
        LLAMA2_7B_GPTQ: {
            "link": "TheBloke/Llama-2-7B-GPTQ",
        },
        LLAMA2_7B_AWQ: {
            "link": "TheBloke/Llama-2-7B-AWQ",
        },
        FIN_LLAMA: {
            "link": "TheBloke/finance-LLM-GPTQ",
        },
        PHI: {"link": "microsoft/phi-1_5"},
        LLAMA3_8B_GPTQ: {
            "link": "TechxGenus/Meta-Llama-3-8B-GPTQ",
        },
        LLAMA3_8B_INSTRUCT_GPTQ: {
            "link": "TechxGenus/-Llama-3-8B-Instruct-GPTQ",
        },
        
        GEMMA_2B_GPTQ: {"link": "TechxGenus/gemma-2b-GPTQ"},
        GEMMA_2B_INSTRUCT_GPTQ: {"link": "TechxGenus/gemma-1.1-2b-it-GPTQ"},
        GEMMA_9B_GPTQ: {"link": "ModelCloud/gemma-2-9b-gptq-4bit",
                   },
        GEMMA_9B_QLORA: {"link": "google/gemma-2-9b-it",},
        GEMMA_9B_INSTRUCT_GPTQ: {"link": "ModelCloud/gemma-2-9b-it-gptq-4bit"},
        
        GEMMA_9B_INSTRUCT_UNSLOTH: {"link": "unsloth/gemma-2-9b-bnb-4bit"},
        
        LLAMA_31_8B_GPTQ: {"link": "shuyuej/Meta-Llama-3.1-8B-GPTQ"},
        LLAMA_31_8B_INSTRUCT_GPTQ: {"link": "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"},
        
    }
    
    @staticmethod
    def get_context_window(model_name: str) -> int:
        """
        Return the context window size of the model
        """
        if model_name in [
            PretrainedTrf.BERT_TINY,
            PretrainedTrf.BERT_MINI,
            PretrainedTrf.BERT_BASE,
            PretrainedTrf.DISTILBERT_BASE,
            PretrainedTrf.SEC_BERT,
            PretrainedTrf.FIN_LANG,
            PretrainedTrf.SBERT,
            PretrainedTrf.BGE_BASE
        ]:
            return 512

        elif model_name in [PretrainedTrf.DISTIL_GPT2, PretrainedTrf.GPT2]:
            return 1024

        elif model_name in [
            PretrainedTrf.FALCON_1B,
            PretrainedTrf.FIN_LLAMA,
            PretrainedTrf.VICUNA_1B,
            PretrainedTrf.VICUNA_1B_GPTQ,
        ]:
            return 2048

        elif model_name in [
            PretrainedTrf.LLAMA2_7B,
            PretrainedTrf.LLAMA2_8B,
            PretrainedTrf.LLAMA2_7B_GPTQ,
            PretrainedTrf.LLAMA3_8B_GPTQ,
            PretrainedTrf.LLAMA3_8B_INSTRUCT_GPTQ,
            PretrainedTrf.LLAMA2_7B_AWQ,
            PretrainedTrf.PHI,
            PretrainedTrf.GEMMA_2B_GPTQ,
            PretrainedTrf.GEMMA_2B_INSTRUCT_GPTQ,
            PretrainedTrf.GEMMA_9B_GPTQ,
            PretrainedTrf.GEMMA_9B_QLORA,
            PretrainedTrf.GEMMA_9B_INSTRUCT_GPTQ,
            PretrainedTrf.GEMMA_9B_INSTRUCT_UNSLOTH,
            PretrainedTrf.LLAMA_31_8B_GPTQ,
            PretrainedTrf.LLAMA_31_8B_INSTRUCT_GPTQ,
        ]:

            # return 4096 #For now we  don't use the full context window
            return 2048
        else:
            raise Exception(f"No context window size for model {model_name}")

    @staticmethod
    def load(model_name):
        if not model_name in PretrainedTrf.MODEL_INDEX:
            raise Exception(f"Non supported model : {model_name}")

        model_info = PretrainedTrf.MODEL_INDEX[model_name]

        ModelClass = model_info["class"]
        model_link = model_info["link"]
        model = ModelClass.from_pretrained(model_link)

        tokenizer = AutoTokenizer.from_pretrained(model_link, use_fast=True)

        if PretrainedTrf.is_gpt(model_name):
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        return model, tokenizer

    @staticmethod
    def is_llm(model_name):
        return model_name in [
            PretrainedTrf.LLAMA2_7B,
            PretrainedTrf.FALCON_1B,
            PretrainedTrf.VICUNA_1B,
            PretrainedTrf.VICUNA_1B_GPTQ,
            PretrainedTrf.LLAMA2_7B_GPTQ,
            PretrainedTrf.LLAMA2_7B_AWQ,
            PretrainedTrf.LLAMA3_8B,
            PretrainedTrf.LLAMA3_8B_GPTQ,
            PretrainedTrf.LLAMA3_8B_INSTRUCT_GPTQ,
            
            PretrainedTrf.PHI,
            PretrainedTrf.FIN_LLAMA,
            PretrainedTrf.FIN_LLAMA3,
            PretrainedTrf.GEMMA_2B_GPTQ,
            PretrainedTrf.GEMMA_2B_INSTRUCT_GPTQ,
            PretrainedTrf.GEMMA_9B_GPTQ,
            PretrainedTrf.GEMMA_9B_INSTRUCT_GPTQ,
            PretrainedTrf.GEMMA_9B_INSTRUCT_UNSLOTH,
            PretrainedTrf.GEMMA_9B_QLORA,
            PretrainedTrf.LLAMA_31_8B_GPTQ,
            PretrainedTrf.LLAMA_31_8B_INSTRUCT_GPTQ,
        ]

    @staticmethod
    def is_gpt(model_name):
        return model_name in [
            PretrainedTrf.GPT2,
            PretrainedTrf.DISTIL_GPT2,
            PretrainedTrf.GPT2_LLM,
        ]
