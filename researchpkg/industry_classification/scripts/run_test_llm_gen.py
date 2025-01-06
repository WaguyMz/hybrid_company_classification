import argparse
import os
import sys

import torch

from researchpkg.industry_classification.dataset.sec_transformer_datamodule import (
    TextTransformerTemplateType,
)
from researchpkg.industry_classification.dataset.sec_transformer_sft_dataset import (
    SecTextTransformerDatasetSFT,
)
from researchpkg.industry_classification.models.transformers.text_llm_sft import (
    TextLLMForInstructionSFT,
)
from researchpkg.industry_classification.utils.sics_loader import load_sic_codes

dir_path = os.path.realpath(os.path.join(__file__, "..", "..", ".."))
sys.path.append(dir_path)

from researchpkg.industry_classification.config import (  # noqa
    LOGS_DIR,
    SEC_ROOT_DATA_DIR,
)
from researchpkg.industry_classification.dataset.sec_datamodule import (  # noqa
    SecDataset,
)
from researchpkg.industry_classification.models.transformers.pretrained_index import (  # noqa
    PretrainedTrf,
)
from researchpkg.industry_classification.utils.experiment_utils import (  # noqa
    ExperimentUtils,
)

parser = argparse.ArgumentParser(
    prog="Run Training of TextNum Transformer Model",
    description="Training TextNumTransformer model.",
)
parser.add_argument(
    "-g",
    "--global_exp_name",
    help="The global experiment name to use."
    " Should match one of existing generated datasets"
    "Example : _count30_sic1agg, _count30_sic1agg_with12ratios",
)

parser.add_argument(
    "-lr", "--learning_rate", type=float, default=1e-3, help="The learning rate"
)
parser.add_argument(
    "-bs", "--batch_size", type=int, default=128, help="The batch size of the model"
)
parser.add_argument(
    "-a",
    "--accelerator",
    default="cpu" if not torch.cuda.is_available() else "cuda",
    help="The accelerator to use",
)
parser.add_argument("-x", "--experiment_name", default="")
parser.add_argument(
    "-ptr",
    "--pretrained_text_transformer",
    default=PretrainedTrf.BERT_TINY,
    help="The pretrained transformer model to use",
)


parser.add_argument(
    "-w",
    "--num_workers",
    type=int,
    default=7,
    help="Dataloaders num workers",
)
parser.add_argument(
    "-m", "--load_in_memory", default=0, type=int, help="Load the dataset in memory"
)
parser.add_argument(
    "--sic_digits",
    default=1,
    type=int,
    help="The target features sic{sic_digits}. Ex: sic1 , sci2",
)

parser.add_argument(
    "-dp", "--dropout_rate", type=float, default=0.2, help="The dropout rate"
)
parser.add_argument("--balance_sampling", action="store_true", help="Use     sampling")
parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss")
parser.add_argument(
    "--max_tag_depth",
    type=int,
    default=None,
    help="The maximum depth of tags in the taxonomy to considerer.By default all tags are considered.",
    # noqa
)

parser.add_argument(
    "--template_type",
    default=TextTransformerTemplateType.DESCRIPTIVE,
    help="The template to use for the text transformer",
)
parser.add_argument(
    "-ga",
    "--gradients_accumulation_steps",
    type=int,
    default=1,
)
parser.add_argument(
    "-sl",
    "--seq_max_length",
    type=int,
    default=None,
    help="The maximum length of the sequence",
)

parser.add_argument(
    "--add_explanation",
    action="store_true",
    help="Prompt the llm to generation and explanation of its answer."
)
parser.add_argument(
    "--zero_shot",
    action="store_true",
)

def train(
    global_exp_name: str,
    learning_rate: float,
    batch_size: int,
    accelerator: str,
    experiment_name: str,
    pretrained_text_transformer: str,
    num_workers: int,
    load_in_memory: bool,
    sic_digits: int,
    dropout_rate: float,
    balance_sampling: bool,
    weighted_loss: bool,
    max_tag_depth: int,
    template_type: TextTransformerTemplateType,
    gradients_accumulation_steps: int,
    seq_max_length: int,
    add_explanation: bool,
    zero_shot: bool,
    
):
    """
    Run the training of the balance sheet classification model.
    """
    labels =[ "Mining", "Construction", "Manufacturing", "Transportation & Public Utilities", "Wholesale Trade", "Retail Trade", "Finance", "Services"]
    
#     labels_subtext = f"""
    
# - Manufacturing
# - Transportation & Public Utilities
# - Finance
# - Services
# """

    if "gemma" in pretrained_text_transformer:

        def partial_instruction_formatter(prompt) -> str:
            labels_subtext = "\n".join([f"- {l}" for l in labels])
            return (
                "<start_of_turn>user You are asked to predict the industry sector "
                "of a company based on its balance sheet and income statement.\n"
                "The value of the accounts are normalized by the total assets and given in percentage of totals assets.\n "
                "Given the provided informations about the balance sheet and income statement, "
                "you should predict the most probable industry sector of the "
                "related company.\n"
                "You should answer on a single line with only the name of the predicted "
                "industry sector and  nothing else.\n"
                "Here are the possible industry sectors: \n\n"
                f"{labels_subtext}\n"
                "You must strictly respect the spelling of the predicted industry sector.\n"
                "\n<end_of_turn>\n"
                "<start_of_turn> user \n"
                f"{prompt}\n<end_of_turn>\n"
                "<start_of_turn>model \n"
                "Based on the information provided, the most probable industry sector of the company is: \n"
            )

        def instruction_formatter(prompt, label) -> str:

            return partial_instruction_formatter(prompt) + f"{label} <end_of_turn> \n"
        
        explanation_instruction = None
        if add_explanation:
            explanation_instruction = "<start_of_turn>user Please provide a justification of  your answer. <end_of_turn>\n"
            explanation_instruction += "<start_of_turn>model The justification for the answer is as follows: \n"
    
    elif pretrained_text_transformer == PretrainedTrf.FIN_LLAMA3:
        def partial_instruction_formatter(prompt) -> str:
            labels_subtext = "\n".join([f"- {l}" for l in labels])
            return (
                "Use this fact to answer the question : "
                f"\n{prompt}\n"
                "Given the provided informations about the balance sheet and income statement, "
                "what is the most probable industry sector of the ""related company.\n"
                "Here are the possible industry sectors: \n"
                f"{labels_subtext}\n"
                "Given the information provided, the most probable industry sector of this company is : <|start_header_id|>assistant<|end_header_id|>")
                
            
        def instruction_formatter(prompt, label) -> str:

            return partial_instruction_formatter(prompt) + f"{label}"

        explanation_instruction = None
        if add_explanation:
            explanation_instruction = "\n\n The justification for the answer is as follows: \n"
    

    elif "llama" in pretrained_text_transformer:

        def partial_instruction_formatter(prompt) -> str:
            labels_subtext = "\n".join([f"- {l}" for l in labels])
            return (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                "You are asked to predict the industry sector "
                "of a company based on its balance sheet and income statement.\n"
                "The value of the accounts are normalized by the total assets and given in percentage of totals assets.\n "
                "Given the provided informations about the balance sheet and income statement, "
                "you should predict the most probable industry sector of the "
                "related company.\n"
                "You should answer on a single line with the name of the predicted "
                "industry sector and \n"
                "Here are the possible industry sectors: \n"
                f"{labels_subtext}\n"
                "\n\n You must strictly respect the spelling of the predicted industry sector.\n"
                "<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|> {prompt} <|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|> Based on the information provided, the most probable industry sector of the company is: \n"
            )

        def instruction_formatter(prompt, label) -> str:

            return partial_instruction_formatter(prompt) + f"{label} <|eot_id|>"

        explanation_instruction = None
        if add_explanation:
            explanation_instruction = "<|start_header_id|>user<|end_header_id|> Please provide a justification of  your answer. <|eot_id|>\n"
            explanation_instruction += "<|start_header_id|>assistant<|end_header_id|>  The justification for the answer is as follows: \n"

    ExperimentUtils.check_global_experiment_name(global_exp_name)
    dataset_dir = os.path.join(SEC_ROOT_DATA_DIR, global_exp_name)
    experiments_dir = os.path.join(LOGS_DIR, f"experiments_{global_exp_name}")
    
    
        

    accounts_index, registrants_index, _ = SecDataset.load_index(
        dataset_dir, sic_digits=sic_digits
    )

    labels = list(sorted(registrants_index[f"sic{sic_digits}"].unique().tolist()))

    sic_code_df = load_sic_codes()[["sic", "industry_title"]]
    labels = [
        sic_code_df[sic_code_df["sic"] == l]["industry_title"].values[0] for l in labels
    ]
    n_labels = len(labels)

    n_accounts = len(accounts_index)

    model = TextLLMForInstructionSFT(
        n_accounts=n_accounts,
        pretrained_model=pretrained_text_transformer,
        n_classes=n_labels,
        class_names=labels,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    experiment_name = f"{model.__name__}" f"_{experiment_name}_sic{sic_digits}"
    if zero_shot:
        experiment_name += "_zero_shot"

    if balance_sampling:
        experiment_name += "_balanced"

    if weighted_loss:
        experiment_name += "_weighted_loss"

    if max_tag_depth is not None:
        experiment_name += f"_max_tag_depth_{max_tag_depth}"
        
    

    experiment_name += f"_template_{template_type}"

    experiment_dir = os.path.join(experiments_dir, experiment_name)
    # 3. Trainer
    seq_max_length = seq_max_length or PretrainedTrf.get_context_window(
        pretrained_text_transformer
    )


    test_dataset = SecTextTransformerDatasetSFT(
        type="test",
        dataset_dir=dataset_dir,
        tokenizer=model.tokenizer,
        sic_digits=sic_digits,
        seq_max_length=seq_max_length,
        balance_sampling=balance_sampling,
        weighted_loss=weighted_loss,
        max_tag_depth=max_tag_depth,
        template_type=template_type,
        load_in_memory=load_in_memory,
        instruction_formatter=instruction_formatter,
        partial_instruction_formatter=partial_instruction_formatter,
        
    )
    
    ######################################
    ## GETTING BEST MODEL CHECKPOINT DIR##
    ######################################
    if zero_shot:
        #Create the experiment directory
        ExperimentUtils.initialize_experiment(
            experiment_dir,
            dataset_dir,
            model.hparams,
            training_config={
                "batch_size": batch_size,
                "device": accelerator,
                "ngpus": torch.cuda.device_count() if accelerator == "cuda" else 0,
                "epochs": 0,
                "seq_max_length": seq_max_length,
    
            },
        )
    else:
        model = TextLLMForInstructionSFT.load_best_model(experiment_dir)
    
    #First save the sft_dataset in the experiment_dir (to able to read raw text an labels.)
    sft_dataset = test_dataset.get_sft_dataset()
    sft_dataset.to_csv(os.path.join(experiment_dir, "test_sft_dataset.csv"), index=False)
    
    return 
    
    model.run_prediction(
        test_dataset, experiment_dir, mode="test", epoch=0, batch_size=batch_size,
        explanation_instruction = explanation_instruction, verbose=False
        )
    
    
if __name__ == "__main__":
    print("Run eval")
    args: argparse.Namespace = parser.parse_args()
    # Call with kwargs
    train(**vars(args))
