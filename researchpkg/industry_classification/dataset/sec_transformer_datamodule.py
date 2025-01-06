import multiprocessing
import os
import re
from enum import Enum
from typing import Any, Callable, Dict, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn
import torch.utils
import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from researchpkg.industry_classification.config import (
    MAX_CORE_USAGE,
    SEC_TAX,
    SEC_TAX_DATA_DIR,
    SEC_TAX_VERSION,
)
from researchpkg.industry_classification.dataset.numero_logic_formatter import (
    numerologic_preprocess_all_numbers,
)
from researchpkg.industry_classification.dataset.sec_datamodule import SecDataset
from researchpkg.industry_classification.dataset.utils import DatasetType
from researchpkg.industry_classification.preprocessing.gaap_taxonomy_parser import (
    CalculationTree,
    CalculationTreeType,
)
from researchpkg.industry_classification.preprocessing.sec_preprocessing_utils import (
    RATIOS_12,
)


class TextTransformerTemplateType(str, Enum):
    DESCRIPTIVE_LOG = "DESCRIPTIVE_LOG"  # Decriptive log values of accounts.
    DESCRIPTIVE_VERBOSE = "DESCRIPTIVE_VERBOSE"
    DESCRIPTIVE_SORTED = "DESCRIPTIVE_SORTED"
    RAW = "RAW"  # Raw amounts
    DESCRIPTIVE_WITH_RAW = "DESCRIPTIVE_WITH_RAW"  # Descriptive  + Raw amounts
    DESCRIPTIVE_WITH_RATIOS = (
        "DESCRIPTIVE_WITH_RATIOS"  # Descriptive text but no scaling
    )
    DESCRIPTIVE_WITH_MISSING = "DESCRIPTIVE_WITH_MISSING"
    COMPARATIVE = "COMPARATIVE"
    MIXED = "MIXED"
    INCOME_RATIO = "INCOME_RATIO"
    DESCRIPTIVE = "DESCRIPTIVE"
    DESCRIPTIVE_BREADTH = "DESCRIPTIVE_BREADTH"
    DESCRIPTIVE_NUMLOGIC = "DESCRIPTIVE_NUMLOGIC"
    DESCRIPTIVE_WITH_ASSETS = "DESCRIPTIVE_WITH_ASSETS"
    DESCRIPTIVE_ROUND_ZERO = "DESCRIPTIVE_ROUND_ZERO"  # Consider only the interfer part of account percentage, account with 0% are not considered
    NO_CHANGE = "NO_CHANGE"  # Only tags with no change


class SecTextTransformerDataset(SecDataset):
    """
    Convert full balance sheet to a single sequence of tokens.
    """

    def __init__(
        self,
        dataset_dir: str,
        type: DatasetType,
        tokenizer,
        sic_digits=1,
        seq_max_length=512,
        max_tag_depth=None,
        min_tag_depth=None,
        balance_sampling=False,
        weighted_loss=False,
        is_encoder_transformer=True,
        template_type=TextTransformerTemplateType.DESCRIPTIVE,
        max_comparative_pair_depth_gap=2,
        load_in_memory=False,
        return_raw=False,
        prompt_formatter: Callable[[str], str] = None,
        bread_first_tree_exploration: bool = False,
    ):
        """
        Dataset for sic classification using a transformer model combining text and
        amounts. :param dataset_dir: The directory of the dataset :param type: The
        type of the dataset (train, val or test) :param tokenizer: The tokenizer to
        use :param sic_digits: The number of digits of the sic code (Not really used
        for now) :param seq_max_length: The maximum length of the sequence :param
        max_tag_depth: The maximum depth of the tags to consider. If None, all tags
        are considered. :param balance_sampling: If True, use a balanced sampler
        :param weighted_loss: If True, use a weighted loss :param
        is_encoder_transformer: If True, use an encoder transformer, otherwise use a
        decoder transformer :param template_type: The type of template to use (
        descriptive or comparative or mixed) :param max_comparative_pair_depth_gap:
        The maximum gap between the depth of the tags in a comparative pair.

        """
        super().__init__(dataset_dir, type, sic_digits=sic_digits, use_aggregate=True)
        self.bread_first_tree_exploration = bread_first_tree_exploration
        self.sic_digits = sic_digits
        self.sic_col = f"sic{sic_digits}"
        self.tokenizer = tokenizer
        self.registrants_index_dict = self.registrants_index.astype(
            {self.sic_col: int}
        )[self.sic_col].to_dict()
        self.filter_data()
        self.account_descriptions_index = self.accounts_index["description"].to_dict()
        self.all_data_dict = {}
        self.data_in_memory = False
        self.label_weights: Union[list, None] = None
        self.max_tag_depth = max_tag_depth
        self.min_tag_depth = min_tag_depth

        self.load_in_memory = load_in_memory
        ##Load tags_index
        tag_index_file = os.path.join(dataset_dir, "index", "tags_index.csv")
        self.tags_index = pd.read_csv(tag_index_file, index_col=None)[
            ["tag", "tlabel", "tag_depth"]
        ].drop_duplicates(
            subset="tag"
        )  # type: ignore
        self.max_seq_length = seq_max_length
        # Max Depth of the tags
        if self.max_tag_depth is None and self.min_tag_depth is None:
            pass
        else:
            max_depth = self.max_tag_depth or 1000
            min_depth = self.min_tag_depth or -1

            # If max_tag_depth is not None, then we only keep the first max_tag_depth tags
            relevant_tags = self.tags_index.query(
                f"(tag_depth <= {max_depth} & tag_depth != -1 & tag_depth>= {min_depth}) | tag == 'Assets'"
            )["tag"].values.tolist()
            print(
                f"Max depth : {self.max_tag_depth}, Min depth : {self.min_tag_depth}, Number of tags : {len(relevant_tags)}"
            )

        self.is_encoder_transformer = is_encoder_transformer
        ##Compute class weights
        if balance_sampling or weighted_loss:
            print("Computing label weights")
            self.compute_label_weights()
        self.balance_sampling = balance_sampling
        self.weighted_loss = weighted_loss

        # Template type
        self.template_type = template_type

        # Taxonomy tree
        self.bs_taxonomy_tree = CalculationTree.build_taxonomy_tree(
            SEC_TAX_DATA_DIR,
            SEC_TAX,
            SEC_TAX_VERSION,
            type=CalculationTreeType.BALANCE_SHEET,
        )
        self.is_taxonomy_tree = CalculationTree.build_taxonomy_tree(
            SEC_TAX_DATA_DIR,
            SEC_TAX,
            SEC_TAX_VERSION,
            type=CalculationTreeType.INCOME_STATEMENT,
        )

        # 2. Sampling probas( invertily proportional to the number of tag of the same depth)
        self.tag_depth_count = self.bs_taxonomy_tree.depth_count_dict()
        self.tag_depth_dict = self.tags_index.set_index("tag")["tag_depth"].to_dict()

        # Max comparative pair depth gap
        self.max_comparative_pair_depth_gap = max_comparative_pair_depth_gap

        self.prompt_formatter = prompt_formatter
        print("Dataset Size : ", len(self.data_files))
        self.return_raw = return_raw
        # Load all data in memorydescriptive_
        if load_in_memory:
            self.load_all_in_memory()
            self.data_in_memory = True

    def filter_data(self):
        max_core = MAX_CORE_USAGE
        num_cores = min(max_core, multiprocessing.cpu_count() - 2)
        chunk_size = len(self.data_files) // num_cores

        def filter_chunk(chunk):
            data_files_filtered = []
            for f in tqdm.tqdm(chunk, "Filtering dataset"):
                cik = pd.read_csv(f, index_col=None, usecols=["cik"], nrows=2)["cik"][0]
                if cik not in self.registrants_index_dict:
                    continue
                data_files_filtered.append(f)
            return data_files_filtered

        data_files_chunks = Parallel(n_jobs=num_cores)(
            delayed(filter_chunk)(self.data_files[i : i + chunk_size])
            for i in range(0, len(self.data_files), chunk_size)
        )
        self.data_files = []
        for f in data_files_chunks:
            self.data_files += f

    def get_descriptive_account_text_in_logscaling(self, account_name, account_value):
        """
        Get the comparator text for the account.
        Example :
        Assets are greater than one million
        """

        def value_to_string(value):
            if value >= 1e10:
                return "ten billion dollars"
            elif value >= 1e9:
                return "one billion dollars"
            elif value >= 1e8:
                return "one hundred million dollars"
            elif value >= 1e7:
                return "ten million dollars"
            elif value >= 1e6:
                return "one million dollars"
            elif value >= 1e5:
                return "one hundred thousand dollars"
            elif value >= 1e4:
                return "ten thousand dollars"
            elif value >= 1e3:
                return "one thousand dollars"
            elif value >= 1e2:
                return "one hundred dollars"
            elif value >= 1e1:
                return "ten dollars"
            elif value >= 1e0:
                return "one dollar"
            else:
                return str(value) + " dollars"

        abs_value = abs(account_value)
        sign = -1 if account_value < 0 else 1

        threshold = 10 ** int(abs_value)
        threshold = value_to_string(threshold)

        if sign == -1:
            threshold = "-" + str(threshold)
        op = ">" if sign == 1 else "<"
        return f"{account_name} is {op} than {threshold}"

    def get_descriptive_account_text_in_percentage(
        self, splitted_account_name, account_amount
    ):
        """ """

        sign = -1 if account_amount < 0 else 1
        abs_value = abs(account_amount)
        abs_value = round(abs_value, 2)

        text = str(abs_value) + "%"
        sign_text = "" if sign == 1 else "-"
        # return f"{account_name} is {sign_text} {text}"
        return f"{splitted_account_name} = {sign_text} {text}" + "."

    def get_descriptive_account_text_in_percentage_round_zero(
        self, splitted_account_name, account_amount
    ):
        """ """

        sign = -1 if account_amount < 0 else 1
        abs_value = abs(account_amount)
        abs_value = int(abs_value)

        # if abs_value == 0:
        #     return ""

        text = str(abs_value) + "%"
        sign_text = "" if sign == 1 else "-"
        return f"{splitted_account_name} = {sign_text} {text}" + "."

    def get_descriptive_account_text_in_log(
        self, splitted_account_name, account_amount
    ):
        """ """

        sign = -1 if account_amount < 0 else 1
        abs_value = abs(account_amount)
        abs_value = round(abs_value, 2)

        text = str(abs_value)
        sign_text = "" if sign == 1 else "-"
        # return f"{account_name} is {sign_text} {text}"
        return f"{splitted_account_name} = {sign_text}{text}" + "."

    def get_full_text_no_change(self, df):
        """
        Get only the presents tag of dataframe with depth >=4
        """
        # Only tag with depth >=2
        df = df.query("tag_depth >= 2").copy()
        # split by capital letter
        tags = [self.split_by_capital_letter(tag) for tag in df["tag"].values.tolist()]
        full_text = ". \n".join(tags)
        full_text = "The list of tags in this report are : \n" + full_text
        return full_text

    def get_descriptive_account_text_in_percentage_verbose(
        self, splitted_account_name, account_amount
    ):
        sign = -1 if account_amount < 0 else 1
        abs_value = abs(account_amount)
        abs_value = round(abs_value, 2)

        text = str(abs_value) + "%"
        sign_text = "" if sign == 1 else "-"
        # return f"{account_name} is {sign_text} {text}"
        return (
            f"{splitted_account_name} represents {sign_text} {text} of Assets." + "\n"
        )

    def get_nb_comparative_pairs(self) -> int:
        """
        Get the number of comparative pairs in the dataset
        Using heuristics
        if seq_max_length ==512  => number of pairs = 20
        => [seq_max_length/512] * 20
        """
        return int(self.max_seq_length / 512) * 20

    def get_comparative_account_text(
        self, account_name1, account_name2, account_amount1, account_amount2
    ):
        """
        Get the comparative text of a pair of account.
        Always write the account with the largest amount first.
        ([Larger account] is [ratio] times [smaller account])
        :param account_name1: The name of the first account
        :param account_name2: The name of the second account
        :param account_amount1: The amount of the first account
        :param account_amount2: The amount of the second account
        """

        if account_amount1 > account_amount2:
            larger_account_name = account_name1
            smaller_account_name = account_name2
            larger_account_amount = account_amount1
            smaller_account_amount = account_amount2
        else:
            larger_account_name = account_name2
            smaller_account_name = account_name1
            larger_account_amount = account_amount2
            smaller_account_amount = account_amount1

        ratio = larger_account_amount / smaller_account_amount

        ratio = round(ratio, 2)

        comparative_text = f"{self.split_by_capital_letter(larger_account_name)} is {ratio} {self.split_by_capital_letter(smaller_account_name)}"
        return comparative_text

    def load_all_in_memory(self):
        """
        Multiprocessed loading of the dataset in memory.
        Load all the dataset in memory.
        """
        all_indices = list(range(self.__len__()))

        max_core = 40
        num_cores = min(max_core, multiprocessing.cpu_count() - 2)

        chunk_size = len(all_indices) // num_cores

        chunked_indices = [
            all_indices[i : i + chunk_size]
            for i in range(0, len(all_indices), chunk_size)
        ]

        def process_chunk(indices):
            inputs_desc = []
            inputs_attn_mask = []
            targets = []
            lengths = []
            for i in tqdm.tqdm(indices):
                data = self.__getitem__(i)
                inputs_desc.append(data["input"])
                inputs_attn_mask.append(data["input_attn_mask"])
                targets.append(data["target"])
                lengths.append(data["length"])

            return (
                torch.stack(inputs_desc),
                torch.stack(inputs_attn_mask),
                torch.stack(targets),
                torch.stack(lengths),
            )

        all_data = Parallel(n_jobs=num_cores)(
            delayed(process_chunk)(chunk)
            for chunk in tqdm.tqdm(
                chunked_indices, f"Loading {self.type} dataset in memory"
            )
        )

        self.all_data_dict["input"] = []
        self.all_data_dict["input_attn_mask"] = []
        self.all_data_dict["target"] = []
        self.all_data_dict["length"] = []

        for input, input_attn_mask, target, length in all_data:  # type: ignore
            self.all_data_dict["input"].append(input)
            self.all_data_dict["input_attn_mask"].append(input_attn_mask)
            self.all_data_dict["target"].append(target)
            self.all_data_dict["length"].append(length)

        self.all_data_dict["input"] = torch.cat(self.all_data_dict["input"], dim=0)

        self.all_data_dict["input_attn_mask"] = torch.cat(
            self.all_data_dict["input_attn_mask"], dim=0
        )

        self.all_data_dict["target"] = torch.cat(self.all_data_dict["target"], dim=0)
        self.all_data_dict["length"] = torch.cat(self.all_data_dict["length"], dim=0)

    def split_by_capital_letter(self, tag) -> str:
        """
        Separate a tag by capital letters(Adding a space)
        :param tag: The tag to separate
        :return: The separated tag
        """
        return re.sub(r"([A-Z])", r" \1", tag).strip()

    def __getitem__(
        self,
        idx,
        inference_mode=False,
        verbose=False,
    ) -> Dict[str, Any]:
        """
        Get a sample from the dataset as a dictionary of tensors.
        :param idx: The index of the sample
        :param inference_mode: If True, return more data for inference
        :param verbose: If True, print more information
        """

        if self.data_in_memory:
            input = self.all_data_dict["input"][idx]
            attn_mask = self.all_data_dict["input_attn_mask"][idx]
            target = self.all_data_dict["target"][idx]
            length = self.all_data_dict["length"][idx]

            sample = {
                "input": input,
                "input_attn_mask": attn_mask,
                "sample_idx": idx,
                "target": target,
                "length": length,
            }
            return sample

        else:
            filepath = self.data_files[idx]
            df = pd.read_csv(
                filepath,
                dtype={"account_num": self.accounts_index.index.dtype},
                usecols=["account_num", "cik", "tag", "net_change"],
            ).fillna("")

            df = pd.merge(df, self.tags_index[["tag", "tag_depth"]], on="tag")
            if self.max_tag_depth is not None:
                # If max_tag_depth is not None, then we only keep the first max_tag_depth tags
                # Tags with detph=-1 are not considered(they are not in the taxonomy)

                df = df.query("tag_depth>0")
                df = df.query(f"tag_depth <= {self.max_tag_depth}").copy()

            if self.min_tag_depth is not None:
                df = df.query("(tag_depth>=4) | (tag=='Assets')").copy()

            if (
                self.template_type
                == TextTransformerTemplateType.DESCRIPTIVE_WITH_MISSING
            ):
                # Left join with the tags_index to get the tag_depth
                df = pd.merge(
                    df,
                    self.tags_index.query("tag_depth>0 & tag_depth<4")[["tag"]],
                    on="tag",
                    how="right",
                )
                # Fill missing values with 0
                df.fillna(0, inplace=True)

            # 1. Get the tag number from taxonomy
            tag_number_dict = {}
            for tag in df["tag"].values:
                node = self.bs_taxonomy_tree.get_node_by_concept_name(
                    tag
                ) or self.is_taxonomy_tree.get_node_by_concept_name(tag)
                if node:
                    tag_number_dict[tag] = node.number
                else:
                    tag_number_dict[tag] = 1e5

            df["tag_number"] = df["tag"].map(tag_number_dict)
            # sort_by_account_num = True

            if self.bread_first_tree_exploration:
                df.loc[df["tag_depth"] == -1, "tag_depth"] = 1e5
                df = df.sort_values(
                    by=["tag_depth", "tag_number"], ascending=[True, True]
                ).reset_index(drop=True)
            else:
                df = df.sort_values(by="tag_number").reset_index(drop=True)

            if self.template_type == TextTransformerTemplateType.DESCRIPTIVE:
                full_text = self.get_descriptive_full_text(df)

            elif self.template_type == TextTransformerTemplateType.DESCRIPTIVE_NUMLOGIC:
                full_text = self.get_descriptive_full_text(df)

                # Remove all the "="
                full_text = full_text.replace(" = ", " ").replace("%", "")
                # apply numero_logic formatting.
                full_text = numerologic_preprocess_all_numbers(full_text)

            elif (
                self.template_type
                == TextTransformerTemplateType.DESCRIPTIVE_WITH_ASSETS
            ):
                full_text = self.get_descriptive_full_text_with_assets(df)

            elif (
                self.template_type == TextTransformerTemplateType.DESCRIPTIVE_ROUND_ZERO
            ):
                full_text = self.get_descriptive_full_text(df, round_to_int=True)

            elif self.template_type == TextTransformerTemplateType.DESCRIPTIVE_LOG:
                full_text = self.get_descriptive_log_full_text(df)

            elif self.template_type == TextTransformerTemplateType.DESCRIPTIVE_VERBOSE:
                full_text = self.get_descriptive_full_text_verbose(df)

            elif self.template_type == TextTransformerTemplateType.DESCRIPTIVE_SORTED:
                full_text = self.get_descriptive_full_text(df, sort_values=True)

            elif self.template_type == TextTransformerTemplateType.RAW:
                full_text = self.get_descriptive_raw_text(df)

            elif self.template_type == TextTransformerTemplateType.DESCRIPTIVE_WITH_RAW:
                full_text = self.get_descriptive_full_text_with_raw(df)

            elif self.template_type == TextTransformerTemplateType.COMPARATIVE:
                # 1. Get all the pair of accounts satisfying the depth gap constraint
                full_text = self.get_comparative_full_text(df, verbose=verbose)

            elif (
                self.template_type
                == TextTransformerTemplateType.DESCRIPTIVE_WITH_RATIOS
            ):
                full_text = self.get_descriptive_full_text_with_ratios(df)

            elif self.template_type == TextTransformerTemplateType.MIXED:
                # Mixed
                full_text = self.get_mixed_full_text(df)
            elif self.template_type == TextTransformerTemplateType.NO_CHANGE:
                full_text = self.get_full_text_no_change(df)

            else:
                raise ValueError(f"Template type {self.template_type} not supported")

            # Target
            cik = df["cik"].values[0]
            sic = self.registrants_index_dict[cik]

            if self.return_raw:
                return {
                    "cik": cik,
                    "full_text": full_text,
                    "sample_idx": idx,
                    "label": self.sic_to_title[sic],
                }

            target = torch.tensor(self.sic_id_index[sic])
            if verbose:
                print("Full balance text :\n ", full_text)

                tokenizer_out_full = self.tokenizer(
                    full_text, return_tensors="pt", add_special_tokens=True
                )
                print(
                    "\n Initial number of tokens:",
                    tokenizer_out_full["input_ids"].shape,
                )

            # Padding the sequenceprompt_formatter_fn
            if self.prompt_formatter:
                full_text = self.prompt_formatter(full_text)

            tokenizer_out = self.tokenizer(
                full_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
                add_special_tokens=True,
            )
            if verbose:
                print(
                    "\n Total tokens (After padding or truncation) :",
                    tokenizer_out["input_ids"][tokenizer_out["input_ids"] != 0].shape,
                )

            input_ids = tokenizer_out["input_ids"].squeeze(0)
            attn_mask = tokenizer_out["attention_mask"].squeeze(0)

            data = {
                "input": input_ids,
                "input_attn_mask": attn_mask,
                "sample_idx": idx,
                "n_tags": df.tag.nunique(),
                "target": target,
                "length": torch.tensor(len(self.tokenizer(full_text)["input_ids"])),
            }

            if verbose:
                data["full_text"] = full_text
                data["df"] = df

            if self.weighted_loss:
                data["class_weights"] = self.label_weights[idx]
            return data


class SecTextTransformerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int = 0,
        sic_digits: int = 3,
        balance_sampling: bool = False,
        weighted_loss: bool = False,
        max_tag_depth=None,
        min_tag_depth=None,
        seq_max_length: int = 512,
        load_in_memory: bool = True,
        template_type=TextTransformerTemplateType.DESCRIPTIVE,
        prompt_formatter: Callable[[str], str] = None,
        bread_first_tree_exploration: bool = False,
    ):
        """
        :param dataset: The folder of the dat
        :param embedding_experiment : The experiment name of the pretrained word embedding model.
        :param batch_size: represents the batch size for the data.
        :param resize: represents the resize value for the image. e.g. (256, 256)
        :param augment: represents the data augmentation techniques.
        :param max_tag_depth: The maximum depth of the tags to consider. If None, all tags are considered.
        """

        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.sic_digits = sic_digits
        self.tokenizer = tokenizer
        self.seq_max_length = seq_max_length
        self.max_tag_depth = max_tag_depth
        self.min_tag_depth = min_tag_depth
        self.balance_sampling = balance_sampling
        self.weighted_loss = weighted_loss
        self.bread_first_tree_exploration = bread_first_tree_exploration
        self.train_dataset = SecTextTransformerDataset(
            dataset_dir=self.dataset_dir,
            type=DatasetType.TRAIN,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            seq_max_length=self.seq_max_length,
            balance_sampling=balance_sampling,
            weighted_loss=weighted_loss,
            max_tag_depth=self.max_tag_depth,
            min_tag_depth=self.min_tag_depth,
            template_type=template_type,
            load_in_memory=load_in_memory,
            prompt_formatter=prompt_formatter,
            bread_first_tree_exploration=bread_first_tree_exploration,
        )

        self.val_dataset = SecTextTransformerDataset(
            dataset_dir=self.dataset_dir,
            type=DatasetType.VAL,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            seq_max_length=self.seq_max_length,
            max_tag_depth=self.max_tag_depth,
            min_tag_depth=self.min_tag_depth,
            load_in_memory=load_in_memory,
            template_type=template_type,
            prompt_formatter=prompt_formatter,
            bread_first_tree_exploration=bread_first_tree_exploration,
        )

        self.test_dataset = SecTextTransformerDataset(
            dataset_dir=self.dataset_dir,
            type=DatasetType.TEST,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            seq_max_length=self.seq_max_length,
            max_tag_depth=self.max_tag_depth,
            min_tag_depth=self.min_tag_depth,
            load_in_memory=load_in_memory,
            template_type=template_type,
            prompt_formatter=prompt_formatter,
            bread_first_tree_exploration=bread_first_tree_exploration,
        )

    def train_dataloader(self, shuffle=True) -> DataLoader:
        if self.train_dataset.label_weights is not None and self.balance_sampling:
            """
            Use a balanced sampler
            """
            weights = self.train_dataset.label_weights
            sampler = torch.utils.data.sampler.WeightedRandomSampler(
                weights, len(weights), replacement=True
            )
        else:
            sampler = None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            num_workers=self.num_workers,
        )


class SecTransformerDatasetRatio(SecDataset):
    def __init__(
        self,
        dataset_dir: str,
        sic_digits=1,
        max_tag_depth=10,
    ):
        """
        Dataset for sic classification using a transformer model combining
        text and amounts.
        :param dataset_dir: The directory of the dataset
        :param type: The type of the dataset (train, val or test)
        :param tokenizer: The tokenizer to use
        :param sic_digits: The number of digits of the sic code (Not really used for now)
        :param seq_max_length: The maximum length of the sequence
        :param max_tag_depth: The maximum depth of the tags to consider. If None, all tags are considered.
        :param balance_sampling: If True, use a balanced sampler
        :param is_encoder_transformer: If True, use an encoder transformer, otherwise use a decoder transformer
        :param template_type: The type of template to use (descriptive or comparative or mixed)
        :param max_comparative_pair_depth_gap: The maximum gap between the depth of the tags in a comparative pair.

        """
        super().__init__(
            dataset_dir, DatasetType.ALL, sic_digits=sic_digits, use_aggregate=True
        )
        self.sic_digits = sic_digits
        self.sic_col = f"sic{sic_digits}"
        self.registrants_index_dict = self.registrants_index.astype(
            {self.sic_col: int}
        )[self.sic_col].to_dict()
        self.account_descriptions_index = self.accounts_index["description"].to_dict()
        self.all_data_dict = {}
        self.data_in_memory = False
        self.class_loss_weights = None
        self.label_weights = None
        self.max_tag_depth = max_tag_depth
        ##Load tags_index
        tag_index_file = os.path.join(dataset_dir, "index", "tags_index.csv")
        self.tags_index = pd.read_csv(tag_index_file, index_col=None)[
            ["tag", "tlabel", "tag_depth"]
        ].drop_duplicates(
            subset="tlabel"
        )  # type: ignore
        # Max Depth of the tags
        if self.max_tag_depth is None:
            pass
        else:
            # If max_tag_depth is not None, then we only keep the first max_tag_depth tags
            relevant_tags = self.tags_index.query(
                f"tag_depth <= {self.max_tag_depth} & tag_depth != -1"
            )["tag"].values.tolist()
            print(
                f"Max depth : {self.max_tag_depth}, Number of tags : {len(relevant_tags)}"
            )

        # Taxonomy tree
        self.taxonomy_tree = CalculationTree.build_taxonomy_tree(
            SEC_TAX_DATA_DIR, SEC_TAX, SEC_TAX_VERSION
        )

        # 2. Sampling probas( invertily proportional to the number of tag of the same depth)
        self.tag_depth_count = self.taxonomy_tree.depth_count_dict()
        self.tag_depth_dict = self.tags_index.set_index("tag")["tag_depth"].to_dict()

        print("Dataset Size : ", len(self.data_files))

    def __getitem__(self, idx):
        """
        Get a sample form the dataset
        """
        filepath = self.data_files[idx]
        df = pd.read_csv(filepath, usecols=["cik", "tag", "net_change"]).fillna("")
        if self.max_tag_depth is not None:
            # If max_tag_depth is not None, then we only keep the first max_tag_depth tags
            # Tags with detph=-1 are not considered(they are not in the taxonomy)
            df = pd.merge(df, self.tags_index[["tag", "tag_depth"]], on="tag")
            df = df.query("tag_depth>0")
            df = df.query(f"tag_depth <= {self.max_tag_depth}")

        # Get the sic code
        cik = df["cik"].values[0]
        sic = self.registrants_index_dict[cik]

        df["sic"] = sic

        tags_value_dict = df.set_index("tag")["net_change"].to_dict()

        all_tags = list(tags_value_dict.keys())

        new_features = []

        for i in range(len(all_tags)):
            for j in range(i, len(all_tags)):
                tag_i = all_tags[i]
                value_i = tags_value_dict[tag_i]

                tag_j = all_tags[j]
                value_j = tags_value_dict[tag_j]

                feature_name = (
                    f"{tag_i}_{tag_j}" if value_i > value_j else f"{tag_j}_{tag_i}"
                )
                feature_value = (
                    value_i / value_j if value_i > value_j else value_j / value_i
                )

                new_features.append(
                    {
                        "tag": feature_name,
                        "net_change": feature_value,
                    }
                )

        # Single row dataframe(columns are tags, values are ratios(net_change), sic is the target)
        df_new = pd.DataFrame(new_features).set_index("tag").T
        df_new["sic"] = sic
        df_new["idx"] = idx
        df_new = df_new.drop(columns=["tag"]).set_index("idx")
        return df_new
