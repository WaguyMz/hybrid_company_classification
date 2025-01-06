import glob
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import pandas as pd
import torch.utils.data
import tqdm

from researchpkg.industry_classification.config import SIC1_EXCLUDED
from researchpkg.industry_classification.dataset.utils import DatasetType
from researchpkg.industry_classification.utils.sics_loader import load_sic_codes


class NormalizationType:
    MINMAX = "minmax"
    STD = "std"
    ROBUST = "robust"
    LOG = "log"
    LOG_INT = "log_int"
    LOCAL = "local"
    LOCAL_WITH_RAW = "local_with_raw"
    INDICATOR = "indicator"
    NONE = "none"
    PARENT = "parent"
    COMMON_PERCENTAGES = "common_percentages"
    COMMON_PERCENTAGES_WITH_BASE_FEATURES = "common_percentages_with_base_features"


class SecDataset(torch.utils.data.Dataset):
    """
    Base class for managing sec US data
    """

    # Class attributes
    NORMALIZATION_TYPES = [
        NormalizationType.MINMAX,
        NormalizationType.STD,
        NormalizationType.ROBUST,
        NormalizationType.LOG,
        NormalizationType.LOG_INT,
        NormalizationType.LOCAL,
        NormalizationType.LOCAL_WITH_RAW,
        NormalizationType.NONE,
        NormalizationType.PARENT,
        NormalizationType.INDICATOR,
        NormalizationType.COMMON_PERCENTAGES,
        NormalizationType.COMMON_PERCENTAGES_WITH_BASE_FEATURES,
    ]

    def __init__(
        self,
        data_dir: str,
        type: DatasetType,
        sic_digits=2,
        use_aggregate=True,
        normalization_type=NormalizationType.LOCAL,
    ):
        """
        :param data_dir The root data dir (which the follo  ng structure)
            - index
            - balanced_sheets:
                * train
                * test
            - balanced_sheets_agg:
            * train
                * test
        :param type :
            DatasetType: Train, Val or Test

        :param use_aggregate:
            If True load data from data_dir.balanced_sheets
            else load from balanced_sheets_agg.

        """
        self.data_files = []
        self.normalization_type = normalization_type
        self.use_aggregate = use_aggregate
        self.data_dir = data_dir
        self.sic_digits = sic_digits
        self.type = type

        if self.type == DatasetType.ALL:
            self.balanced_sheets_dir = os.path.join(
                self.data_dir,
                "balance_sheets" if not self.use_aggregate else "balance_sheets_agg",
            )
        else:
            self.balanced_sheets_dir = os.path.join(
                self.data_dir,
                "balance_sheets" if not self.use_aggregate else "balance_sheets_agg",
                self.type,
            )

        (
            self.accounts_index,
            self.registrants_index,
            self.sic_id_index,
        ) = self.load_index(self.data_dir, sic_digits=self.sic_digits)

        sic_code_df = load_sic_codes()[["sic", "industry_title"]]
        self.sic_to_title = sic_code_df.set_index("sic")["industry_title"].to_dict()

        self.accounts_index.fillna(torch.inf, inplace=True)

        # 1. Account_index
        accounts_index_file = os.path.join(data_dir, "index", "accounts_index.csv")
        self.accounts_index_global = pd.read_csv(
            accounts_index_file,
            index_col=None,
            dtype={"account_num": str},
        ).query("account_num=='-1'")

        self.load_data_index()

    @staticmethod
    def load_index(data_dir, sic_digits):
        """
        Load account_index and registrants index.
        """

        # 1. Account_index
        accounts_index_file = os.path.join(data_dir, "index", "accounts_index.csv")
        accounts_index = (
            pd.read_csv(accounts_index_file).set_index("account_num").sort_index()
        )
        accounts_index["idx"] = np.arange(len(accounts_index))
        accounts_index = accounts_index.query("account_num!='-1' & account_num!=-1 ")

        # 2. registrants index
        registrants_index_file = os.path.join(
            data_dir, "index", "registrants_index.csv"
        )
        registrants_index = pd.read_csv(
            registrants_index_file,
            index_col="cik",
        )
        registrants_index = registrants_index[
            ~registrants_index["sic1"].isin(SIC1_EXCLUDED)
        ]
        # Note. Sic is actually typed as int. Can not work if consedering sic1=0
        all_sics = list(sorted(registrants_index[f"sic{sic_digits}"].unique().tolist()))
        sic_id_index = {sic: i for i, sic in enumerate(all_sics)}
        return accounts_index, registrants_index, sic_id_index

    @staticmethod
    def load_sic_counts(data_dir, sic_digits):
        file = os.path.join(data_dir, "index", f"sic{sic_digits}_count.csv")
        df = pd.read_csv(file, index_col=None)
        return df.set_index(f"sic{sic_digits}")["count"].to_dict()

    def load_data_index(self):
        self.data_files = list(
            glob.glob(f"{self.balanced_sheets_dir}/**/**.csv", recursive=True)
        )

    def log_scaling_transform(self, x):
        """
        Transform a given sample to sign(x)*log_10(|x|)
        """
        x = torch.sign(x) * torch.log10(torch.abs(x) + 1)
        # x = torch.log10(torch.abs(x) + 1)
        return x

    def revert_log_scaling_transform(self, x):

        return torch.sign(x) * (10 ** (abs(x)) - 1)

    def log_unscaling_transform(self, x):
        """
        Transform a given sample to sign(x)*log_10(|x|)
        """
        x = torch.sign(x) * (10 ** torch.abs(x) - 1)
        return x

    def log_scaling_transform_integer(self, x):
        """
        Transform a given sample to [sign(x)*log_10(|x|)]
        """
        # Check if it is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        x = torch.sign(x) * torch.log10(torch.abs(x) + 1)
        x = x.round()
        return x

    def indicator_representation(self, x):
        """
        Transform a given sample to [0,1] representation
        return 0 if |x| >0 else 1
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        return (x != 0).float()

    def minmax_transform(self, x, accounts):
        """
        Transform a given sample x (net_change)
        Apply a min max_scaling.
        """
        x_max = torch.tensor(self.accounts_index.loc[accounts]["max"].values).float()
        x_min = torch.tensor(self.accounts_index.loc[accounts]["min"].values).float()
        x = (x - x_min) / (x_max - x_min)
        return x

    def std_scaling_transform(self, x, accounts):
        """
        Transform a given sample x (net_change)
        Apply standard scaling (z-score normalization).
        """
        x_mean = torch.tensor(self.accounts_index.loc[accounts]["mean"].values).float()
        x_std = torch.tensor(self.accounts_index.loc[accounts]["std"].values).float()

        x = (x - x_mean) / x_std
        # ensuring no nan values forwared to the model.
        x[x != x] = 0
        x[x.abs() == np.inf] = 0

        return x

    def local_scaling_transform(self, x, ref, in_percent=False):
        """
        apply x = x/x["Assets]
        """
        return (x / ref) * (100 if in_percent else 1)

    def std_unscaling_transform(self, x, accounts):
        """
        Transform a given sample x (net_change)
        Apply standard scaling (z-score normalization).
        """
        x_mean = torch.tensor(self.accounts_index.loc[accounts]["mean"].values).float()
        x_std = torch.tensor(self.accounts_index.loc[accounts]["std"].values).float()
        return x * x_std + x_mean

    def robust_scaling_transform(self, x, accounts):
        """
        Transform a given sample x (net_change)
        Apply standard scaling (z-score normalization).
        """
        x_median = torch.tensor(
            self.accounts_index.loc[accounts]["median"].values
        ).float()
        x_q75 = torch.tensor(
            self.accounts_index.loc[accounts]["quantile_75"].values
        ).float()
        x_q25 = torch.tensor(
            self.accounts_index.loc[accounts]["quantile_25"].values
        ).float()

        x = (x - x_median) / (x_q75 - x_q25)
        x[[x_q25 == x_q75]] = 0
        return x

    def compute_label_weights(self):
        """
        Iterate over the dataset and compute the label weights.
        """
        target_list = []
        registrants_index_dict = self.registrants_index_dict
        sic_id_index = self.sic_id_index

        def get_file_label(filepath):
            df = pd.read_csv(filepath, nrows=1, usecols=["cik"])
            cik = df["cik"][0]
            sic = registrants_index_dict[cik]
            target = sic_id_index[sic]
            return target

        # num_cores = min(MAX_CORE_USAGE, multiprocessing.cpu_count() - 2)

        target_list = [
            get_file_label(self.data_files[i])
            for i in tqdm.tqdm(range(self.__len__()), desc="Computing label weights")
        ]

        self.label_weights = SecDataset.calculate_class_weights(
            target_list, return_dict=False
        )

    @staticmethod
    def calculate_class_weights(
        class_idxs: List[int], beta: float = 1, return_dict: bool = True
    ) -> Dict[int, float]:
        """
        :param class_idxs: List of class indexes
        :param beta: Beta parameter for the F-beta score
        :param return_dict: If True return a dictionary else
        return a list of weights of all classes.
        Return a dictionary of class weights for each class in the dataset.
        """
        class_counts = Counter(class_idxs)
        n_classes = len(np.unique(class_idxs))
        if return_dict:
            return {
                k: len(class_idxs) / (beta * (v) * n_classes)
                for k, v in class_counts.items()
            }
        else:
            n_samples = len(class_idxs)
            return (n_samples / n_classes) * np.array(
                [1 / (beta * class_counts[x]) for x in class_idxs]
            )  # type: ignore

    def __len__(self):
        return len(self.data_files)

    # Method used by Natural Language Representation
    def split_by_capital_letter(self, tag) -> str:
        """
        Separate a tag by capital letters(Adding a space)
        :param tag: The tag to separate
        :return: The separated tag
        """
        return re.sub(r"([A-Z])", r" \1", tag).strip()

    def get_descriptive_full_text(self, df, sort_values=False):
        """
        Get the descriptive text for the dataframe
        A list of sentences of the form :
        """

        if sort_values:
            df = df.copy().sort_values("net_change", ascending=True)

        # 1. Normalise the net change by the assets
        assets_amount = df.query("tag == 'Assets'")["net_change"].values[0]
        df = df.query(
            " tag != 'LiabilitiesAndStockholdersEquity' & tag != 'Assets'"
        ).copy()
        # df = df.query('tag_depth != 1').copy()

        df["original_net_change"] = df["net_change"].copy().abs()

        df["net_change"] = df.apply(
            lambda x: self.local_scaling_transform(
                x["net_change"], assets_amount, in_percent=True
            ),
            axis=1,
        )

        df["account_text"] = df.apply(
            lambda x: self.get_descriptive_account_text_in_percentage(
                self.split_by_capital_letter(x["tag"]), x["net_change"]
            ),
            axis=1,
        )
        # Kept original tag _values for tag_depth = 1
        # # And scaling other
        # df.loc[df["tag_depth"] == 1, "net_change"] = df["original_net_change"]
        # df.loc[df["tag_depth"] == 1, "account_text"] = self.get_descriptive_raw_text(df.query("tag_depth == 1"))

        full_text = "\n ".join(df["account_text"].values.tolist())
        return full_text

    def get_descriptive_relative_full_text(self, df, sort_values=False):
        """
        Get the descriptive text of a dataframe.
        Each tag is expressed as a percentage of the parent tag.
        """

        def relative_net_change_to_text(splitted_account_name, account_amount):
            """
            No sign
            """
            abs_value = abs(account_amount * 100)
            abs_value = round(abs_value, 2)

            text = str(abs_value) + "%"
            # return f"{account_name} is {sign_text} {text}"
            return f"{splitted_account_name} = {text}" + "."

        def get_parent_tag(tag):
            """
            Get the parent tag of a tag
            """
            node = self.bs_taxonomy_tree.get_node_by_concept_name(
                tag
            ) or self.is_taxonomy_tree.get_node_by_concept_name(tag)

            if node is None or node.parent_concept_id is None:
                return ""

            parent_node = self.bs_taxonomy_tree.get_node_by_concept_id(
                node.parent_concept_id
            ) or self.is_taxonomy_tree.get_node_by_concept_id(node.parent_concept_id)
            return parent_node.concept_name

        df["parent_tag"] = df["tag"].apply(get_parent_tag)

        # fill a column with the parent tag's net change
        df["parent_net_change"] = df["parent_tag"].apply(
            lambda x: df.query(f"tag == '{x}'")["net_change"].values[0]
            if x != ""
            else 0
        )
        df = df.query("parent_tag != ''").copy()
        df["relative_net_change"] = df["net_change"] / df["parent_net_change"]

        df["account_text"] = df.apply(
            lambda x: relative_net_change_to_text(
                self.split_by_capital_letter(x["tag"]), x["relative_net_change"]
            ),
            axis=1,
        )

        df["account_text"] = full_text = "\n ".join(df["account_text"].values.tolist())
        return full_text

    def get_descriptive_full_text_verbose(self, df, sort_values=False):
        """
        Get the descriptive text for the dataframe
        A list of sentences of the form :
        """

        if sort_values:
            df = df.copy().sort_values("net_change", ascending=True)

        # 1. Normalise the net change by the assets
        assets_amount = df.query("tag == 'Assets'")["net_change"].values[0]

        # Removing assets and LiabilitiesAndStockHoldersEquity
        df = df.query(
            "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
        ).copy()

        df["net_change"] = df.apply(
            lambda x: self.local_scaling_transform(
                x["net_change"], assets_amount, in_percent=True
            ),
            axis=1,
        )
        df["account_text"] = df.apply(
            lambda x: self.get_descriptive_account_text_in_percentage_verbose(
                self.split_by_capital_letter(x["tag"]), x["net_change"]
            ),
            axis=1,
        )
        full_text = " ".join(df["account_text"].values.tolist())
        return full_text

    def get_descriptive_account_text_in_percentage_with_original_value(
        self, splitted_account_name, account_amount_normalized, account_amount_raw
    ):
        """
        Get the comparator text for the account.
        Example :
        """

        sign = -1 if account_amount_normalized < 0 else 1
        sign_text = "" if sign == 1 else "-"
        abs_value = abs(round(account_amount_normalized, 2))

        # value_text  = f"{abs_value}%" if not "Ratio" in splitted_account_name else f"{abs_value}"

        return (
            f"{splitted_account_name} = ${sign_text}{account_amount_raw}:{abs_value}%."
        )
        return f"{splitted_account_name}:{sign_text}{value_texte}, \n"

    def get_raw_text_in_percentage(
        self, splitted_account_name, account_name, account_amount
    ):
        """
        Get the comparator text for the account.
        Example :
        """

        sign = -1 if account_amount < 0 else 1
        sign_text = "" if sign == 1 else "-"
        abs_value = abs(round(account_amount * 100, 2))

        value_text = (
            f"{abs_value}%" if not account_name in RATIOS_12 else f"{abs_value}"
        )

        # return f"{account_name} is ${sign_text}{account_amount_raw} and represents {abs_value}%\n"
        return f"{splitted_account_name} = {sign_text}{value_text}, ."

    def get_descriptive_account_original_value(
        self, splitted_account_name, account_amount_raw
    ):
        """
        Get the comparator text for the account.
        Example :
        """

        sign = -1 if account_amount_raw < 0 else 1
        account_amount_raw = abs(account_amount_raw)
        sign_text = "" if sign == 1 else "-"

        account_amount_raw = "{:,.0f}".format(account_amount_raw)

        # value_text  = f"{abs_value}%" if not "Ratio" in splitted_account_name else f"{abs_value}"

        return f"{splitted_account_name} = ${sign_text}{account_amount_raw}."

    def get_descriptive_full_text_with_raw(self, df):
        """
        Get the descriptive text for the dataframe
        A list of sentences of the form :
        """

        def split_by_capital_letter(tag) -> str:
            """
            Separate a tag by capital letters(Adding a space)
            :param tag: The tag to separate
            :return: The separated tag
            """
            return re.sub(r"([A-Z])", r" \1", tag).strip()

        # 1. Normalise the net change by the assets
        assets_amount = df.query("tag == 'Assets'")["net_change"].values[0]

        # Removing assets and LiabilitiesAndStockHoldersEquity
        df = df.query(
            "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
        ).copy()

        df["net_change_normalized"] = df.apply(
            lambda x: self.local_scaling_transform(
                x["net_change"], assets_amount, in_percent=True
            ),
            axis=1,
        )
        df["account_text"] = df.apply(
            lambda x: self.get_descriptive_account_text_in_percentage_with_original_value(
                split_by_capital_letter(x["tag"]),
                x["net_change_normalized"],
                x["net_change"],
            ),
            axis=1,
        )
        # Permute the order of the accounts
        # df = df.sample(frac=1)
        # Join all text
        full_text = "\n ".join(df["account_text"].values.tolist())
        return full_text

    def get_descriptive_raw_text(self, df):
        """
        Get the descriptive text for the dataframe
        A list of sentences of the form :
        """

        def split_by_capital_letter(tag) -> str:
            """
            Separate a tag by capital letters(Adding a space)
            :param tag: The tag to separate
            :return: The separated tag
            """
            return re.sub(r"([A-Z])", r" \1", tag).strip()

        # Removing assets and LiabilitiesAndStockHoldersEquity
        # df = df.query(
        #     "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
        # ).copy()

        df["account_text"] = df.apply(
            lambda x: self.get_descriptive_account_original_value(
                split_by_capital_letter(x["tag"]), x["net_change"]
            ),
            axis=1,
        )
        # Permute the order of the accounts
        # df = df.sample(frac=1)
        # Join all text
        full_text = "\n ".join(df["account_text"].values.tolist())
        return full_text

    def get_descriptive_full_text_with_ratios(self, df):
        """
        Get the descriptive text for the dataframe
        A list of sentences of the form :
        """

        def split_by_capital_letter(tag) -> str:
            """
            Separate a tag by capital letters(Adding a space)
            :param tag: The tag to separate
            :return: The separated tag
            """
            return re.sub(r"([A-Z])", r" \1", tag).strip()

        # Removing assets and LiabilitiesAndStockHoldersEquity
        df = df.query(
            "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
        ).copy()

        df["account_text"] = df.apply(
            lambda x: self.get_raw_text_in_percentage(
                split_by_capital_letter(x["tag"]), x["tag"], x["net_change"]
            ),
            axis=1,
        )
        # Permute the order of the accounts
        # df = df.sample(frac=1)
        # Join all text
        full_text = "\n ".join(df["account_text"].values.tolist())
        return full_text

    def get_income_ratio_full_text(self, df):
        """
        Get the descriptive text for the dataframe
        A list of sentences of the form :
        """

        # 1. Normalise the net change by the assets
        assets_amount = df.query("tag == 'Assets'")["net_change"].values[0]

        # Removing assets and LiabilitiesAndStockHoldersEquity
        df = df.query(
            "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
        ).copy()

        df["net_change_normalized"] = df.apply(
            lambda x: self.local_scaling_transform(
                x["net_change"], assets_amount, in_percent=True
            ),
            axis=1,
        )
        df["account_text"] = df.apply(
            lambda x: self.get_descriptive_account_text_in_percentage_with_original_value(
                self.split_by_capital_letter(x["tag"]),
                x["net_change_normalized"],
                x["net_change"],
            ),
            axis=1,
        )
        # Permute the order of the accounts
        # df = df.sample(frac=1)
        # Join all text
        full_text = "\n ".join(df["account_text"].values.tolist())
        return full_text

    def get_comparative_full_text(self, df: pd.DataFrame, verbose=False):
        """
        Get the comparative text for the dataframe
        A list of sentences of the form :
        ([Larger account] is [ratio] [smaller account]. )
        """

        def sample_comparative_pairs(df: pd.DataFrame):
            """
            Sample comparative pairs from the dataframe
            1. The number of pairs is given by self.get_nb_comparative_pairs()(Using heuristics)
            2. The sampling probas are inversely proportional to the number of tags of the same depth
            3. The tags are sampled from the tags with depth <= self.max_tag_depth
            4. The tags are sampled from tags from different branches
            5. The tags are sampled from tags with depth gap <= self.max_comparative_pair_depth_gap
            """

            def get_depth(tag):
                return self.tag_depth_dict.get(tag, -1)

            # 1. The number of comparative pairs to sample
            nb_pairs = self.get_nb_comparative_pairs()

            df = df.query(
                "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
            ).copy()

            # 3. All tags
            all_tags = df["tag"].values.tolist()

            # 4. Sampling probas
            all_tags = [
                t
                for t in all_tags
                if t != "Assets"
                and t != "LiabilitiesAndStockholdersEquity"
                and get_depth(t) in self.tag_depth_count.keys()
            ]
            probs = [self.tag_depth_count[get_depth(t)] for t in all_tags]
            probs = [1 / p for p in probs]
            # Convert to probas
            probs = [p / sum(probs) for p in probs]

            # 5. Sampling the tags
            sampled_tags = np.random.choice(
                all_tags, size=nb_pairs, replace=nb_pairs > len(all_tags), p=probs
            )
            # 6. Get the pairs
            accounts_values_pairs = []

            bs_tag_list = set(self.bs_taxonomy_tree.get_all_tags())
            is_tag_list = set(self.is_taxonomy_tree.get_all_tags())

            for tag in sampled_tags:
                # Get the accounts with the tag
                # 1. Candidate tags shoudl be in the same tree.
                # 2. # Candidate are tags from different branches and with
                # a depth gap <= max_comparative_pair_depth_gap

                is_bg_tag = tag in is_tag_list

                ref_tree = self.is_taxonomy_tree if is_bg_tag else self.bs_taxonomy_tree
                ref_list = is_tag_list if is_bg_tag else bs_tag_list

                candidates = [
                    t
                    for t in all_tags
                    if t != tag
                    and t in ref_list
                    and not ref_tree.are_in_same_branch(t, tag)
                    and abs(get_depth(t) - get_depth(tag))
                    <= self.max_comparative_pair_depth_gap
                ]

                if len(candidates) == 0:
                    continue

                second_tag = np.random.choice(candidates, size=1)[0]

                # Add the pair
                accounts_values_pairs.append(
                    (
                        (tag, df.query(f"tag == '{tag}'")["net_change"].values[0]),
                        (
                            second_tag,
                            df.query(f"tag == '{second_tag}'")["net_change"].values[0],
                        ),
                    )
                )

            return accounts_values_pairs

        # Removing assets and LiabilitiesAndStockHoldersEquity
        df = df.query(
            "tag != 'Assets' & tag != 'LiabilitiesAndStockholdersEquity'"
        ).copy()

        accounts_values_pairs = sample_comparative_pairs(df)

        if verbose:
            print(f"{len(accounts_values_pairs)} Comparative pairs : ")
            for i, p in enumerate(accounts_values_pairs):
                print(f"{i} : ({p[0][0]}, {p[1][0]})")
            print("\n---------------\n")

            # 2. Get the comparative text for each pair
        comparative_texts = [
            self.get_comparative_account_text(
                account_name1,
                account_name2,
                account_amount1,
                account_amount2,
            )
            for (
                (account_name1, account_amount1),
                (account_name2, account_amount2),
            ) in accounts_values_pairs
        ]

        full_text = "\n".join(comparative_texts)
        return full_text

    def get_mixed_full_text(self, df):
        """
        Get the a mix of descriptive and comparative text for the dataframe
        """
        full_text = (
            self.get_descriptive_full_text(df)
            + ".\n "
            + self.get_comparative_full_text(df)
        )
        return full_text
