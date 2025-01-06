"""
Sec Dataset for GBDT MEthods
"""

import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from joblib import Parallel, delayed
from overrides import override

from researchpkg.industry_classification.config import (
    MAX_CORE_USAGE,
    SEC_TAX,
    SEC_TAX_DATA_DIR,
    SEC_TAX_VERSION,
)
from researchpkg.industry_classification.dataset.sec_datamodule import (
    NormalizationType,
    SecDataset,
)
from researchpkg.industry_classification.dataset.sec_transformer_datamodule import (
    SecTextTransformerDataset,
)
from researchpkg.industry_classification.dataset.utils import DatasetType
from researchpkg.industry_classification.models.transformers.text_transformer import (
    TextTransformerForClassification,
)
from researchpkg.industry_classification.preprocessing.gaap_taxonomy_parser import (
    CalculationTree,
)
from researchpkg.industry_classification.preprocessing.sec_preprocessing_utils import (
    compute_common_percentages_15,
)

# from researchpkg.industry_classification.constants import NUM_QUANTILES


class SecGBDTDataset(SecDataset):
    """
    Derived SecDataset for GDBT Algorithms trainings
    """

    def __init__(
        self,
        dataset_dir: str,
        type: DatasetType,
        sic_digits=1,
        max_tag_depth=10,
        normalization_type=NormalizationType.LOCAL,
    ):
        super().__init__(
            dataset_dir,
            type,
            sic_digits=sic_digits,
            use_aggregate=True,
            normalization_type=normalization_type,
        )

        self.sic_digits = sic_digits
        self.sic_col = f"sic{sic_digits}"
        self.taxonomy_tree = CalculationTree.build_taxonomy_tree(
            SEC_TAX_DATA_DIR, SEC_TAX, SEC_TAX_VERSION
        )
        self.max_tag_depth = max_tag_depth

        # Load tags_index
        tag_index_file = os.path.join(dataset_dir, "index", "tags_index.csv")
        self.tags_index = pd.read_csv(tag_index_file, index_col=None)[
            ["tag", "tag_depth"]
        ].drop_duplicates(
            subset=["tag"]
        )  # type: ignore

        if self.max_tag_depth is not None:
            # Tags with detph=-1 are not considered(they are not in the taxonomy)
            self.tags_index = self.tags_index.query(
                "(tag_depth>0) & (tag_depth<={})".format(self.max_tag_depth)
            )
        
        #Filter min tag depth
        # self.tags_index = self.tags_index.query("tag_depth>=2")

        self.account_to_tag_index = (
            self.accounts_index.reset_index().set_index("account_num")["tag"].to_dict()
        )
        self.tag_to_account_index = (
            self.accounts_index.reset_index().set_index("tag")["account_num"].to_dict()
        )

        self.registrants_index_dict = self.registrants_index.astype(
            {self.sic_col: int}
        )[self.sic_col].to_dict()
        self.filter_data()

        self.all_data_dict = {}
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

    @override
    def std_scaling_transform(self, x, accounts):
        """
        Transform a given sample x (net_change)
        Apply standard scaling (z-score normalization).
        """
        x_mean = self.accounts_index.loc[accounts]["mean"].values
        x_std = self.accounts_index.loc[accounts]["std"]
        x = (x - x_mean) / x_std
        x[x_std == 0] = 0
        x[x != x] = 0
        return x

    def normalize_by_parent_tag(self, x, accounts):
        """
        Normalize a given sample x (net_change) by the parent tag.
        """
        values_dict = {account: x[i] for i, account in enumerate(accounts)}
        values = []

        for account, value in zip(accounts, x):
            if value == 0:
                values.append(0)
                continue
            tag = self.account_to_tag_index[account]
            parent_id = self.taxonomy_tree.get_node_by_concept_name(
                tag
            ).parent_concept_id
            parent = self.taxonomy_tree.get_node_by_concept_id(parent_id)

            if parent is None:
                values.append(value)

            else:
                parent_name = parent.concept_name
                parent_account_num = self.tag_to_account_index[parent_name]
                parent_value = values_dict[parent_account_num]

                if parent_value == 0:
                    ratio = 1
                else:
                    ratio = value / parent_value

                values.append(ratio)

        return np.array(values)

    def randon_mask(self, sample: np.ndarray, mask_prob=0.1):
        """
        Transform input data by adding random noise
        """
        mask = np.random.uniform(0, 1, sample.shape) < mask_prob
        sample[mask] = 0
        return sample

    def load_all_in_memory(self):
        """
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
            inputs_net_change = []
            targets = []
            ciks = []
            for i in tqdm.tqdm(indices):
                data = self.__getitem__(i)
                inputs_net_change.append(data["input_net_change"])
                targets.append(data["target"])
                ciks.append(data["cik"])

            return np.stack(inputs_net_change), np.stack(targets), np.stack(ciks)

        all_data = Parallel(n_jobs=num_cores)(
            delayed(process_chunk)(chunk)
            for chunk in tqdm.tqdm(
                chunked_indices, f"Loading {self.type} dataset in memory"
            )
        )

        self.all_data_dict["input_net_change"] = []
        self.all_data_dict["target"] = []
        self.all_data_dict["cik"] = []

        for input_net_change, target, cik in all_data:
            self.all_data_dict["input_net_change"].append(input_net_change)
            self.all_data_dict["target"].append(target)
            self.all_data_dict["cik"].append(cik)
        self.all_data_dict["input_net_change"] = np.concatenate(
            self.all_data_dict["input_net_change"], axis=0
        )

        self.all_data_dict["target"] = np.concatenate(
            self.all_data_dict["target"], axis=0
        )
        self.all_data_dict["cik"] = np.concatenate(self.all_data_dict["cik"], axis=0)

    def __getitem__(self, idx, inference_mode=False):
        filepath = self.data_files[idx]

        try:
            df = (
                pd.read_csv(
                    filepath,
                    dtype={"account_num": self.accounts_index.index.dtype},
                    usecols=["account_num", "tag", "cik", "net_change"],
                )
                .fillna("")
                .set_index("account_num")
            )
            
            allowed_tags = self.tags_index["tag"].values
            
            df = df.query(f"tag in {list(allowed_tags)}")
            
            # df= pd.merge(df,self.tags_index[['tag','tag_depth']].drop_duplicates(),on='tag')
            # df = df.query("tag_depth>=2")

        except Exception as e:
            print(e)
            print(filepath)
            import sys

            sys.exit(-1)

        net_changes_dict = df["net_change"].to_dict()
        cik = df["cik"].iloc[0]

        datas = []
        accounts = self.accounts_index.index
        
        for account in accounts:
            if account in net_changes_dict:
                datas.append(net_changes_dict[account])
            else:
                datas.append(0)

        datas = np.array(datas)

        if self.normalization_type == NormalizationType.NONE:
            pass
        elif self.normalization_type == NormalizationType.LOCAL:
            assets_amount = df.query("tag=='Assets'")["net_change"].values[0]
            datas = self.local_scaling_transform(datas, assets_amount)
        elif self.normalization_type == NormalizationType.LOCAL_WITH_RAW:
            assets_amount = df.query("tag=='Assets'")["net_change"].values[0]
            datas_normalized = self.local_scaling_transform(datas, assets_amount)
            datas = np.concatenate([datas, datas_normalized])

        elif self.normalization_type == NormalizationType.PARENT:
            datas = self.normalize_by_parent_tag(datas, accounts)
        elif self.normalization_type == NormalizationType.LOG:
            datas = self.log_scaling_transform_integer(datas)
        elif self.normalization_type == NormalizationType.INDICATOR:
            datas = self.indicator_representation(datas)

        elif self.normalization_type == NormalizationType.COMMON_PERCENTAGES:
            tags_dict = df.set_index("tag")["net_change"].to_dict()
            common_percentages_values = compute_common_percentages_15(tags_dict)
            datas = list(common_percentages_values.values())
            datas = np.array(datas)

        elif (
            self.normalization_type
            == NormalizationType.COMMON_PERCENTAGES_WITH_BASE_FEATURES
            # noqa
        ):
            tags_dict = df.set_index("tag")["net_change"].to_dict()
            common_percentages_values = compute_common_percentages_15(tags_dict)
            datas = np.concatenate([datas, list(common_percentages_values.values())])

        else:
            raise Exception(f"Unknown normalization type {self.normalization_type}")

        sic = self.registrants_index_dict[cik]
        target = self.sic_id_index[sic]

        return {
            "input_net_change": datas,
            "target": target,
            "sample_idx": idx,
            "cik": cik,
        }

    @property
    def X(self):
        """
        Get the array of input net changes.
        """
        print("x_shape:", self.all_data_dict["input_net_change"].shape)
        return self.all_data_dict["input_net_change"]

    @property
    def Y(self):
        """
        Get the array of targets.
        """
        return self.all_data_dict["target"]

    def get_all_ciks(self):
        """
        Get all the ciks in the dataset.
        """
        return self.all_data_dict["cik"]

    def __len__(self):
        return len(self.data_files)


class SecGBDTDatasetWithTransformerFE(SecGBDTDataset):
    """
    SecGBDT Dataset with transformer text feature extraction from

    """

    def __init__(
        self,
        dataset_dir: str,
        type: DatasetType,
        transformer_encoder: TextTransformerForClassification,
        sic_digits=1,
        seq_max_length=100,
        max_tag_depth=1e10,
        only_transformer_features=False,
        features_encoding_batch_size=64,
    ):
        self.transformer_encoder = transformer_encoder
        self.only_transformer_features = only_transformer_features

        # All the encoded features of the dataset
        self.all_encoded_features = None

        self.text_transformer_dataset = SecTextTransformerDataset(
            dataset_dir=dataset_dir,
            type=type,
            tokenizer=self.transformer_encoder.tokenizer,
            sic_digits=sic_digits,
            seq_max_length=seq_max_length,
            max_tag_depth=max_tag_depth,
        )

        self.features_encoding_batch_size = features_encoding_batch_size
        super().__init__(dataset_dir, type, sic_digits=sic_digits)

    def compute_all_encoded_features(self):
        """
        Encode all the features of the dataset using the transformer encoder.
        Set the tensor of all encoded features in self.all_encoded_features
        :return: None
        """
        # Compute all the encoded features of the dataset

        dataloader = torch.utils.data.DataLoader(
            self.text_transformer_dataset,
            batch_size=self.features_encoding_batch_size,
            shuffle=False,
            num_workers=min(10, multiprocessing.cpu_count() - 2),
        )

        all_encoded_features = []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, "Computing all transformer features"):
                batch = {
                    k: v.to(self.transformer_encoder.device) for k, v in batch.items()
                }

                input, attn_mask = batch["input"], batch["input_attn_mask"]

                out = self.transformer_encoder(input, attn_mask, return_dict=True)
                all_encoded_features.append(out["x_enc"].cpu().numpy())

        self.all_encoded_features = np.concatenate(all_encoded_features, axis=0)

    def load_all_in_memory(self):
        """
        Load all the dataset in memory.
        """

        # 1. First encode all the features with the transformer encoder
        self.compute_all_encoded_features()

        # 2. Load all the data in memory
        all_indices = list(range(self.__len__()))

        max_core = 40
        num_cores = min(max_core, multiprocessing.cpu_count() - 2)
        chunk_size = len(all_indices) // num_cores
        chunked_indices = [
            all_indices[i : i + chunk_size]
            for i in range(0, len(all_indices), chunk_size)
        ]

        def process_chunk(indices):
            inputs_net_change = []
            targets = []
            ciks = []
            for i in tqdm.tqdm(indices):
                data = self.__getitem__(i)
                inputs_net_change.append(data["input_net_change"])
                targets.append(data["target"])
                ciks.append(data["cik"])

            return np.stack(inputs_net_change), np.stack(targets), np.stack(ciks)

        all_data = Parallel(n_jobs=num_cores)(
            delayed(process_chunk)(chunk)
            for chunk in tqdm.tqdm(
                chunked_indices, f"Loading {self.type} dataset in memory"
            )
        )

        self.all_data_dict["input_net_change"] = []
        self.all_data_dict["target"] = []
        self.all_data_dict["cik"] = []

        for input_net_change, target, cik in all_data:
            self.all_data_dict["input_net_change"].append(input_net_change)
            self.all_data_dict["target"].append(target)
            self.all_data_dict["cik"].append(cik)

        if not self.only_transformer_features:
            self.all_data_dict["input_net_change"] = np.concatenate(
                self.all_data_dict["input_net_change"], axis=0
            )
            self.all_data_dict["input_net_change"] = np.concatenate(
                [self.all_data_dict["input_net_change"], self.all_encoded_features],
                axis=1,
            )  # type: ignore
        else:
            self.all_data_dict["input_net_change"] = self.all_encoded_features

        # .Concatenate the encoded features with the net changes

        self.all_data_dict["target"] = np.concatenate(
            self.all_data_dict["target"], axis=0
        )
        self.all_data_dict["cik"] = np.concatenate(self.all_data_dict["cik"], axis=0)


class SecGBDTDatasetWithTFIDF(SecGBDTDataset):
    """
    Sec GBDT Dataset with TFIDF features
    """

    def __init__(
        self, dataset_dir: str, type: DatasetType, sic_digits=1, max_features=2000
    ):
        self.max_features = max_features
        super().__init__(
            dataset_dir,
            type,
            sic_digits,
        )

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

    def __getitem__(self, idx, inference_mode=False) -> dict:
        """
        Read a single sample from the dataset.
        :param idx: index of the sample
        :param inference_mode: whether to return the target or not
        :return: dictionary containing the sample
        """

        df = pd.read_csv(
            self.data_files[idx], usecols=["tag", "cik", "net_change"]
        ).fillna("")

        tag = df["tag"].values.tolist()
        all_tag_as_string = " ".join(tag)

        cik = df["cik"].iloc[0]
        sic = self.registrants_index_dict[cik]
        target = self.sic_id_index[sic]

        return {
            "input_tags": all_tag_as_string,
            "target": target,
            "sample_idx": idx,
            "cik": cik,
        }

    def load_all_in_memory(self):
        """
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
            inputs_tag = []
            targets = []
            ciks = []
            for i in tqdm.tqdm(indices):
                data = self.__getitem__(i)
                inputs_tag.append(data["input_tags"])
                targets.append(data["target"])
                ciks.append(data["cik"])

            return inputs_tag, np.stack(targets), np.stack(ciks)

        all_data = Parallel(n_jobs=num_cores)(
            delayed(process_chunk)(chunk)
            for chunk in tqdm.tqdm(
                chunked_indices, f"Loading {self.type} dataset in memory"
            )
        )

        self.all_data_dict["input_tags"] = []
        self.all_data_dict["target"] = []
        self.all_data_dict["cik"] = []

        for input_tags, target, cik in all_data:
            self.all_data_dict["input_tags"].extend(input_tags)
            self.all_data_dict["target"].append(target)
            self.all_data_dict["cik"].append(cik)

        self.all_data_dict["target"] = np.concatenate(
            self.all_data_dict["target"], axis=0
        )
        self.all_data_dict["cik"] = np.concatenate(self.all_data_dict["cik"], axis=0)

    @property
    def X(self):
        """
        Get the inputs of the dataset.
        """

        X = self.vectorizer.fit_transform(self.all_data_dict["input_tags"])
        return X.toarray()

    @property
    def Y(self):
        """
        Get the array of targets.
        """
        return self.all_data_dict["target"]
