import multiprocessing
import os

import pandas as pd
import pytorch_lightning as pl
import torch.nn
import torch.utils
import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from researchpkg.industry_classification.config import MAX_CORE_USAGE
from researchpkg.industry_classification.dataset.sec_datamodule import SecDataset
from researchpkg.industry_classification.dataset.utils import DatasetType


class SecTrfClassificationDataset(SecDataset):
    """
    SecDataset Class to classify Type of business using CNN
    """

    def __init__(
        self,
        dataset_dir: str,
        type: DatasetType,
        tokenizer,
        sic_digits=1,
        max_desc_len=12,
        max_tag_depth=None,
        max_tags=100,
        load_in_memory=True,
        use_change=True,
        compute_label_weights=False,
        weighted_loss=False,
    ):
        """
        Dataset for sic classification using a transformer model combining
        text and amounts.
        """
        super().__init__(dataset_dir, type, sic_digits=sic_digits, use_aggregate=True)

        self.sic_digits = sic_digits
        self.sic_col = f"sic{sic_digits}"
        self.tokenizer = tokenizer
        self.max_desc_len = max_desc_len
        self.max_tag_depth = max_tag_depth
        self.max_tags = max_tags
        self.registrants_index_dict = self.registrants_index.astype(
            {self.sic_col: int}
        )[self.sic_col].to_dict()
        self.filter_data()
        self.account_descriptions_index = self.accounts_index["description"].to_dict()

        self.all_data_dict = {}
        self.data_in_memory = False
        self.use_change = use_change
        self.weighted_loss = weighted_loss

        self.label_weights = None

        if self.weighted_loss:
            self.compute_label_weights()
            self.class_loss_weights = self.label_weights
        else:
            self.class_loss_weights = None

        # Load tags_index
        tag_index_file = os.path.join(dataset_dir, "index", "tags_index.csv")
        self.tags_index = pd.read_csv(tag_index_file, index_col=None)[
            ["tag", "tlabel", "tag_depth"]
        ].drop_duplicates(subset="tlabel")

        if self.max_tag_depth is None:
            pass
        else:
            max_depth = self.max_tag_depth or 1000
            # If max_tag_depth is not None, then we only keep the first max_tag_depth tags
            relevant_tags = self.tags_index.query(
                f"tag_depth <= {max_depth} & tag_depth != -1 "
            )["tag"].values.tolist()
            print(
                f"Max depth : {self.max_tag_depth}, , Number of tags : {len(relevant_tags)}"
            )

        if load_in_memory:
            self.load_all_in_memory()
            self.data_in_memory = True

        # Compute class weights
        if compute_label_weights:
            print("Computing label weights")
            self.compute_label_weights()

            # Label weights

        print("Dataset Size : ", len(self.data_files))

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

    def load_all_in_memory(self):
        """
        Load all the dataset in memory.
        """

        max_core = 40
        num_cores = min(max_core, multiprocessing.cpu_count() - 2)

        all_indices = list(range(self.__len__()))
        chunk_size = len(all_indices) // num_cores
        chunked_indices = [
            all_indices[i : i + chunk_size]
            for i in range(0, len(all_indices), chunk_size)
        ]

        def process_chunk(indices):
            inputs_desc = []
            inputs_net_change = []
            inputs_attn_mask = []
            targets = []
            for i in tqdm.tqdm(indices):
                data = self.__getitem__(i)
                inputs_desc.append(data["input_desc"])
                inputs_net_change.append(data["input_net_change"])
                inputs_attn_mask.append(data["input_attn_mask"])
                targets.append(data["target"])

            return (
                torch.stack(inputs_desc),
                torch.stack(inputs_net_change),
                torch.stack(inputs_attn_mask),
                torch.stack(targets),
            )

        all_data = Parallel(n_jobs=num_cores)(
            delayed(process_chunk)(chunk)
            for chunk in tqdm.tqdm(
                chunked_indices, f"Loading {self.type} dataset in memory"
            )
        )

        self.all_data_dict["input_desc"] = []
        self.all_data_dict["input_net_change"] = []
        self.all_data_dict["input_attn_mask"] = []
        self.all_data_dict["target"] = []

        for input_desc, input_net_change, input_attn_mask, target in all_data:
            self.all_data_dict["input_desc"].append(input_desc)
            self.all_data_dict["input_net_change"].append(input_net_change)
            self.all_data_dict["input_attn_mask"].append(input_attn_mask)
            self.all_data_dict["target"].append(target)

        self.all_data_dict["input_desc"] = torch.cat(
            self.all_data_dict["input_desc"], dim=0
        )
        self.all_data_dict["input_net_change"] = torch.cat(
            self.all_data_dict["input_net_change"], dim=0
        )
        self.all_data_dict["input_attn_mask"] = torch.cat(
            self.all_data_dict["input_attn_mask"], dim=0
        )

        self.all_data_dict["target"] = torch.cat(self.all_data_dict["target"], dim=0)

    def __getitem__(self, idx, inference_mode=False):
        if self.data_in_memory:
            desc_tokens = self.all_data_dict["input_desc"][idx]
            net_change = self.all_data_dict["input_net_change"][idx]
            attn_mask = self.all_data_dict["input_attn_mask"][idx]
            target = self.all_data_dict["target"][idx]

            sample = {
                "input_desc": desc_tokens,
                "input_net_change": net_change,
                "input_attn_mask": attn_mask,
                "sample_idx": idx,
                "target": target,
            }

            if self.class_loss_weights is not None:
                sample["class_weights"] = self.class_loss_weights[idx]

            return sample

        else:
            filepath = self.data_files[idx]
            df = pd.read_csv(
                filepath,
                dtype={"account_num": self.accounts_index.index.dtype},
                usecols=["account_num", "cik", "tag", "net_change"],
            ).fillna("")

            cik = df["cik"][0]
            sic = self.registrants_index_dict[cik]
            target = torch.tensor(self.sic_id_index[sic])

            df = pd.merge(df, self.tags_index, on="tag").drop_duplicates(subset="tag")

            if self.max_tag_depth is not None:
                # If max_tag_depth is not None, then we only keep the first max_tag_depth tags
                # Tags with detph=-1 are not considered(they are not in the taxonomy)
                df = df[df.tag_depth <= self.max_tag_depth].copy()

            all_desc = df.tlabel.values.tolist()
            accounts = self.accounts_index.loc[df["account_num"].values].idx.values
            accounts = torch.from_numpy(accounts).to(dtype=torch.long)
            # n_accounts = len(self.accounts_index)

            tokenizer_out = self.tokenizer(
                all_desc,
                padding="max_length",
                truncation=True,
                max_length=self.max_desc_len,
                return_tensors="pt",
            )
            desc_tokens = tokenizer_out["input_ids"]
            attn_mask = tokenizer_out["attention_mask"]

            desc_tokens = torch.nn.functional.pad(
                desc_tokens,
                (0, 0, 0, self.max_tags - desc_tokens.shape[0]),
                "constant",
                0,
            )

            # Paddding all accounts as well
            accounts = torch.nn.functional.pad(
                accounts, (0, self.max_tags - accounts.shape[0])
            )

            attn_mask = torch.nn.functional.pad(
                attn_mask, (0, 0, 0, self.max_tags - attn_mask.shape[0]), "constant", 0
            )

            if self.use_change:
                net_change = torch.from_numpy(df["net_change"].values).to(
                    dtype=torch.float32
                )

                # Std scaling
                # net_change = self.std_scaling_transform(net_change, df.account_num)
                net_change = self.log_scaling_transform(net_change)

                # assets_amount = df.query("tag == 'Assets'")["net_change"].values[0]
                # net_change = self.local_scaling_transform(net_change, assets_amount)

                net_change = torch.nn.functional.pad(
                    net_change, (0, self.max_tags - net_change.shape[0]), "constant", 0
                )

            else:
                # net_change is not used so just fill zeros.
                net_change = torch.ones(self.max_tags)

            net_change = net_change.unsqueeze(1)
            assert torch.isnan(net_change).sum() == 0

        tags = df["tag"].values
        if not inference_mode :
            
            return {
                "input_desc": desc_tokens,
                "input_net_change": net_change,
                "input_attn_mask": attn_mask,
                "target": target,
                "sample_idx": idx,
                "tags": ";".join(tags),
            }

        else:
            # In inference mode, add more data.
            df = pd.merge(
                df.reset_index(), self.registrants_index, how="left", on="cik"
            )
            df_updated = (
                pd.merge(
                    df, self.accounts_index.reset_index(), how="right", on="account_num"
                )
                .fillna(0)
                .set_index("account_num")
            )

            return {
                "input_desc": desc_tokens,
                "input_net_change": net_change,
                "input_attn_mask": attn_mask,
                "input_accounts": accounts,
                "raw": df_updated,
                "class_weights": (
                    torch.from_numpy(self.class_loss_weights[idx])
                    if self.class_loss_weights is not None
                    else None
                ),
                "sample_idx": idx,
                "target": target,
                "tags": ";".join(tags),
            }

    def __len__(self):
        return len(self.data_files)


class SecTrfClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int = 0,
        load_in_memory: bool = True,
        sic_digits: int = 3,
        max_desc_len: int = 64,
        max_tags: int = 100,
        use_change: bool = True,
        balance_sampling: bool = False,
        weighted_loss: bool = False,
        max_tag_depth: int = None,
    ):
        """
        :param dataset: The folder of the dat
        :param embedding_experiment : The experiment name of the  word embedding model.
        :param batch_size: represents the batch size for the data.
        :param resize: represents the resize value for the image. e.g. (256, 256)
        :param augment: represents the data augmentation techniques.

        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.sic_digits = sic_digits
        self.tokenizer = tokenizer
        self.load_in_memory = load_in_memory
        self.use_change = use_change
        self.max_tag_depth = max_tag_depth

        self.train_dataset = SecTrfClassificationDataset(
            dataset_dir=self.dataset_dir,
            type=DatasetType.TRAIN,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            load_in_memory=self.load_in_memory,
            max_desc_len=max_desc_len,
            use_change=self.use_change,
            compute_label_weights=balance_sampling,
            weighted_loss=weighted_loss,
            max_tag_depth=self.max_tag_depth,
        )

        self.val_dataset = SecTrfClassificationDataset(
            dataset_dir=self.dataset_dir,
            type=DatasetType.VAL,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            max_desc_len=max_desc_len,
            max_tags=max_tags,
            load_in_memory=self.load_in_memory,
            use_change=self.use_change,
            compute_label_weights=balance_sampling,
            weighted_loss=weighted_loss,
            max_tag_depth=self.max_tag_depth,
        )

        self.test_dataset = SecTrfClassificationDataset(
            dataset_dir=self.dataset_dir,
            type=DatasetType.TEST,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            max_desc_len=max_desc_len,
            max_tags=max_tags,
            load_in_memory=self.load_in_memory,
            use_change=self.use_change,
            compute_label_weights=balance_sampling,
            weighted_loss=weighted_loss,
            max_tag_depth=self.max_tag_depth,
        )

    def train_dataloader(self, shuffle=True) -> DataLoader:
        if self.train_dataset.label_weights is not None:
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


class SecTrfClassificationDataSetMultiScaling(SecTrfClassificationDataset):
    def __init__(
        self,
        dataset_dir: str,
        type: DatasetType,
        tokenizer,
        sic_digits=1,
        max_desc_len=12,
        max_tags=100,
        load_in_memory=True,
        use_change=True,
    ):
        """
        Identical to super class constructor
        """
        super().__init__(
            dataset_dir,
            type,
            tokenizer,
            sic_digits=sic_digits,
            max_desc_len=max_desc_len,
            max_tags=max_tags,
            load_in_memory=load_in_memory,
            use_change=use_change,
        )

    def __getitem__(self, idx, inference_mode=False):
        if self.data_in_memory:
            desc_tokens = self.all_data_dict["input_desc"][idx]
            net_change = self.all_data_dict["input_net_change"][idx]
            attn_mask = self.all_data_dict["input_attn_mask"][idx]
            target = self.all_data_dict["target"][idx]
            tags = self.all_data_dict["tags"][idx]
            sample = {
                "input_desc": desc_tokens,
                "input_net_change": net_change,
                "input_attn_mask": attn_mask,
                "sample_idx": idx,
                "target": target,
                "tags": tags,
            }

            if self.class_loss_weights is not None:
                sample["class_weights"] = torch.tensor(self.class_loss_weights[idx])
            return sample

        else:
            filepath = self.data_files[idx]
            df = pd.read_csv(
                filepath,
                dtype={"account_num": self.accounts_index.index.dtype},
                usecols=["account_num", "cik", "tag", "net_change"],
            ).fillna("")

            df = pd.merge(df, self.tags_index, on="tag").drop_duplicates(subset="tag")

            all_desc = df.tlabel.values.tolist()
            accounts = self.accounts_index.loc[df["account_num"].values].idx.values
            accounts = torch.from_numpy(accounts).to(dtype=torch.long)
            # n_accounts = len(self.accounts_index)

            tokenizer_out = self.tokenizer(
                all_desc,
                padding="max_length",
                truncation=True,
                max_length=self.max_desc_len,
                return_tensors="pt",
            )
            desc_tokens = tokenizer_out["input_ids"]
            attn_mask = tokenizer_out["attention_mask"]

            desc_tokens = torch.nn.functional.pad(
                desc_tokens,
                (0, 0, 0, self.max_tags - desc_tokens.shape[0]),
                "constant",
                0,
            )

            # Paddding all accounts as well
            accounts = torch.nn.functional.pad(
                accounts, (0, self.max_tags - accounts.shape[0])
            )

            attn_mask = torch.nn.functional.pad(
                attn_mask, (0, 0, 0, self.max_tags - attn_mask.shape[0]), "constant", 0
            )

            if self.use_change:
                net_change = torch.from_numpy(df["net_change"].values).to(
                    dtype=torch.float32
                )

                # Std scaling
                net_change_std = self.std_scaling_transfoqrm(net_change, df.account_num)
                # net_change_log = self.log_scaling_transform_integer(net_change)

                # Assets scaling
                assets_amount = df.query("tag == 'Assets'")["net_change"].values[0]
                net_change_assets = self.local_scaling_transform(
                    net_change, assets_amount
                )
                net_change = torch.stack([net_change_std, net_change_assets], dim=1)

                # Padding to max_tags
                net_change = torch.nn.functional.pad(
                    net_change,
                    (0, 0, 0, self.max_tags - net_change.shape[0]),
                    "constant",
                    0,
                )

            else:
                # net_change is not used so just fill zeros.
                net_change = torch.zeros(self.max_tags)

            cik = df["cik"][0]
            sic = self.registrants_index_dict[cik]
            target = torch.tensor(self.sic_id_index[sic])
            tags = df["tag"].values

        if not inference_mode:
            # Randomly permute the order of the tags

            perm_tags = torch.randperm(len(tags))
            perm  = torch.cat([perm_tags, torch.arange(len(tags), self.max_tags)])
            desc_tokens = desc_tokens[perm]
            net_change = net_change[perm]
            attn_mask = attn_mask[perm]
            tags = tags[perm_tags]

            return {
                "tags": ";".join(tags),
                "input_desc": desc_tokens,
                "input_net_change": net_change,
                "input_attn_mask": attn_mask,
                "target": target,
                "sample_idx": idx,
            }

        else:
            # In inference mode, add more data.
            df = pd.merge(
                df.reset_index(), self.registrants_index, how="left", on="cik"
            )
            df_updated = (
                pd.merge(
                    df, self.accounts_index.reset_index(), how="right", on="account_num"
                )
                .fillna(0)
                .set_index("account_num")
            )

            return {
                "input_desc": desc_tokens,
                "input_net_change": net_change,
                "input_attn_mask": attn_mask,
                "input_accounts": accounts,
                "raw": df_updated,
                "sample_idx": idx,
                "target": target,
                "tags": ";".join(tags),
            }


class SecTrfClassificationMultiScalingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_workers: int = 0,
        load_in_memory: bool = True,
        sic_digits: int = 3,
        max_desc_len: int = 64,
        max_tags: int = 100,
        use_change: bool = True,
    ):
        """
        :param dataset: The folder of the dat
        :param embedding_experiment : The experiment name of the  word embedding model.
        :param batch_size: represents the batch size for the data.
        :param resize: represents the resize value for the image. e.g. (256, 256)
        :param augment: represents the data augmentation techniques.

        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.sic_digits = sic_digits
        self.tokenizer = tokenizer
        self.load_in_memory = load_in_memory
        self.use_change = use_change

        self.train_dataset = SecTrfClassificationDataSetMultiScaling(
            dataset_dir=self.dataset_dir,
            type=DatasetType.TRAIN,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            load_in_memory=self.load_in_memory,
            max_desc_len=max_desc_len,
            max_tags=max_tags,
            use_change=self.use_change,
        )

        self.val_dataset = SecTrfClassificationDataSetMultiScaling(
            dataset_dir=self.dataset_dir,
            type=DatasetType.VAL,
            tokenizer=self.tokenizer,
            sic_digits=self.sic_digits,
            max_desc_len=max_desc_len,
            max_tags=max_tags,
            load_in_memory=self.load_in_memory,
            use_change=self.use_change,
        )

    def train_dataloader(self, shuffle=True) -> DataLoader:
        if self.train_dataset.label_weights is not None:
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
