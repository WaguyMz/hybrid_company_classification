GLOBAL_EXP_NAME = "_count30_sic1agg"

"""
Script to preprocess sec dataset without gaap taxonomy tags
"""
import argparse
import multiprocessing
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
from pandas_parallel_apply import DataFrameParallel

dir_path = os.path.realpath(
    os.path.join(
        __file__,
        "..",
        "..",
        "..",
    )
)
sys.path.append(dir_path)

from researchpkg.industry_classification.config import (
    MAX_CORE_USAGE,
    MAX_SUB_PER_CIK,
    MIN_SUB_PER_CIK,
    RANDOM_SEED,
    SEC_MAX_YEAR,
    SEC_MIN_YEAR,
    SEC_RAW_DATA_DIR,
    SEC_RAW_TEST_DATA_DIR,
    SEC_ROOT_DATA_DIR,
    SEC_TAX,
    SEC_TAX_DATA_DIR,
    SEC_TAX_MAX_TAGS_DEPTH,
    SEC_TAX_MIN_TAGS_DEPTH,
    SEC_TAX_VERSION,
    SEC_TRAIN_VAL_RATIO,
    TAGS_PERSHEET_COUNT_THRESHOLD,
    TOP_K_TAGS,
)
from researchpkg.industry_classification.preprocessing.gaap_taxonomy_parser import (
    CalculationTree,
    CalculationTreeType,
)
from researchpkg.industry_classification.preprocessing.raw_sec_dataset_extraction import (
    merge_all_directories_data,
)
from researchpkg.industry_classification.preprocessing.sec_preprocessing_utils import (
    build_sic_count_index,
    format_file_name,
    save_dataset_config,
)

SEC_CLEAN_DATA_DIR = os.path.join(SEC_ROOT_DATA_DIR, GLOBAL_EXP_NAME)
SEC_TAGS_TREE_FILE = os.path.join(SEC_CLEAN_DATA_DIR, "tags_tree.csv")
SEC_BS_DIR = os.path.join(SEC_CLEAN_DATA_DIR, "balance_sheets_agg")
SEC_BS_AGG_DIR = os.path.join(SEC_CLEAN_DATA_DIR, "balance_sheets_agg")

np.random.seed(RANDOM_SEED)


def save_balanced_sheet_agg(
    submission_data,
):
    """
    Save a single proceed to balance sheet to a csv
    :param submission_data: A signle balance sheet submission data
    :return : None
    """

    (registrant_name, year, quarter, adsh,) = (
        submission_data[
            [
                "name",
                "year",
                "quarter",
                "adsh",
            ]
        ]
        .iloc[0]
        .values.tolist()
    )

    is_train = submission_data["train_flag"].values[0]
    is_test = submission_data["test_flag"].values[0]
    subdir = "train" if is_train else "test" if is_test else "val"
    subdir = os.path.join(
        SEC_BS_AGG_DIR,
        subdir,
    )
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    cols = [
        "account_num",
        "net_change",
        "tag",
        "tag_depth",
        "cik",
    ]
    sub = submission_data[cols]

    filename = format_file_name(
        registrant_name,
        year,
        quarter,
        adsh,
    )

    sub.to_csv(
        os.path.join(
            subdir,
            filename,
        ),
        index=False,
    )


def generate_bs_dataset(
    df_merged: pd.DataFrame,
    min_year=SEC_MIN_YEAR,
    max_year=SEC_MAX_YEAR,
    fill_missing_values=True,
):
    """
    Generate balanced sheet dataset from merge dataframe.

    :param df_merged  A pandas datafame merging all data from various directories.
    :param min_year: minimum year to consider
    :param max_year: maximum year to conside
    :param fill_missing_values: If true, fill missing values using the calculation tree
    :return : None
    """

    # 1. First filter by number of submssions pergenerate_bs_da cik
    # 1.1. MIN_SUB_PER_CIK
    df_merged = df_merged.groupby("cik").filter(
        lambda x: x.adsh.nunique() >= MIN_SUB_PER_CIK
    )

    # 1.2. MAX_SUB_PER_CIK
    retain_n_random_items = lambda group_df: group_df.sample(
        n=min(
            MAX_SUB_PER_CIK,
            group_df.shape[0],
        )
    )
    df_adsh_selected = df_merged[
        [
            "cik",
            "adsh",
        ]
    ].drop_duplicates()
    df_adsh_selected = (
        df_adsh_selected.groupby("cik")
        .apply(retain_n_random_items)
        .reset_index(drop=True)
    )
    df_merged = pd.merge(
        df_merged,
        df_adsh_selected,
        on=[
            "adsh",
            "cik",
        ],
    )

    # Use sicagg as sic1
    df_merged.drop(
        columns=["sic1"],
        inplace=True,
    )
    df_merged["sic1"] = df_merged["sicagg"]
    df_merged["net_change"] = df_merged.value.abs()
    df_merged = df_merged[df_merged.net_change != 0].copy()
    df_merged.drop(
        columns=["value"],
        inplace=True,
    )
    df_merged.rename(
        columns={
            "sic": "sic4",
            "doc": "description",
        },
        inplace=True,
    )
    # df_merged.dropna(subset="description",inplace=True)  # removing tag without doc.

    # 0. Create  directories
    index_dir = os.path.join(
        SEC_CLEAN_DATA_DIR,
        "index",
    )
    for d in [
        index_dir,
        SEC_BS_DIR,
        SEC_BS_AGG_DIR,
    ]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Extract calculation tree to fill missing values
    bs_calculation_tree = CalculationTree.build_taxonomy_tree(
        SEC_TAX_DATA_DIR,
        SEC_TAX,
        SEC_TAX_VERSION,
        type=CalculationTreeType.BALANCE_SHEET,
    )

    # 1. Index of tags(tags without version)
    tags_index = df_merged[
        [
            "tag",
            "tag_depth",
            "version",
            "crdr",
            "tlabel",
            "description",
            "value_sign",
        ]
    ].drop_duplicates()

    # Update tag detph as follow: If the tag is the taxonomy tree, then the tag depth is the depth of the tag in the tree
    # Otherwise, the tag depth is -1
    def get_tag_depth(
        x,
    ):
        node = bs_calculation_tree.get_node_by_concept_name(x)
        if node != None:
            return node.level
        else:
            return -1

    tags_index["tag_depth"] = tags_index["tag"].apply(get_tag_depth)
    tags_index.to_csv(
        os.path.join(
            index_dir,
            "tags_index.csv",
        ),
        index=False,
    )

    # 2.Index of accounts (tags without version)
    accounts_index = pd.DataFrame(
        df_merged[
            [
                "adsh",
                "tag",
                "tag_depth",
                "description",
                "line",
                "crdr",
                "tlabel",
                "value_sign",
                "custom",
            ]
        ].copy()
    )

    # 1.1. Ordering tags by line(average relative positive (rel_line)
    accounts_index = accounts_index.sort_values(
        by=[
            "adsh",
            "line",
        ]
    )
    accounts_index["line"] = accounts_index.groupby("adsh").cumcount() + 1
    accounts_index["rel_line"] = accounts_index.groupby("adsh")["line"].transform(
        lambda x: 100 * x / x.shape[0]
    )
    accounts_index["count_tag"] = 1
    accounts_index = accounts_index.groupby("tag").aggregate(
        {
            "line": "mean",
            "rel_line": "mean",
            "count_tag": "sum",
            "description": "first",
            "tag_depth": "first",
            "crdr": "first",
            "tlabel": "first",
            "value_sign": "first",
        }
    )
    accounts_index = accounts_index.sort_values(by="rel_line").reset_index()
    accounts_index["account_num"] = accounts_index.index

    # 2. Index of registrants
    registrants_index = df_merged[
        [
            "cik",
            "name",
            "sic4",
            "sic1",
            "sic2",
            "sic3",
            "countryba",
            "bas1",
        ]
    ].drop_duplicates("cik")
    registrants_index.to_csv(
        os.path.join(
            index_dir,
            "registrants_index.csv",
        ),
        index=False,
    )

    KEEP_ONLY_TOP_TAGS = False
    # 5. Apply tags filtering( top k tags and min number of tags per sheet)
    top_k_tags = []
    if KEEP_ONLY_TOP_TAGS:
        tag_counts = df_merged["tag"].value_counts()

        top_k_tags = tag_counts.head(
            min(
                TOP_K_TAGS,
                tag_counts.shape[0],
            )
        )
        df_merged = df_merged[df_merged["tag"].isin(top_k_tags.index)]

    print("Number of tags :", df_merged.tag.nunique())

    print(f"Filtering sheets with less than {TAGS_PERSHEET_COUNT_THRESHOLD} tags")
    df_merged["nline"] = df_merged["nline_bs"]
    df_merged = df_merged.loc[df_merged.nline >= TAGS_PERSHEET_COUNT_THRESHOLD]

    # 4. Index of submissions
    submissions_index = df_merged[
        [
            "adsh",
            "cik",
            "sic1",
            "sic2",
            "sic3",
            "sic4",
            "nline",
            "name",
            "year",
            "quarter",
            "test_flag",
        ]
    ].drop_duplicates()

    mandatory_tags = ["Assets"]
    df_values = df_merged[
        [
            "adsh",
            "tag",
            "net_change",
        ]
    ].copy()
    new_values = {}
    nb_new_tags = []

    if fill_missing_values:
        for (adsh, group,) in tqdm.tqdm(
            df_values.groupby("adsh"),
            desc="Filling missing values",
        ):
            values_dict = group.set_index("tag")["net_change"].to_dict()
            calculated_values = bs_calculation_tree.apply(
                values_dict,
                min_level=SEC_TAX_MIN_TAGS_DEPTH,
                max_level=SEC_TAX_MAX_TAGS_DEPTH,
            )

            new_tags = [k for k, v in calculated_values.items() if k not in values_dict]

            if KEEP_ONLY_TOP_TAGS:
                new_tags = [k for k in new_tags if k in top_k_tags.index]
                values_dict.update(
                    {k: v for k, v in calculated_values.items() if k in new_tags}
                )
            else:
                values_dict = calculated_values

            nb_new_tags.append(len(new_tags))
            is_valid = True
            for tag in mandatory_tags:
                if tag not in values_dict:
                    is_valid = False
                    break
            if is_valid:
                new_values[adsh] = values_dict
    else:
        # Keep the original values.
        for (adsh, group,) in tqdm.tqdm(
            df_values.groupby("adsh"),
            desc="",
        ):
            values_dict = group.set_index("tag")["net_change"].to_dict()
            new_values[adsh] = values_dict
            nb_new_tags.append(0)

    avg_nb_calculated_tag = np.mean(nb_new_tags).item()
    print(
        "Average number of new tags per sheet",
        avg_nb_calculated_tag,
    )

    # 6.1. Add new values to the dataframe
    new_values_df = pd.DataFrame(new_values).T.reset_index()
    new_values_df = pd.melt(
        new_values_df,
        id_vars="index",
        value_vars=new_values_df.columns[1:],
        var_name="tag",
        value_name="net_change",
    ).dropna()
    new_values_df.rename(
        columns={"index": "adsh"},
        inplace=True,
    )

    # Merge with the tags index to get the tag depth
    df_merged = pd.merge(
        new_values_df,
        accounts_index,
        on=["tag"],
    )
    df_merged = pd.merge(
        df_merged,
        submissions_index,
        on="adsh",
    )

    df_merged["nline_after_filtering"] = df_merged.groupby("adsh")["tag"].transform(
        "size"
    )

    # Only balance sheets which have Assets
    ok_adsh = df_merged[df_merged.tag == "Assets"].adsh.unique()
    df_merged = df_merged[df_merged.adsh.isin(ok_adsh)]

    # Accounts stats
    df_account_stats = (
        df_merged.groupby("account_num")
        .agg(
            tag=(
                "tag",
                "first",
            ),  # "tag
            tag_depth=(
                "tag_depth",
                "first",
            ),
            crdr=(
                "crdr",
                "first",
            ),
            value_sign=(
                "value_sign",
                "first",
            ),
            description=(
                "description",
                "first",
            ),
            account_name=(
                "tlabel",
                "first",
            ),
            mean=(
                "net_change",
                "mean",
            ),
            median=(
                "net_change",
                "median",
            ),
            std=(
                "net_change",
                "std",
            ),
            min=(
                "net_change",
                "min",
            ),
            max=(
                "net_change",
                "max",
            ),
            count=(
                "net_change",
                "count",
            ),
            quantile_25=(
                "net_change",
                lambda x: np.percentile(
                    x,
                    25,
                ),
            ),
            quantile_75=(
                "net_change",
                lambda x: np.percentile(
                    x,
                    75,
                ),
            ),
        )
        .reset_index()
        .drop_duplicates()
    )
    # Save account stats.
    df_account_stats.to_csv(
        os.path.join(
            index_dir,
            "accounts_index.csv",
        ),
        index=False,
    )

    # 8. Train validation test splitting
    df_merged_trainval = df_merged[~df_merged.test_flag]
    df_merged_trainval["nb_sub"] = df_merged_trainval.groupby("cik")["adsh"].transform(
        "nunique"
    )
    df_merged_trainval.sort_values(
        by=["cik"],
        ascending=[True],
        inplace=True,
    )
    df_merged_trainval["sub"] = 1
    df_merged_trainval["cum_nb_sub"] = df_merged_trainval["sub"].cumsum()
    df_merged_trainval["cum_nb_sub"] = df_merged_trainval.groupby("cik")[
        "cum_nb_sub"
    ].transform("max")
    cutoff_valid = len(df_merged_trainval) * (1 - SEC_TRAIN_VAL_RATIO)

    df_merged_trainval["train_flag"] = df_merged_trainval.cum_nb_sub >= cutoff_valid

    df_test = df_merged[df_merged.test_flag]
    df_test["train_flag"] = False

    df_merged = pd.concat([df_merged_trainval, df_test])

    df_data = df_merged[
        [
            "adsh",
            "account_num",
            "net_change",
            "train_flag",
            "test_flag",
            "sic1",
            "sic2",
            "sic3",
            "sic4",
        ]
    ]

    build_sic_count_index(
        df_data,
        index_dir,
    )

    # 11. Save all individual balance sheets
    df_data = pd.merge(
        df_data,
        accounts_index[
            [
                "account_num",
                "tag",
                "tag_depth",
            ]
        ],
        on="account_num",
        how="left",
    )

    df_data = pd.merge(
        df_data,
        submissions_index[
            [
                "adsh",
                "cik",
                "year",
                "quarter",
                "name",
            ]
        ],
        on="adsh",
        how="left",
    )

    # Fill nan with the first value of the column
    df_data["cik"] = df_data["cik"].fillna(method="ffill")
    df_data["year"] = df_data["year"].fillna(method="ffill")
    df_data["quarter"] = df_data["quarter"].fillna(method="ffill")
    df_data["name"] = df_data["name"].fillna(method="ffill")

    DataFrameParallel(
        df_data,
        n_cores=min(multiprocessing.cpu_count() - 1, MAX_CORE_USAGE),
        pbar=True,
    ).groupby("adsh").apply(save_balanced_sheet_agg)

    # 3. Save dataset config
    save_dataset_config(
        dataset_dir=SEC_CLEAN_DATA_DIR,
        seed=RANDOM_SEED,
        min_year=min_year,
        max_year=max_year,
        sic1_used=sorted(df_data.sic1.unique().tolist()),
        nb_cik=df_data.cik.nunique(),
        max_sub_per_cik=MAX_SUB_PER_CIK,
        min_sub_per_cik=MIN_SUB_PER_CIK,
        top_k_tags=TOP_K_TAGS if KEEP_ONLY_TOP_TAGS else 0,
        tags_persheet_count_threshold=TAGS_PERSHEET_COUNT_THRESHOLD,
        dataset_size=df_data.adsh.nunique(),
        train_dataset_size=df_data[df_data.train_flag].adsh.nunique(),
        val_dataset_size=df_data[
            ((~df_data.train_flag) & (~df_data.test_flag))
        ].adsh.nunique(),
        test_dataset_size=df_data[df_data.test_flag].adsh.nunique(),
        avg_initial_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline",
            ]
        ]
        .drop_duplicates()
        .nline.mean()
        .item(),
        avg_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline_after_filtering",
            ]
        ]
        .drop_duplicates()
        .nline_after_filtering.mean()
        .item(),
        max_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline_after_filtering",
            ]
        ]
        .drop_duplicates()
        .nline_after_filtering.max()
        .item(),
        min_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline_after_filtering",
            ]
        ]
        .drop_duplicates()
        .nline_after_filtering.min()
        .item(),
        avg_nb_calculated_tag_per_sheet=avg_nb_calculated_tag,
        quantile_5_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline_after_filtering",
            ]
        ]
        .drop_duplicates()
        .nline_after_filtering.quantile(0.05)
        .item(),
        quantile_25_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline_after_filtering",
            ]
        ]
        .drop_duplicates()
        .nline_after_filtering.quantile(0.25)
        .item(),
        quantile_75_nb_tag_per_sheet=df_merged[
            [
                "adsh",
                "nline_after_filtering",
            ]
        ]
        .drop_duplicates()
        .nline_after_filtering.quantile(0.75)
        .item(),
        min_tag_depth=SEC_TAX_MIN_TAGS_DEPTH,
        max_tag_depth=SEC_TAX_MAX_TAGS_DEPTH,
    )


if __name__ == "__main__":
    begin = datetime.now()

    parser = argparse.ArgumentParser(
        prog="Run sec preprocessing with gaap",
        description="Preprocess SEC data to extract balance sheets",
    )
    parser.add_argument(
        "--min_year",
        default=SEC_MIN_YEAR,
        type=int,
    )
    parser.add_argument(
        "--max_year",
        default=SEC_MAX_YEAR,
        type=int,
    )
    parser.add_argument(
        "--no_fill_missing_values",
        action="store_true",
    )
    args = parser.parse_args()

    if os.path.exists(SEC_CLEAN_DATA_DIR):
        shutil.rmtree(SEC_CLEAN_DATA_DIR)

    df_merged_train = merge_all_directories_data(
        SEC_RAW_DATA_DIR,
        min_year=args.min_year,
        max_year=args.max_year,
        only_balance_sheet_tags=False,
    )
    df_merged_train["test_flag"] = False

    df_merged_test = merge_all_directories_data(
        SEC_RAW_TEST_DATA_DIR,
        min_year=1990,
        max_year=2100,
        only_balance_sheet_tags=False,
    )
    # For df_merged_test only keep cik which are not in df_merged_train
    cik_train = df_merged_train.cik.unique()
    df_merged_test = df_merged_test[~df_merged_test.cik.isin(cik_train)]
    df_merged_test["test_flag"] = True

    df_merged = pd.concat([df_merged_train, df_merged_test])

    generate_bs_dataset(
        df_merged,
        min_year=args.min_year,
        max_year=args.max_year,
        fill_missing_values=not args.no_fill_missing_values,
    )

    duration = datetime.now() - begin
    print(
        "Process duration ",
        duration,
    )
