import glob
import multiprocessing
import os

import pandas as pd
import tqdm
from joblib import Parallel, delayed

from researchpkg.industry_classification.config import (
    MAX_CORE_USAGE,
    SEC_FILENAMES,
    SIC1_EXCLUDED,
)
from researchpkg.industry_classification.preprocessing.sec_preprocessing_utils import (
    get_ith_label,
    normalize_tlabel,
)


def get_sicagg(sic2):
    sic2 = str(sic2)
    if sic2.startswith("0"):
        return "0"
    if sic2 in ["10", "11", "12", "13", "14"]:
        return "10"
    if sic2 in ["15", "16", "17"]:
        return "15"
    if sic2 in [
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "38",
        "39",
    ]:
        return "20"
    if sic2 in ["40", "41", "42", "43", "44", "45", "46", "47", "48", "49"]:
        return "40"
    if sic2 in ["50", "51"]:
        return "50"

    if sic2 in ["52", "53", "54", "55", "56", "57", "58", "59"]:
        return "52"
    if sic2 in ["60", "61", "62", "63", "64", "65", "67"]:
        return "60"
    if sic2 in [
        "70",
        "72",
        "73",
        "75",
        "76",
        "78",
        "79",
        "80",
        "81",
        "82",
        "83",
        "84",
        "85",
        "86",
        "87",
        "89",
    ]:
        return "70"
    if sic2 in ["90,91", "92", "93", "94", "95", "96", "97", "98", "99"]:
        return "90"

    return "0"


def load_quarter_dataset(directory):
    """
    Extract  sub, tag, num and pre dataset for a specific quarter.

    :param directory: Path to the directory containing the quarter dataset
    :return: A dict of pandas dataframe
    """
    dataset = {}

    dataset_name = os.path.basename(directory)
    dataset_info = {"year": dataset_name[:4], "quarter": dataset_name[4:]}
    for filename in SEC_FILENAMES:
        dataset_file = os.path.join(directory, f"{filename}.txt")
        if filename == "sub":
            # ensure sic column is read as string
            dataset[filename] = pd.read_csv(
                dataset_file, sep="\t", low_memory=False, dtype={"sic": str}
            )
        else:
            dataset[filename] = pd.read_csv(dataset_file, sep="\t", low_memory=False)

    return dataset, dataset_info


def extract_quarter_dataset(directory, only_balance_sheet_tags=True):
    """
    Extract a single quarter dataset and return dataset with all relevant columns.

    :param directory: Path to the directory containing the quarter dataset
    :param only_balance_sheet_tags: If true, only consider balance sheets tags.(stmt == 'BS')
    :return: A pandas dataframe
    """

    dataset, dataset_info = load_quarter_dataset(directory)
    df_sub, df_tag, df_num, df_pre = (
        dataset["sub"],
        dataset["tag"],
        dataset["num"],
        dataset["pre"],
    )

    # Remove duplicates from df_pre
    if only_balance_sheet_tags:
        df_pre = df_pre.query("stmt == 'BS'").drop_duplicates(subset=["adsh", "tag"])
    else:
        # def transform_stmt(stmt_list):
        #     if "BS" in stmt_list:
        #         return "BS"
        #     elif "IS"in stmt_list:
        #         return "IS"
        #     else:
        #         return stmt_list[0]
        # df_pre["stmt"] = df_pre.groupby("tag")["stmt"].transform(transform_stmt)
        df_pre = df_pre.drop_duplicates(subset=["adsh", "tag", "stmt"])

    df_tag = df_tag[(df_tag.datatype == "monetary") & (df_tag.custom == 0)]
    df_tag = df_tag[["tag", "version", "datatype", "custom", "crdr", "tlabel", "doc"]]

    # Remove doc
    # df_tag["doc"] = ""  # TODO : Handle this with a parameter

    # Only tag where version is Us-gaap( 'gaap' df_tag.version.str.contains("gaap"))
    df_tag = df_tag[df_tag.version.str.contains("gaap")]

    df_tag.dropna(subset=["tag", "crdr", "datatype", "tlabel"], inplace=True)
    df_tag["value_sign"] = df_tag["crdr"].apply(lambda x: -1 if x.lower() == "c" else 1)
    df_num = df_num.dropna(subset=["value", "adsh"])
    df_num = df_num.sort_values(by=["adsh", "tag", "ddate"]).drop_duplicates(
        subset=["adsh", "tag"], keep="last"
    )

    for i in range(1, 5):
        df_tag[f"label{i}"] = df_tag.tlabel.apply(lambda x: get_ith_label(x, i))

    df_sub_min = df_sub[["adsh", "cik", "name", "sic", "countryba", "bas1"]].fillna("0")

    # Create a new cik column
    df_sub_min["cik"] = df_sub_min["cik"].astype(str) + "_" + df_sub_min["name"]

    df_sub_min["sic3"] = df_sub_min.sic.apply(lambda x: x[:3]).astype(
        int
    )  # Only considering the 3 first digits.
    df_sub_min["sic2"] = df_sub_min.sic.apply(lambda x: x[:2]).astype(
        int
    )  # Only considering the 2 first digits.
    df_sub_min["sic1"] = df_sub_min.sic.apply(lambda x: x[:1]).astype(
        int
    )  # Only considering the 1 first digits.
    df_sub_min["sicagg"] = df_sub_min.sic2.apply(lambda x: get_sicagg(x))

    df_sub_min = df_sub_min[~df_sub_min.sic1.isin(SIC1_EXCLUDED)]

    df = pd.merge(df_num, df_pre, on=["adsh", "tag", "version"])
    df = pd.merge(df, df_tag, on=["tag", "version"])
    df = pd.merge(df, df_sub_min, on=["adsh"])

    df["year"] = dataset_info["year"]
    df["quarter"] = dataset_info["quarter"]

    df = df[df.uom == "USD"]

    df["tlabel"] = df.tlabel.apply(lambda t: normalize_tlabel(t))
    df["tag_depth"] = df.tlabel.apply(lambda t: len(t.split(",")))

    df["nline_bs"] = (
        df.query("stmt == 'BS'&value!=0").groupby("adsh")["tag"].transform("nunique")
    )
    return df


def merge_all_directories_data(
    root_dir, min_year, max_year, only_balance_sheet_tags=True
):
    """
    Merge all df from all directories in a single dataframe.

    :param root_dir The root directory containy a list of quarterly directory
    :param min_year : First year to consider in the dataset.
    :param max_year : Max year to consider in the datasnet
    :param only_balance_sheet_tags: If true, only consider balance sheets tags.
    :return: A pandas dataframe
    """
    all_directories = list(
        [
            directory
            for directory in glob.glob(f"{root_dir}/**")
            if os.path.isdir(directory)
        ]
    )

    # Only processing data after MIN YEAR for a first run
    all_directories = list(
        filter(
            lambda d: min_year <= int(os.path.basename(d)[:4]) < max_year,
            all_directories,
        )
    )

    print(f"{len(all_directories)} datasets to process")

    njobs = min(MAX_CORE_USAGE, multiprocessing.cpu_count())
    njobs = min(njobs, len(all_directories))
    all_dfs = Parallel(n_jobs=njobs)(
        delayed(extract_quarter_dataset)(directory, only_balance_sheet_tags)
        for directory in tqdm.tqdm(all_directories, "Extracting dataset")
    )

    return pd.concat(all_dfs).reset_index().sort_values(["adsh", "tag"], ascending=True)
