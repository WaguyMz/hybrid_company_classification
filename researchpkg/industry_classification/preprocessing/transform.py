"""Script to transform any input balanced sheet into a csv with the following structure
account, account_name, change.
"""

import csv
import re

import numpy as np
import pandas as pd

ACCOUNT_COL = "account_num"
ACCOUNT_NAME_COL = "account_name"
NET_CHANGE_COL = "net_change"
ACCOUNT_DESCRIPTION_COL = "description"
ACCOUNT_N_DIGITS = 3  # Every account is represented with exactly that number of digits


def find_delimiter(filename):
    sniffer = csv.Sniffer()
    with open(filename, encoding="latin1") as fp:
        delimiter = sniffer.sniff(fp.read(5000)).delimiter
    return delimiter


def read_df(file: str):
    if file.endswith(".csv"):
        return pd.read_csv(file, sep=find_delimiter(file), encoding="latin1").dropna()
        # Python engine is used to automatically parse the separator.
    elif file.endswith(".xlsx"):
        return pd.read_excel(file).fillna(0)
    else:
        raise Exception("Unsupported file")


def set_account_n_digits(df, to=9):
    """
    Increase the number of digits of df[account] so that all account have the same number of digits "to".
    Zeros are filled at the right to match the target number rof digits.
    """

    df[ACCOUNT_COL] = (
        df[ACCOUNT_COL]
        .astype(str)
        .apply(
            lambda x: (
                x
                if len(x) == to
                else x[:to]
                if len(x) > to
                else x + "0" * (to - len(x))
            )
        )
        .astype("int64")
    )

    # Update accounts according to number of digits and merge corresponding change.
    df = (
        df.groupby(ACCOUNT_COL)
        .agg(
            account_name=(
                ACCOUNT_NAME_COL,
                lambda x: "|".join(
                    list(filter(lambda x: x.strip() != "", set(x)))
                ).strip(),
            ),
            net_change=(NET_CHANGE_COL, np.sum),
        )
        .reset_index()
    )

    return df


def extract_columns_data(df, CANDIDATES, dtype=None):
    """
    Return the first column that match a candidate column according to a type constraint.
    :param df The dataset from where to extract the column
    :param CANDIDATES  List of candidates patterns for the name of the target column.
    :param dtype. Default= None.
    """

    for candidate in CANDIDATES:
        for column in df.columns:
            if candidate.lower() in column.lower():
                return df[column] if dtype is None else df[column].astype(dtype)

    # No Match found
    return None


def get_accounts(df):
    """
    Extact list of accounts from a df
    """
    candidates = ["numéro de compte", "compte", "cpte"]
    accounts = extract_columns_data(df, candidates, dtype="int64")
    return accounts


def get_accounts_names(df):
    """
    Extract list of accounts name from a df
    """
    candidates = ["Libellé", "CpteLib", "Intitul"]
    accounts_names = extract_columns_data(df, CANDIDATES=candidates)
    if accounts_names is not None:
        accounts_names = accounts_names.apply(
            lambda x: re.sub("\d\d{8,}", "", x).lower().strip()
        )
        return accounts_names
    else:
        return [""] * len(df)


def get_changes(df):
    """
    Extract the value of accounts
    """
    candidates = ["net change", "Sum_Total", "Soldes", "Solde"]
    return extract_columns_data(df, CANDIDATES=candidates, dtype="float64")


def load_extract_df(file):
    """
    Read a single input dataframe and return a new dataframe with the following structure.
    account, account_name,  net_change
    Note. account_name can be empty.
    """
    df = read_df(file)
    transformed_df = pd.DataFrame(
        {
            ACCOUNT_COL: get_accounts(df),
            ACCOUNT_NAME_COL: get_accounts_names(df),
            NET_CHANGE_COL: get_changes(df),
        }
    )
    transformed_df = set_account_n_digits(transformed_df, ACCOUNT_N_DIGITS)
    return transformed_df
