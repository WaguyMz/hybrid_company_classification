""""
Script to build the referentials of sics.
"""

import os

import pandas as pd

from researchpkg.industry_classification.config import RAW_SIC_FILE
from researchpkg.industry_classification.preprocessing.sec_preprocessing_utils import (
    distinct_pipe_join,
)


def build_sic_referentials(base_sic_file: str):
    """
    Build mapping form sic numbers to Industry Title
    """

    df = pd.read_csv(base_sic_file, sep="|")
    df.rename(
        columns={"SIC Code": "sic", "Industry Title": "industry_title"}, inplace=True
    )
    df = df.astype({"sic": "int"})

    df["sic4"] = df.sic
    df["sic3"] = df.sic.apply(
        lambda x: int(float(x) // 10)
    )  # Only considering the 3 first digits.
    df["sic2"] = df.sic.apply(
        lambda x: int(float(x) // 100)
    )  # Only considering the 2 first digits.
    df["sic1"] = df.sic.apply(
        lambda x: int(float(x) // 1000)
    )  # Only consideering hte fist digit

    for i in range(1, 5):
        df_sic = df[[f"sic{i}", "industry_title"]]

        df_sic = (
            df_sic.groupby(f"sic{i}")
            .aggregate(
                industry_title=("industry_title", distinct_pipe_join),
            )
            .reset_index()
        )

        file = os.path.join(os.path.dirname(base_sic_file), f"sics{i}.csv")
        df_sic.to_csv(file, index=False)

        print(f"Sic {i} successfully saved to {file}")


if __name__ == "__main__":
    build_sic_referentials(RAW_SIC_FILE)
