import os

import pandas as pd

from researchpkg.industry_classification.config import SEC_ROOT_DATA_DIR



def load_sics_to_naics2_mapping() -> pd.DataFrame:
    """ "
    Load a mapping from sic4 to naics2

    """
    file = os.path.join(SEC_ROOT_DATA_DIR, "sics4_to_naics2.csv")
    return pd.read_csv(
        file,
        index_col=None,
        dtype={"sic4": str, "naics2": str},
        usecols=["sic4", "sic_description", "naics2", "naics2_description"],
    )


def load_sic_codes() -> pd.DataFrame:
    """
    Load sic codes dataframe.
    """

    file = os.path.join(SEC_ROOT_DATA_DIR, "sic_codes.csv")
    return pd.read_csv(file, index_col=None, dtype={"sic": int})
