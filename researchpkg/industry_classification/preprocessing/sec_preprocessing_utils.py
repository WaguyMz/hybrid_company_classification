import os
from typing import Dict


def distinct_pipe_join(x):
    return "|".join(list(filter(lambda x: x.strip() != "", set(x)))).strip()


def normalize_tlabel(tlabel):
    """
    Normalize a tlabel by lowercasing and replacing "and" by ",".

    :param tlabel: A tlabel string
    :return: A normalized tlabel string
    """

    return tlabel.lower().replace(" and ", ", ")


def get_ith_label(tlabel, i):
    """
    Get the ith label of a tlabel.

    :param tlabel: A tlabel string
    :param i: The index of the label to extract
    """
    all_labels = tlabel.split(",")
    if i > len(all_labels):
        return ""
    else:
        return all_labels[i - 1]


def format_file_name(registrant_name, year, quarter, adsh):
    """
    Format the file name of a submission.

    :param registrant_name: Name of the registrant
    :param year: Year of the submission
    :param quarter: Quarter of the submission
    :param adsh: Adsh of the submission
    :return: A formatted file name
    """
    replacements = [" ", ".", "-", "/", ",", "."]
    for r in replacements:
        registrant_name = registrant_name.replace(r, "_")
    registrant_name = registrant_name.lower()

    return f"{adsh}_{year}_{quarter}.csv"


def save_dataset_config(dataset_dir, **kwargs):
    """
    Save the config of the dataset in a  yaml file.

    :param global_exp_dir:Root directory of the  dataset
    :param min_year: Min year of the dataset
    :param max_year: Max year of the dataset
    :param sic1_used: List of sic1 used (not excluded)
    :param nb_cik: Number of companies in the dataset
    :param max_sub_per_cik: Max number of submission per cik(company)
    :param top_k_tags: Number of top tags to consider
    :param tags_persheet_count_threshold: Threshold of number of appearance of tags to consider
    :param dataset_size: Size of the dataset
    :param train_dataset_size: Size of the train dataset
    :param val_dataset_size: Size of the validation dataset
    """
    config = kwargs
    config["global_exp_dir"] = dataset_dir
    config_file = os.path.join(dataset_dir, "dataset_config.yaml")
    import pyaml

    with open(config_file, "w") as f:
        pyaml.yaml.dump(config, f, default_flow_style=None)


def build_sic_count_index(df_merged, index_dir):
    """
    Build the index of count of each label in the dataset.

    :param df_merged: A pandas dataframe containing all the data of a submission
    :param index_dir: Path to the index directory
    :return: None

    """
    for sic_digit in [1, 2, 3, 4]:
        sic_col = f"sic{sic_digit}"
        count_df = (
            df_merged[["adsh", sic_col]]
            .drop_duplicates()[sic_col]
            .value_counts()
            .to_frame()
        )
        count_df.to_csv(os.path.join(index_dir, f"sic{sic_digit}_count.csv"))


COMMON_PERCENTAGES_15 = [
    "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet",
    "AssetsCurrent",
    "PropertyPlantAndEquipmentNet",
    "IntangibleAssetsNetIncludingGoodwill",
    "Investments",
    "AccountsPayableCurrent",
    "DebtCurrent",
    "DeferredRevenueCurrent",
    "LiabilitiesCurrent",
    "LongTermDebt",
    "Liabilities",
    "PreferredStockValue",
    "CommonStockValue",
    "AccountsReceivableNetCurrent",
]


def compute_common_percentages_15(values_dict: Dict) -> Dict:
    """
    Compute 15 common percentages from the values dict
    """
    assert "Assets" in values_dict, "Assets must be in values_dict"
    # applying rules to compute common percentages
    # Rule1: Calculating the LongTermDebt tag (LongTermDebtCurrent+LongTermDebtNoncurrent)
    if "LongTermDebt" not in values_dict:
        values_dict["LongTermDebt"] = values_dict.get(
            "LongTermDebtCurrent",
            0,
        ) + values_dict.get(
            "LongTermDebtNoncurrent",
            0,
        )

    # Rule2. Calculating the gross profit tag (Revenues-CostOfGoodsAndServicesSold)
    if "GrossProfit" not in values_dict:
        if "Revenues" in values_dict and "CostOfGoodsAndServicesSold" in values_dict:
            values_dict["GrossProfit"] = (
                values_dict["Revenues"] - values_dict["CostOfGoodsAndServicesSold"]
            )

    # Rule3. Compute total capital
    total_capital = (
        values_dict.get(
            "PreferredStockValue",
            0,
        )
        + values_dict.get(
            "CommonStockValue",
            0,
        )
        + values_dict.get(
            "LongTermDebt",
            0,
        )
    )
    values_dict["TotalCapital"] = total_capital

    # Rule 4. Calcaulting investments (ShortTermInvestments+LongTermInvestments)
    values_dict["Investments"] = values_dict.get(
        "ShortTermInvestments",
        0,
    ) + values_dict.get("LongTermInvestments", 0)

    return {
        k: values_dict.get(k, 0) / values_dict["Assets"] for k in COMMON_PERCENTAGES_15
    }


RATIOS_12 = [
    "GrossMargin",
    "RAndDRatio",
    "NetIncomeRatio",
    "DaysOfReceivables",
    "InventoryTurnover",
    "FixedAssetTurnover",
    "TotalAssetTurnover",
    "NetIncomeAssetsRatio",
    "NetIncomeEquityRatio",
    "AssetsEquityRatio",
    "DebtEquityRatio",
    "LongTermDebtCapitalRatio",
]


def computeRatios12(valuesDict: Dict) -> Dict:
    """
    Compute 12 ratios from the values dict
    """
    ratiosDict = {}

    revenues = valuesDict.get("Revenues", 0)
    inventory_net = valuesDict.get("InventoryNet", 0)
    property_plant_and_equipment_net = valuesDict.get("PropertyPlantAndEquipmentNet", 0)
    stockholders_equity = valuesDict.get("StockholdersEquity", 0)
    total_capital = valuesDict.get("TotalCapital", 0)

    # R1. Gross margin

    ratiosDict["GrossMargin"] = (
        (valuesDict.get("GrossProfit", 0) / revenues) if revenues != 0 else 0
    )

    # R2. R&D ratio
    ratiosDict["RAndDRatio"] = (
        (valuesDict.get("ResearchAndDevelopmentExpense", 0) / revenues)
        if revenues != 0
        else 0
    )

    # R3. Net income ratio
    ratiosDict["NetIncomeRatio"] = (
        (valuesDict.get("NetIncomeLoss", 0) / revenues) if revenues != 0 else 0
    )

    # R4. Days of receivables
    ratiosDict["DaysOfReceivables"] = (
        (valuesDict.get("AccountsReceivableNetCurrent", 0) / (revenues / 365))
        if revenues != 0
        else 0
    )

    # R5. Inventory turnover
    ratiosDict["InventoryTurnover"] = (
        (valuesDict.get("CostOfGoodsAndServicesSold", 0) / inventory_net)
        if inventory_net != 0
        else 0
    )

    # R6. Fixed asset turnover
    ratiosDict["FixedAssetTurnover"] = (
        (valuesDict.get("Revenues", 0) / property_plant_and_equipment_net)
        if property_plant_and_equipment_net != 0
        else 0
    )

    # R7. Total asset turnover
    ratiosDict["TotalAssetTurnover"] = (
        valuesDict.get("Revenues", 0) / valuesDict["Assets"]
    )

    # R8. Net income/assets ratio
    ratiosDict["NetIncomeAssetsRatio"] = (
        valuesDict.get("NetIncomeLoss", 0) / valuesDict["Assets"]
    )

    # R9. Net income/equity ratio
    ratiosDict["NetIncomeEquityRatio"] = (
        (valuesDict.get("NetIncomeLoss", 0) / stockholders_equity)
        if stockholders_equity != 0
        else 0
    )

    # R10. Assets/equity ratio
    ratiosDict["AssetsEquityRatio"] = (
        (valuesDict.get("Assets", 0) / stockholders_equity)
        if stockholders_equity != 0
        else 0
    )

    # R11. Debt/equity ratio
    ratiosDict["DebtEquityRatio"] = (
        (valuesDict.get("Liabilities", 0) / stockholders_equity)
        if stockholders_equity != 0
        else 0
    )

    # R12. Long-term debt/capital ratio
    ratiosDict["LongTermDebtCapitalRatio"] = (
        (valuesDict.get("LongTermDebt", 0) / total_capital) if total_capital != 0 else 0
    )

    return ratiosDict
