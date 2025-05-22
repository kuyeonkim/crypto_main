import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler

# ----------------------------------------------------------------------
# Data paths
# ----------------------------------------------------------------------

VOLCORR_PATH = Path("data/volcorrultimate.csv")
TOTAL_DF_PATH = Path("data/total_df_tier_mean_spread.csv")
OTHER_DATA_PATH = Path("data/other_dataset.csv")


# ----------------------------------------------------------------------
# volcorrultimate.csv preprocessing
# ----------------------------------------------------------------------

def load_volcorr(path: Path = VOLCORR_PATH) -> pd.DataFrame:
    """Load the volcorrultimate dataset."""
    return pd.read_csv(path)


def preprocess_volcorr(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess volcorrultimate data.

    - Drop missing rows
    - Replace zero p-values with the smallest positive value
    - Apply -log10 transformation to p-values
    - Standard scale beta columns and robust scale p-value columns
    """
    df = df.dropna().copy()

    pval_cols = ["abs_ret_pval", "pos_pval", "interact_pval"]
    for col in pval_cols:
        m = df.loc[df[col] > 0, col].min()
        df.loc[df[col] == 0, col] = m
        df[col + "_transformed"] = -np.log10(df[col])

    beta = df[["beta1", "beta2", "beta3"]].values
    pvals = df[[c + "_transformed" for c in pval_cols]].values

    beta_scaled = StandardScaler().fit_transform(beta)
    pval_scaled = RobustScaler().fit_transform(pvals)

    X = np.hstack([beta_scaled, pval_scaled])
    return pd.DataFrame(X, columns=["beta1", "beta2", "beta3"] + [c + "_t" for c in pval_cols])


# ----------------------------------------------------------------------
# total_df_tier_mean_spread.csv preprocessing
# ----------------------------------------------------------------------

def load_total_df(path: Path = TOTAL_DF_PATH) -> pd.DataFrame:
    """Load the market cap correlation dataset."""
    return pd.read_csv(path)


def preprocess_total_df(df: pd.DataFrame, scalemethod: str = "standard") -> pd.DataFrame:
    """Preprocess the total_df dataset.

    This mirrors the transformations from the notebook and keeps the
    column structure intact so that later steps can consume the output.
    """
    cap_list = ["Large", "Medium", "Small", "Micro"]

    pairs = []
    for i in range(len(cap_list)):
        for j in range(i + 1, len(cap_list)):
            pairs.append((cap_list[i], cap_list[j]))

    mean_cols = [f"{cap}_mean" for cap in cap_list]
    sd_cols = [f"{cap}_sd" for cap in cap_list]
    diff_cols = [f"{c1}_{c2}_diff" for c1, c2 in pairs]
    corr_cols = [f"{c1}_{c2}_corr" for c1, c2 in pairs]

    scaler = StandardScaler() if scalemethod == "standard" else RobustScaler()
    signed_log = lambda x: np.sign(x) * np.log1p(np.abs(x))

    cap_mean_df = df[["exchange"] + mean_cols].copy().set_index("exchange")
    cap_sd_df = df[["exchange"] + sd_cols].copy().set_index("exchange")
    cap_diff_df = df[["exchange"] + diff_cols].copy().set_index("exchange")
    cap_corr_df = df[["exchange"] + corr_cols].copy().set_index("exchange")

    cap_mean_df = pd.DataFrame(
        scaler.fit_transform(signed_log(cap_mean_df)),
        columns=cap_mean_df.columns,
        index=cap_mean_df.index,
    )
    cap_sd_df = pd.DataFrame(
        scaler.fit_transform(signed_log(cap_sd_df)),
        columns=cap_sd_df.columns,
        index=cap_sd_df.index,
    )
    cap_diff_df = pd.DataFrame(
        scaler.fit_transform(signed_log(cap_diff_df)),
        columns=cap_diff_df.columns,
        index=cap_diff_df.index,
    )

    cap_mean_df = cap_mean_df.dropna(axis=0)
    cap_sd_df = cap_sd_df.dropna(axis=0)
    cap_diff_df = cap_diff_df.dropna(axis=0)
    cap_corr_df = cap_corr_df.dropna(axis=0)

    processed = pd.concat([cap_mean_df, cap_sd_df, cap_corr_df], axis=1)
    processed = processed.dropna()
    return processed


# ----------------------------------------------------------------------
# placeholder for the third dataset
# ----------------------------------------------------------------------

def load_other(path: Path = OTHER_DATA_PATH) -> pd.DataFrame:
    """Load the third dataset (placeholder)."""
    return pd.read_csv(path)


def preprocess_other(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder preprocessing for the third dataset."""
    return df.dropna()


