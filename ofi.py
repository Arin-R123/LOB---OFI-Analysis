import pandas as pd
from sklearn.decomposition import PCA
import numpy as np


# OFI CORE LOGIC SHARED ACROSS LEVELS — Implements Eq. (1)

def compute_ofi_columns(df, bid_px, bid_sz, ask_px, ask_sz):
    """
    Computes order flow imbalance columns using Cont et al. Eq. (1):
    OFI = ∑ (OF_bid - OF_ask) where:
    - If price ↑: full size counts
    - If price unchanged: delta size
    - If price ↓: full size counts negatively
    """
    df = df.copy()

    # Track prior values to measure size changes
    df["prev_bid_px"] = df[bid_px].shift(1)
    df["prev_bid_sz"] = df[bid_sz].shift(1)
    df["prev_ask_px"] = df[ask_px].shift(1)
    df["prev_ask_sz"] = df[ask_sz].shift(1)

    # Bid-side contribution to OFI
    df["OF_bid"] = df.apply(lambda row:
        row[bid_sz] if row[bid_px] > row["prev_bid_px"] else
        row[bid_sz] - row["prev_bid_sz"] if row[bid_px] == row["prev_bid_px"] else
        -row[bid_sz], axis=1)

    # Ask-side contribution to OFI
    df["OF_ask"] = df.apply(lambda row:
        -row[ask_sz] if row[ask_px] > row["prev_ask_px"] else
        row[ask_sz] - row["prev_ask_sz"] if row[ask_px] == row["prev_ask_px"] else
        row[ask_sz], axis=1)

    # Final OFI = OF_bid - OF_ask
    df["OFI"] = df["OF_bid"] - df["OF_ask"]

    return df


# OFI at a single level m — Best level = m=0 — Eq. (1)

def compute_ofi(df, m, window="1min"):
    bid_px_col, bid_sz_col = f"bid_px_0{m}", f"bid_sz_0{m}"
    ask_px_col, ask_sz_col = f"ask_px_0{m}", f"ask_sz_0{m}"

    df = compute_ofi_columns(df, bid_px_col, bid_sz_col, ask_px_col, ask_sz_col)
    return df.set_index("ts_event")["OFI"].resample(window).sum()


# Best-level OFI — Paper Level 1 = Code Level 0

def compute_best_level_ofi(df, window="1min"):
    return compute_ofi(df, m=0, window=window)


# Multi-Level Normalized OFI Vector — Eq. (3)
# ofi(h) = (ofi1, ..., ofi10)^T where each ofi_m = OFI_m / QM

def compute_multi_level_ofi(df, window="1min", levels=10):
    df = df.sort_values(by=["depth", "ts_event"]).reset_index(drop=True)

    raw_ofis = []
    depth_sums = []

    for m in range(levels):
        # Compute OFI at level m
        ofi_series = compute_ofi(df.copy(), m=m, window=window)
        raw_ofis.append(ofi_series)

        # Average depth at level m: (bid_sz + ask_sz) / 2
        bid_sz_col = f"bid_sz_0{m}"
        ask_sz_col = f"ask_sz_0{m}"
        df[f"depth_{m}"] = 0.5 * (df[bid_sz_col] + df[ask_sz_col])
        depth_series = df.set_index("ts_event")[f"depth_{m}"].resample(window).mean()
        depth_sums.append(depth_series)

    # OFI matrix (10 columns, one per level)
    ofi_df = pd.concat(raw_ofis, axis=1)
    ofi_df.columns = [f"ofi_{m+1}" for m in range(levels)]

    # Average depth across levels (QM,h^i,t)
    depth_df = pd.concat(depth_sums, axis=1)
    avg_depth = depth_df.mean(axis=1)

    # Normalize per Eq. (3): ofi_m = OFI_m / QM
    normalized_ofi_df = ofi_df.div(avg_depth, axis=0).dropna()
    return normalized_ofi_df


# Integrated OFI via PCA Projection — Eq. (4)
# ofi^I = (w1^T * ofi(h)) / ||w1||_1

def compute_integrated_ofi(multi_level_ofi_df, pca_model=None):
    if pca_model is None:
        pca_model = PCA(n_components=1)
        pca_model.fit(multi_level_ofi_df)

    w1 = pca_model.components_[0]
    w1 /= np.abs(w1).sum()  # Normalize by L1 norm

    integrated_ofi = multi_level_ofi_df.dot(w1)
    integrated_ofi.name = "integrated_ofi"
    return integrated_ofi


# Cross-Asset Return Forecasting using Best-Level OFI
# Approx. Eq. (5): r̂_i,t+1 = β_self OFI_i,t + β_cross OFI_j,t

def forecast_return_best_level_lagged(df_self, df_cross, beta_self=1.0, beta_cross=0.5, lag=1, window="1min"):
    """
    Forecast future returns using lagged best-level OFIs from two assets.
    df_self: LOB of the target asset
    df_cross: LOB of the cross-asset
    """
    # Compute best-level OFIs for both
    ofi_self = compute_best_level_ofi(df_self, window=window)
    ofi_cross = compute_best_level_ofi(df_cross, window=window)

    # Merge and align timestamps
    combined = pd.concat([ofi_self, ofi_cross], axis=1).dropna()
    combined.columns = ["self_ofi", "cross_ofi"]

    # Apply lag to simulate prediction
    combined["self_lag"] = combined["self_ofi"].shift(lag)
    combined["cross_lag"] = combined["cross_ofi"].shift(lag)

    # Linear forecast model: return_t+1 ≈ β_self * OFI_self_t + β_cross * OFI_cross_t
    combined["forecast_return"] = beta_self * combined["self_lag"] + beta_cross * combined["cross_lag"]
    return combined[["forecast_return"]].dropna()


# Cross-Asset Return Forecasting using Integrated OFI (PCA)
# Approx. Eq. (6): r̂_i,t+1 = β_self int_OFI_i,t + β_cross int_OFI_j,t

def forecast_return_integrated_lagged(df_self, df_cross, beta_self=1.0, beta_cross=0.5, lag=1, window="1min", levels=10):
    """
    Forecast future returns using PCA-integrated OFIs from two assets.
    df_self: LOB of the target asset
    df_cross: LOB of the cross-asset
    """
    # Step 1: Compute normalized multi-level OFI vectors
    ofi_self_levels = compute_multi_level_ofi(df_self, window=window, levels=levels)
    ofi_cross_levels = compute_multi_level_ofi(df_cross, window=window, levels=levels)

    # Step 2: Train PCA on self asset OFI, apply to both
    pca = PCA(n_components=1)
    pca.fit(ofi_self_levels)

    int_ofi_self = compute_integrated_ofi(ofi_self_levels, pca_model=pca)
    int_ofi_cross = compute_integrated_ofi(ofi_cross_levels, pca_model=pca)

    # Step 3: Combine and shift
    combined = pd.concat([int_ofi_self, int_ofi_cross], axis=1).dropna()
    combined.columns = ["self_ofi", "cross_ofi"]

    combined["self_lag"] = combined["self_ofi"].shift(lag)
    combined["cross_lag"] = combined["cross_ofi"].shift(lag)

    # Final forecast equation
    combined["forecast_return"] = beta_self * combined["self_lag"] + beta_cross * combined["cross_lag"]
    return combined[["forecast_return"]].dropna()


# Example usage

if __name__ == "__main__":
    df = pd.read_csv("first_25000_rows.csv", parse_dates=["ts_event"])
    df = df.sort_values(by=["depth", "ts_event"]).reset_index(drop=True)

    # Best-level OFI (Level 1 in paper = 0 in code)
    print("Best-level OFI:")
    best_ofi = compute_best_level_ofi(df)
    print(best_ofi.head())

    # Multi-level normalized OFI vector
    print("\nMulti-level OFI vector (normalized):")
    multi_ofi = compute_multi_level_ofi(df)
    print(multi_ofi.head())

    # PCA-integrated OFI from the multi-level vector
    print("\nIntegrated OFI (Eq. 4):")
    integrated_ofi = compute_integrated_ofi(multi_ofi)
    print(integrated_ofi.head())
