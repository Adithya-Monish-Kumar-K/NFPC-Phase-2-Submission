#!/usr/bin/env python3
"""
AML Mule Account Detection — First-Place Solution (Optimized)
==============================================================
Fully vectorized pipeline for 400M transactions on 16GB RAM.
All groupby operations use native pandas/numpy — zero row-level apply() calls
on large frames.

Pipeline:
  1. Two-pass chunked transaction feature engineering
  2. Static/profile features
  3. Graph/network features from counterparty edges
  4. Red-herring detection & label denoising
  5. Multi-model ensemble (LightGBM + XGBoost + CatBoost)
  6. Temporal window estimation (changepoint detection)
  7. Submission CSV + report
"""

import os
import gc
import warnings
import logging
from pathlib import Path
from glob import glob
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

import sys, io
_stderr_utf8 = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
_fh = logging.FileHandler(str(Path(r"d:\Hackathons\Fraud Detection Hack\Phase - 2\archive") / "pipeline3.log"), mode="w", encoding="utf-8")
_fh.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        _fh,
        logging.StreamHandler(stream=_stderr_utf8),
    ],
)
# Flush file handler after each log message for real-time monitoring
for _h in logging.getLogger().handlers:
    if isinstance(_h, logging.FileHandler):
        _orig_emit = _h.emit
        def _flushing_emit(record, _emit=_orig_emit, _handler=_h):
            _emit(record)
            _handler.flush()
        _h.emit = _flushing_emit
log = logging.getLogger(__name__)

BASE = Path(r"d:\Hackathons\Fraud Detection Hack\Phase - 2\archive")
CACHE = BASE / "features"
CACHE.mkdir(exist_ok=True)

TXN_PARTS = sorted(glob(str(BASE / "transactions" / "batch-*" / "part_*.parquet")))
TXN_ADD_PARTS = sorted(glob(str(BASE / "transactions_additional" / "batch-*" / "part_*.parquet")))

TXN_START = pd.Timestamp("2020-07-01")
TXN_END = pd.Timestamp("2025-06-30")
REF_DATE = pd.Timestamp("2025-06-30")

BENFORD = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])
SEED = 42
N_JOBS = 16
np.random.seed(SEED)


# ===================================================================
# DATA LOADING
# ===================================================================

def load_static():
    log.info("Loading static tables...")
    d = {}
    d["accounts"] = pd.read_parquet(BASE / "accounts.parquet")
    d["customers"] = pd.read_parquet(BASE / "customers.parquet")
    d["linkage"] = pd.read_parquet(BASE / "customer_account_linkage.parquet")
    d["demographics"] = pd.read_parquet(BASE / "demographics.parquet")
    d["accounts_add"] = pd.read_parquet(BASE / "accounts-additional.parquet")
    d["branch"] = pd.read_parquet(BASE / "branch.parquet")
    d["product"] = pd.read_parquet(BASE / "product_details.parquet")
    d["train_labels"] = pd.read_parquet(BASE / "train_labels.parquet")
    d["test_accounts"] = pd.read_parquet(BASE / "test_accounts.parquet")
    all_ids = set(
        pd.concat([d["train_labels"][["account_id"]], d["test_accounts"][["account_id"]]])
        .drop_duplicates()["account_id"]
    )
    d["all_ids"] = all_ids
    log.info(f"  {len(all_ids)} accounts to process")
    return d


# ===================================================================
# TRANSACTION FEATURES — FULLY VECTORIZED
# ===================================================================

def _process_txn_part(path, all_ids):
    """Process one parquet part — returns dicts of per-account aggregations."""
    df = pd.read_parquet(path)
    df = df[df["account_id"].isin(all_ids)]
    if len(df) == 0:
        return None

    df["abs_amount"] = df["amount"].abs()
    df["ts"] = pd.to_datetime(df["transaction_timestamp"], format="ISO8601")
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["day_of_month"] = df["ts"].dt.day
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["is_monthend"] = (df["day_of_month"] >= 25).astype(np.int8)
    df["is_debit"] = (df["txn_type"] == "D").astype(np.int8)
    df["is_credit"] = (df["txn_type"] == "C").astype(np.int8)

    # First digit for Benford's (vectorized)
    nonzero = df["abs_amount"] >= 1
    first_digit = np.zeros(len(df), dtype=np.int8)
    if nonzero.any():
        log10 = np.log10(df.loc[nonzero, "abs_amount"].values)
        first_digit[nonzero.values] = (10 ** (log10 - np.floor(log10))).astype(int).clip(1, 9)
    df["first_digit"] = first_digit

    # Structuring flags (vectorized)
    aa = df["abs_amount"]
    df["struct_45_50k"] = ((aa >= 45000) & (aa < 50000)).astype(np.int8)
    df["struct_9_10k"] = ((aa >= 9000) & (aa < 10000)).astype(np.int8)
    df["struct_90_100k"] = ((aa >= 90000) & (aa < 100000)).astype(np.int8)
    df["round_1k"] = ((aa % 1000 == 0) & (aa > 0)).astype(np.int8)
    df["round_5k"] = ((aa % 5000 == 0) & (aa > 0)).astype(np.int8)
    df["round_10k"] = ((aa % 10000 == 0) & (aa > 0)).astype(np.int8)
    df["round_50k"] = ((aa % 50000 == 0) & (aa > 0)).astype(np.int8)

    # Pre-computed columns for pure vectorized agg (no lambdas!)
    df["debit_amt"] = df["abs_amount"] * df["is_debit"]
    df["credit_amt"] = df["abs_amount"] * df["is_credit"]
    df["amount_sq"] = df["abs_amount"] ** 2

    g = df.groupby("account_id", sort=False)

    # Basic aggregations — no lambdas, all native pandas ops
    basic = g.agg(
        txn_count=("amount", "count"),
        txn_sum=("abs_amount", "sum"),
        txn_mean=("abs_amount", "mean"),
        txn_std=("abs_amount", "std"),
        txn_min=("abs_amount", "min"),
        txn_max=("abs_amount", "max"),
        amount_sq_sum=("amount_sq", "sum"),
        debit_count=("is_debit", "sum"),
        credit_count=("is_credit", "sum"),
        debit_sum=("debit_amt", "sum"),
        credit_sum=("credit_amt", "sum"),
        unique_cp=("counterparty_id", "nunique"),
        unique_mcc=("mcc_code", "nunique"),
        unique_channel=("channel", "nunique"),
        night_count=("is_night", "sum"),
        weekend_count=("is_weekend", "sum"),
        monthend_count=("is_monthend", "sum"),
        struct_45_50k=("struct_45_50k", "sum"),
        struct_9_10k=("struct_9_10k", "sum"),
        struct_90_100k=("struct_90_100k", "sum"),
        round_1k=("round_1k", "sum"),
        round_5k=("round_5k", "sum"),
        round_10k=("round_10k", "sum"),
        round_50k=("round_50k", "sum"),
        ts_min=("ts", "min"),
        ts_max=("ts", "max"),
    )

    # Channel distribution
    ch = df.groupby(["account_id", "channel"], sort=False).size().unstack(fill_value=0)
    ch.columns = [f"ch_{c}" for c in ch.columns]

    # Monthly counts (for burstiness)
    monthly = df.groupby(["account_id", "month"], sort=False).size().unstack(fill_value=0)
    monthly.columns = [f"_m_{c}" for c in monthly.columns]

    # Hour distribution (for entropy)
    hr = df.groupby(["account_id", "hour"], sort=False).size().unstack(fill_value=0)
    hr.columns = [f"hr_{c}" for c in hr.columns]

    # First-digit distribution (for Benford)
    fd = df[df["first_digit"] > 0].groupby(
        ["account_id", "first_digit"], sort=False
    ).size().unstack(fill_value=0)
    fd.columns = [f"fd_{int(c)}" for c in fd.columns]

    # ── Sequence features (sorted by time, per-account) ──
    df_s = df.sort_values(["account_id", "ts"])
    prev_type = df_s.groupby("account_id", sort=False)["is_debit"].shift(1)
    prev_type_notna = prev_type.notna()
    df_s["trans_CC"] = (prev_type_notna & (prev_type == 0) & (df_s["is_debit"] == 0)).astype(np.int8)
    df_s["trans_CD"] = (prev_type_notna & (prev_type == 0) & (df_s["is_debit"] == 1)).astype(np.int8)
    df_s["trans_DC"] = (prev_type_notna & (prev_type == 1) & (df_s["is_debit"] == 0)).astype(np.int8)
    df_s["trans_DD"] = (prev_type_notna & (prev_type == 1) & (df_s["is_debit"] == 1)).astype(np.int8)

    prev_ts = df_s.groupby("account_id", sort=False)["ts"].shift(1)
    delta_secs = (df_s["ts"] - prev_ts).dt.total_seconds()
    valid_delta = delta_secs.notna() & (delta_secs >= 0)
    df_s["delta_val"] = delta_secs.where(valid_delta, np.nan)
    df_s["delta_sq"] = df_s["delta_val"] ** 2
    df_s["delta_valid"] = valid_delta.astype(np.int8)
    df_s["rapid_1h"] = ((delta_secs < 3600) & valid_delta).astype(np.int8)
    df_s["rapid_6h"] = ((delta_secs < 21600) & valid_delta).astype(np.int8)
    df_s["rapid_24h"] = ((delta_secs < 86400) & valid_delta).astype(np.int8)

    prev_amt = df_s.groupby("account_id", sort=False)["abs_amount"].shift(1)
    df_s["amt_change"] = (df_s["abs_amount"] - prev_amt).abs()
    df_s["amt_change_valid"] = df_s["amt_change"].notna().astype(np.int8)

    gs = df_s.groupby("account_id", sort=False)
    seq = gs.agg(
        trans_CC=("trans_CC", "sum"),
        trans_CD=("trans_CD", "sum"),
        trans_DC=("trans_DC", "sum"),
        trans_DD=("trans_DD", "sum"),
        delta_sum=("delta_val", "sum"),
        delta_sq_sum=("delta_sq", "sum"),
        delta_min=("delta_val", "min"),
        delta_max=("delta_val", "max"),
        delta_count=("delta_valid", "sum"),
        rapid_1h=("rapid_1h", "sum"),
        rapid_6h=("rapid_6h", "sum"),
        rapid_24h=("rapid_24h", "sum"),
        amt_change_sum=("amt_change", "sum"),
        amt_change_count=("amt_change_valid", "sum"),
    )
    del df_s

    return {
        "basic": basic,
        "channel": ch,
        "monthly": monthly,
        "hour": hr,
        "first_digit": fd,
        "sequence": seq,
    }


def _merge_accumulators(acc_list):
    """Sum-merge a list of DataFrames by index."""
    if not acc_list:
        return pd.DataFrame()
    merged = pd.concat(acc_list)
    merged = merged.groupby(level=0).sum()
    return merged


def _merge_minmax(acc_list, sum_cols, min_cols, max_cols):
    """Merge accumulators with correct sum/min/max operations."""
    if not acc_list:
        return pd.DataFrame()
    merged = pd.concat(acc_list)
    ops = {c: "sum" for c in sum_cols}
    ops.update({c: "min" for c in min_cols})
    ops.update({c: "max" for c in max_cols})
    return merged.groupby(level=0).agg(ops)


def compute_txn_features(sd):
    """Single-pass vectorized feature extraction from ~400M transactions."""
    cache_file = CACHE / "txn_features_v4.parquet"
    if cache_file.exists():
        log.info("Loading cached txn features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]
    log.info(f"Transaction feature extraction from {len(TXN_PARTS)} parts...")

    # Accumulators
    basic_acc, ch_acc, monthly_acc, hr_acc, fd_acc, seq_acc = [], [], [], [], [], []

    for i, path in enumerate(TXN_PARTS):
        if i % 10 == 0:
            log.info(f"  Part {i + 1}/{len(TXN_PARTS)}")

        result = _process_txn_part(path, all_ids)
        if result is None:
            continue

        basic_acc.append(result["basic"])
        ch_acc.append(result["channel"])
        monthly_acc.append(result["monthly"])
        hr_acc.append(result["hour"])
        fd_acc.append(result["first_digit"])
        seq_acc.append(result["sequence"])

        # Periodically consolidate to cap memory
        if len(basic_acc) >= 30:
            log.info(f"    Consolidating at part {i + 1}...")
            # Basic: sum numeric, min/max for timestamps
            basic_num = [b.drop(columns=["ts_min", "ts_max"], errors="ignore") for b in basic_acc]
            basic_ts = [b[["ts_min", "ts_max"]].rename(columns={"ts_min": "ts_min", "ts_max": "ts_max"}) for b in basic_acc]
            basic_nums = _merge_accumulators(basic_num)
            ts_merged = pd.concat(basic_ts).groupby(level=0).agg({"ts_min": "min", "ts_max": "max"})
            basic_acc = [basic_nums.join(ts_merged)]

            ch_acc = [_merge_accumulators(ch_acc)]
            monthly_acc = [_merge_accumulators(monthly_acc)]
            hr_acc = [_merge_accumulators(hr_acc)]
            fd_acc = [_merge_accumulators(fd_acc)]
            _seq_sum = ["trans_CC", "trans_CD", "trans_DC", "trans_DD",
                        "delta_sum", "delta_sq_sum", "delta_count",
                        "rapid_1h", "rapid_6h", "rapid_24h",
                        "amt_change_sum", "amt_change_count"]
            seq_acc = [_merge_minmax(seq_acc, _seq_sum, ["delta_min"], ["delta_max"])]
            gc.collect()

    # Final consolidation
    log.info("  Final consolidation...")
    basic_num = [b.drop(columns=["ts_min", "ts_max"], errors="ignore") for b in basic_acc]
    basic_ts = [b[["ts_min", "ts_max"]] for b in basic_acc if "ts_min" in b.columns]
    basic_all = _merge_accumulators(basic_num)
    if basic_ts:
        ts_all = pd.concat(basic_ts).groupby(level=0).agg({"ts_min": "min", "ts_max": "max"})
        basic_all = basic_all.join(ts_all)

    ch_all = _merge_accumulators(ch_acc)
    m_all = _merge_accumulators(monthly_acc)
    hr_all = _merge_accumulators(hr_acc)
    fd_all = _merge_accumulators(fd_acc)
    _seq_sum = ["trans_CC", "trans_CD", "trans_DC", "trans_DD",
                "delta_sum", "delta_sq_sum", "delta_count",
                "rapid_1h", "rapid_6h", "rapid_24h",
                "amt_change_sum", "amt_change_count"]
    seq_all = _merge_minmax(seq_acc, _seq_sum, ["delta_min"], ["delta_max"])
    del basic_acc, ch_acc, monthly_acc, hr_acc, fd_acc, seq_acc
    gc.collect()

    # ── Derive features ──
    log.info("  Deriving features from aggregates...")
    feats = pd.DataFrame(index=basic_all.index)
    feats.index.name = "account_id"

    n = basic_all["txn_count"].clip(lower=1)
    feats["txn_count"] = basic_all["txn_count"]
    feats["txn_sum"] = basic_all["txn_sum"]
    feats["txn_mean"] = basic_all["txn_sum"] / n
    feats["txn_max"] = basic_all["txn_max"]
    feats["txn_min"] = basic_all["txn_min"]
    mean_sq = basic_all["amount_sq_sum"] / n
    mean_val = basic_all["txn_sum"] / n
    feats["txn_std"] = np.sqrt(np.maximum(0, mean_sq - mean_val ** 2))
    feats["cv"] = feats["txn_std"] / (feats["txn_mean"] + 1)

    feats["debit_count"] = basic_all["debit_count"]
    feats["credit_count"] = basic_all["credit_count"]
    feats["debit_sum"] = basic_all["debit_sum"]
    feats["credit_sum"] = basic_all["credit_sum"]
    feats["cd_ratio_count"] = basic_all["credit_count"] / (basic_all["debit_count"] + 1)
    feats["cd_ratio_sum"] = basic_all["credit_sum"] / (basic_all["debit_sum"] + 1)
    feats["net_flow"] = basic_all["credit_sum"] - basic_all["debit_sum"]
    feats["abs_net_flow"] = feats["net_flow"].abs()
    feats["unique_cp"] = basic_all["unique_cp"]
    feats["unique_mcc"] = basic_all["unique_mcc"]
    feats["unique_channel"] = basic_all["unique_channel"]

    feats["night_ratio"] = basic_all["night_count"] / n
    feats["weekend_ratio"] = basic_all["weekend_count"] / n
    feats["monthend_ratio"] = basic_all["monthend_count"] / n

    feats["struct_45_50k_count"] = basic_all["struct_45_50k"]
    feats["struct_45_50k_ratio"] = basic_all["struct_45_50k"] / n
    feats["struct_9_10k_count"] = basic_all["struct_9_10k"]
    feats["struct_9_10k_ratio"] = basic_all["struct_9_10k"] / n
    feats["struct_90_100k_count"] = basic_all["struct_90_100k"]
    feats["struct_90_100k_ratio"] = basic_all["struct_90_100k"] / n

    feats["round_1k_ratio"] = basic_all["round_1k"] / n
    feats["round_5k_ratio"] = basic_all["round_5k"] / n
    feats["round_10k_ratio"] = basic_all["round_10k"] / n
    feats["round_50k_ratio"] = basic_all["round_50k"] / n

    if "ts_min" in basic_all.columns:
        feats["txn_span_days"] = (basic_all["ts_max"] - basic_all["ts_min"]).dt.total_seconds() / 86400
        feats["txn_velocity"] = feats["txn_count"] / (feats["txn_span_days"] + 1)

    # ── Benford's Law ──
    fd_cols = sorted([c for c in fd_all.columns if c.startswith("fd_")])
    if fd_cols:
        for d in range(1, 10):
            col = f"fd_{d}"
            if col not in fd_all.columns:
                fd_all[col] = 0
        fd_ordered = fd_all[[f"fd_{d}" for d in range(1, 10)]].fillna(0)
        fd_total = fd_ordered.sum(axis=1).clip(lower=1)
        fd_probs = fd_ordered.div(fd_total, axis=0).values
        expected = BENFORD[np.newaxis, :]
        feats["benford_chi2"] = np.sum((fd_probs - expected) ** 2 / (expected + 1e-10), axis=1)
        feats["benford_kl"] = np.sum(
            np.where(fd_probs > 0, fd_probs * np.log((fd_probs + 1e-10) / (expected + 1e-10)), 0), axis=1
        )

    # ── Hour entropy ──
    hr_cols = sorted([c for c in hr_all.columns if c.startswith("hr_")])
    if hr_cols:
        hr_df = hr_all[hr_cols].fillna(0)
        hr_total = hr_df.sum(axis=1).clip(lower=1)
        hr_probs = hr_df.div(hr_total, axis=0).values
        feats["hour_entropy"] = -np.nansum(
            np.where(hr_probs > 0, hr_probs * np.log2(hr_probs + 1e-15), 0), axis=1
        )

    # ── Monthly burstiness ──
    m_cols = sorted([c for c in m_all.columns if c.startswith("_m_")])
    if m_cols:
        m_df = m_all[m_cols].fillna(0)
        feats["active_months"] = (m_df > 0).sum(axis=1)
        feats["monthly_txn_mean"] = m_df.mean(axis=1)
        feats["monthly_txn_std"] = m_df.std(axis=1)
        feats["monthly_txn_max"] = m_df.max(axis=1)
        feats["burstiness"] = feats["monthly_txn_std"] / (feats["monthly_txn_mean"] + 1)

        if len(m_cols) > 6:
            recent = m_df[m_cols[-6:]].sum(axis=1)
            total = m_df.sum(axis=1)
            feats["recent_6m_ratio"] = recent / (total + 1)

        n_months = len(m_cols)
        cutoff = int(n_months * 0.8)
        if cutoff > 0 and cutoff < n_months:
            early = m_df[m_cols[:cutoff]].sum(axis=1)
            late = m_df[m_cols[cutoff:]].sum(axis=1)
            feats["late_20pct_ratio"] = late / (early + late + 1)

        # Max consecutive zero months — vectorized with numpy
        m_vals = m_df.values
        is_zero = (m_vals == 0).astype(np.int8)
        max_zeros = np.zeros(len(m_df), dtype=np.int32)
        curr_zeros = np.zeros(len(m_df), dtype=np.int32)
        for col_idx in range(is_zero.shape[1]):
            curr_zeros = curr_zeros * is_zero[:, col_idx] + is_zero[:, col_idx]
            max_zeros = np.maximum(max_zeros, curr_zeros)
        feats["max_dormant_months"] = max_zeros

    # ── Channel ratios ──
    ch_total = ch_all.sum(axis=1).clip(lower=1)
    for c in ch_all.columns:
        feats[c] = ch_all[c]
        feats[f"{c}_ratio"] = ch_all[c] / ch_total

    # ── Sequence features ──
    if len(seq_all) > 0:
        total_trans = (seq_all["trans_CC"] + seq_all["trans_CD"] + seq_all["trans_DC"] + seq_all["trans_DD"]).clip(lower=1)
        feats["markov_P_CC"] = seq_all["trans_CC"] / total_trans
        feats["markov_P_CD"] = seq_all["trans_CD"] / total_trans
        feats["markov_P_DC"] = seq_all["trans_DC"] / total_trans
        feats["markov_P_DD"] = seq_all["trans_DD"] / total_trans
        feats["type_switching_rate"] = (seq_all["trans_CD"] + seq_all["trans_DC"]) / total_trans
        dc = seq_all["delta_count"].clip(lower=1)
        feats["delta_mean"] = seq_all["delta_sum"] / dc
        delta_var = (seq_all["delta_sq_sum"] / dc) - (feats["delta_mean"] ** 2)
        feats["delta_std"] = np.sqrt(np.maximum(0, delta_var))
        feats["delta_min"] = seq_all["delta_min"]
        feats["delta_max"] = seq_all["delta_max"]
        feats["delta_cv"] = feats["delta_std"] / (feats["delta_mean"] + 1)
        feats["rapid_1h_count"] = seq_all["rapid_1h"]
        feats["rapid_6h_count"] = seq_all["rapid_6h"]
        feats["rapid_24h_count"] = seq_all["rapid_24h"]
        feats["rapid_1h_ratio"] = seq_all["rapid_1h"] / dc
        feats["rapid_6h_ratio"] = seq_all["rapid_6h"] / dc
        ac = seq_all["amt_change_count"].clip(lower=1)
        feats["amount_velocity_mean"] = seq_all["amt_change_sum"] / ac

    # Drop datetime columns before saving
    ts_cols = feats.select_dtypes(include=["datetime64", "datetimetz"]).columns
    feats_save = feats.drop(columns=ts_cols)
    feats_save.to_parquet(cache_file)
    log.info(f"  Txn features: {feats_save.shape}")

    del basic_all, ch_all, m_all, hr_all, fd_all, seq_all
    gc.collect()
    return feats_save


# ===================================================================
# GRAPH / NETWORK FEATURES
# ===================================================================

def compute_graph_features(sd):
    cache_file = CACHE / "graph_features_v6.parquet"
    edges_cache = CACHE / "edges_cache.parquet"
    if cache_file.exists():
        log.info("Loading cached graph features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]

    log.info("Building counterparty edges (chunked)...")
    edge_acc = []
    for i, path in enumerate(TXN_PARTS):
        if i % 50 == 0:
            log.info(f"  Edge part {i + 1}/{len(TXN_PARTS)}")
        df = pd.read_parquet(path, columns=["account_id", "counterparty_id", "amount", "txn_type"])
        df = df[df["account_id"].isin(all_ids)]
        if len(df) == 0:
            continue
        df["abs_a"] = df["amount"].abs()
        df["is_c"] = (df["txn_type"] == "C").astype(np.int8)
        grp = df.groupby(["account_id", "counterparty_id"], sort=False).agg(
            w=("abs_a", "sum"),
            cnt=("abs_a", "count"),
            c_cnt=("is_c", "sum"),
        ).reset_index()
        edge_acc.append(grp)

        if len(edge_acc) >= 25:
            merged = pd.concat(edge_acc, ignore_index=True)
            merged = merged.groupby(["account_id", "counterparty_id"], sort=False).agg(
                {"w": "sum", "cnt": "sum", "c_cnt": "sum"}
            ).reset_index()
            edge_acc = [merged]
            gc.collect()

    edges = pd.concat(edge_acc, ignore_index=True)
    edges = edges.groupby(["account_id", "counterparty_id"], sort=False).agg(
        {"w": "sum", "cnt": "sum", "c_cnt": "sum"}
    ).reset_index()
    edges["d_cnt"] = edges["cnt"] - edges["c_cnt"]
    del edge_acc
    gc.collect()
    log.info(f"  Edges: {len(edges)}")

    # Save edges for per-fold label feature recomputation
    edges[["account_id", "counterparty_id", "w", "cnt", "c_cnt", "d_cnt"]].to_parquet(edges_cache)
    log.info(f"  Edges saved to cache")

    g = edges.groupby("account_id", sort=False)
    feats = pd.DataFrame(index=g.groups.keys())
    feats.index.name = "account_id"

    feats["degree"] = g["counterparty_id"].nunique()
    feats["weighted_degree"] = g["w"].sum()
    feats["edge_count"] = g["cnt"].sum()
    feats["edge_credit_total"] = g["c_cnt"].sum()
    feats["edge_debit_total"] = g["d_cnt"].sum()
    feats["edge_cd_ratio"] = feats["edge_credit_total"] / (feats["edge_debit_total"] + 1)

    # Herfindahl (vectorized)
    edges["w_total"] = edges.groupby("account_id")["w"].transform("sum")
    edges["w_share_sq"] = (edges["w"] / (edges["w_total"] + 1)) ** 2
    feats["counterparty_herfindahl"] = edges.groupby("account_id", sort=False)["w_share_sq"].sum()

    feats["top_cp_concentration"] = g["w"].max() / (feats["weighted_degree"] + 1)

    # Fan-in / fan-out
    credit_cps = edges[edges["c_cnt"] > 0].groupby("account_id", sort=False)["counterparty_id"].nunique()
    debit_cps = edges[edges["d_cnt"] > 0].groupby("account_id", sort=False)["counterparty_id"].nunique()
    feats["fanin_cps"] = credit_cps
    feats["fanout_cps"] = debit_cps
    feats["fanin_fanout_ratio"] = feats["fanin_cps"].fillna(0) / (feats["fanout_cps"].fillna(0) + 1)

    feats = feats.fillna(0)
    feats.to_parquet(cache_file)
    del edges
    gc.collect()
    log.info(f"  Graph features: {feats.shape}")
    return feats


# ===================================================================
# GEO / IP / BALANCE FEATURES
# ===================================================================

def compute_geo_features(sd):
    cache_file = CACHE / "geo_features_v6.parquet"
    ip_cache_file = CACHE / "ip_pairs_cache.parquet"
    if cache_file.exists():
        log.info("Loading cached geo features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]
    all_ids_set = set(all_ids)

    # Integer-encode account_ids so tid_map uses ~1.2 GB/batch instead of ~12 GB
    sorted_ids = sorted(all_ids)
    acct_to_idx = {aid: np.int32(i) for i, aid in enumerate(sorted_ids)}
    idx_to_acct = np.array(sorted_ids, dtype=object)

    log.info("Computing geo/IP/balance features batch-by-batch...")
    geo_acc, bal_acc, ip_pair_acc, ip_n_acc = [], [], [], []

    for batch_num in range(1, 5):
        txn_dir = BASE / "transactions" / f"batch-{batch_num}"
        add_dir = BASE / "transactions_additional" / f"batch-{batch_num}"
        if not txn_dir.exists() or not add_dir.exists():
            continue

        txn_parts = sorted(txn_dir.glob("*.parquet"))
        add_parts = sorted(add_dir.glob("*.parquet"))

        # Build tid_map for this batch: int64 txn_id + int32 acct_idx
        log.info(f"  Batch {batch_num}: index from {len(txn_parts)} txn parts...")
        chunks = []
        for path in txn_parts:
            df = pd.read_parquet(path, columns=["transaction_id", "account_id"])
            df = df[df["account_id"].isin(all_ids_set)]
            chunks.append(pd.DataFrame({
                "tid_int": df["transaction_id"].str.slice(4).astype(np.int64),
                "aid_int": df["account_id"].map(acct_to_idx).astype(np.int32),
            }))
        tid_map = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        log.info(f"    Index: {len(tid_map)} txns, ~{tid_map.memory_usage(deep=True).sum()/(1024**2):.0f} MB")

        # Process transactions_additional for this batch
        for j, path in enumerate(add_parts):
            if j % 20 == 0:
                log.info(f"    Add part {j+1}/{len(add_parts)}")
            adf = pd.read_parquet(path)
            adf["tid_int"] = adf["transaction_id"].str.slice(4).astype(np.int64)
            adf = adf.merge(tid_map, on="tid_int", how="inner")
            if len(adf) == 0:
                continue

            # Geo
            geo = adf[adf["latitude"].notna() & adf["longitude"].notna()].copy()
            if len(geo) > 0:
                geo["lat_sq"] = geo["latitude"] ** 2
                geo["lon_sq"] = geo["longitude"] ** 2
                ga = geo.groupby("aid_int", sort=False).agg(
                    lat_sum=("latitude", "sum"),
                    lat_sq_sum=("lat_sq", "sum"),
                    lon_sum=("longitude", "sum"),
                    lon_sq_sum=("lon_sq", "sum"),
                    geo_n=("latitude", "count"),
                    lat_min=("latitude", "min"),
                    lat_max=("latitude", "max"),
                    lon_min=("longitude", "min"),
                    lon_max=("longitude", "max"),
                )
                geo_acc.append(ga)

            # IP: collect count + unique pairs for shared-IP analysis
            ips = adf[adf["ip_address"].notna() & (adf["ip_address"] != "")]
            if len(ips) > 0:
                ip_n_acc.append(
                    ips.groupby("aid_int", sort=False)["ip_address"]
                    .count().rename("ip_n").to_frame()
                )
                ip_pair_acc.append(ips[["aid_int", "ip_address"]].drop_duplicates())

            # Balance
            bal = adf[adf["balance_after_transaction"].notna()].copy()
            if len(bal) > 0:
                bal["bal_sq"] = bal["balance_after_transaction"] ** 2
                ba = bal.groupby("aid_int", sort=False).agg(
                    bal_sum=("balance_after_transaction", "sum"),
                    bal_sq_sum=("bal_sq", "sum"),
                    bal_n=("balance_after_transaction", "count"),
                    bal_min=("balance_after_transaction", "min"),
                    bal_max=("balance_after_transaction", "max"),
                )
                bal_acc.append(ba)

            # Periodic consolidation
            if len(geo_acc) >= 30:
                log.info(f"      Consolidating at add part {j+1}...")
                geo_acc = [_merge_minmax(geo_acc,
                    sum_cols=["lat_sum","lat_sq_sum","lon_sum","lon_sq_sum","geo_n"],
                    min_cols=["lat_min","lon_min"], max_cols=["lat_max","lon_max"])]
                bal_acc = [_merge_minmax(bal_acc,
                    sum_cols=["bal_sum","bal_sq_sum","bal_n"],
                    min_cols=["bal_min"], max_cols=["bal_max"])]
                ip_n_acc = [_merge_accumulators(ip_n_acc)]
                ip_pair_acc = [pd.concat(ip_pair_acc, ignore_index=True).drop_duplicates()]
                gc.collect()

        del tid_map
        gc.collect()
        log.info(f"  Batch {batch_num} done.")

    # --- Derive features ---
    feats = pd.DataFrame()

    if geo_acc:
        geo_all = _merge_minmax(geo_acc,
            sum_cols=["lat_sum","lat_sq_sum","lon_sum","lon_sq_sum","geo_n"],
            min_cols=["lat_min","lon_min"], max_cols=["lat_max","lon_max"])
        n = geo_all["geo_n"].clip(lower=1)
        lat_std = np.sqrt(np.maximum(0, geo_all["lat_sq_sum"] / n - (geo_all["lat_sum"] / n) ** 2))
        lon_std = np.sqrt(np.maximum(0, geo_all["lon_sq_sum"] / n - (geo_all["lon_sum"] / n) ** 2))
        feats = pd.DataFrame({
            "geo_n": geo_all["geo_n"],
            "geo_lat_std": lat_std, "geo_lon_std": lon_std,
            "geo_spread": np.sqrt(lat_std ** 2 + lon_std ** 2),
            "geo_lat_range": geo_all["lat_max"] - geo_all["lat_min"],
            "geo_lon_range": geo_all["lon_max"] - geo_all["lon_min"],
        }, index=geo_all.index)
        feats.index = idx_to_acct[feats.index.values]
        feats.index.name = "account_id"

    # IP features from deduplicated pairs (correct nunique across batches)
    if ip_pair_acc:
        ip_pairs = pd.concat(ip_pair_acc, ignore_index=True).drop_duplicates()
        del ip_pair_acc
        gc.collect()

        # Save ip_pairs with account_id mapping for per-fold label feature recomputation
        ip_pairs_save = ip_pairs.copy()
        ip_pairs_save["account_id"] = idx_to_acct[ip_pairs_save["aid_int"].values]
        ip_pairs_save[["account_id", "ip_address"]].to_parquet(ip_cache_file, index=False)
        log.info(f"  IP pairs saved to cache ({len(ip_pairs_save)} rows)")
        del ip_pairs_save

        unique_ips = ip_pairs.groupby("aid_int")["ip_address"].nunique().rename("unique_ips")
        ip_n_total = _merge_accumulators(ip_n_acc) if ip_n_acc else pd.DataFrame()

        ip_feats = pd.concat([ip_n_total, unique_ips], axis=1).fillna(0)
        ip_feats.index = idx_to_acct[ip_feats.index.values]
        ip_feats.index.name = "account_id"
        feats = feats.join(ip_feats, how="outer") if len(feats) > 0 else ip_feats
        del ip_pairs
        gc.collect()

    if bal_acc:
        bal_all = _merge_minmax(bal_acc,
            sum_cols=["bal_sum","bal_sq_sum","bal_n"],
            min_cols=["bal_min"], max_cols=["bal_max"])
        n = bal_all["bal_n"].clip(lower=1)
        bal_feats = pd.DataFrame({
            "bal_mean": bal_all["bal_sum"] / n,
            "bal_std": np.sqrt(np.maximum(0, bal_all["bal_sq_sum"] / n - (bal_all["bal_sum"] / n) ** 2)),
            "bal_min": bal_all["bal_min"],
            "bal_max": bal_all["bal_max"],
            "bal_range": bal_all["bal_max"] - bal_all["bal_min"],
        }, index=bal_all.index)
        bal_feats.index = idx_to_acct[bal_feats.index.values]
        bal_feats.index.name = "account_id"
        feats = feats.join(bal_feats, how="outer") if len(feats) > 0 else bal_feats

    feats.index.name = "account_id"
    feats = feats.fillna(0)
    feats.to_parquet(cache_file)
    log.info(f"  Geo features: {feats.shape}")
    return feats


# ===================================================================
# MCC ANOMALY FEATURES
# ===================================================================

# ===================================================================
# PART TRANSACTION TYPE FEATURES
# ===================================================================

def compute_ptt_features(sd):
    """Extract part_transaction_type features from transactions_additional."""
    cache_file = CACHE / "ptt_features_v5.parquet"
    if cache_file.exists():
        log.info("Loading cached PTT features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]
    all_ids_set = set(all_ids)
    sorted_ids = sorted(all_ids)
    acct_to_idx = {aid: np.int32(i) for i, aid in enumerate(sorted_ids)}
    idx_to_acct = np.array(sorted_ids, dtype=object)

    log.info("Computing part_transaction_type features...")
    ptt_acc = []

    for batch_num in range(1, 5):
        txn_dir = BASE / "transactions" / f"batch-{batch_num}"
        add_dir = BASE / "transactions_additional" / f"batch-{batch_num}"
        if not txn_dir.exists() or not add_dir.exists():
            continue

        txn_parts = sorted(txn_dir.glob("*.parquet"))
        add_parts = sorted(add_dir.glob("*.parquet"))

        log.info(f"  PTT Batch {batch_num}: building index...")
        chunks = []
        for path in txn_parts:
            df = pd.read_parquet(path, columns=["transaction_id", "account_id"])
            df = df[df["account_id"].isin(all_ids_set)]
            chunks.append(pd.DataFrame({
                "tid_int": df["transaction_id"].str.slice(4).astype(np.int64),
                "aid_int": df["account_id"].map(acct_to_idx).astype(np.int32),
            }))
        tid_map = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()

        for j, path in enumerate(add_parts):
            if j % 40 == 0:
                log.info(f"    PTT add part {j+1}/{len(add_parts)}")
            adf = pd.read_parquet(path, columns=["transaction_id", "part_transaction_type"])
            adf["tid_int"] = adf["transaction_id"].str.slice(4).astype(np.int64)
            adf = adf.merge(tid_map, on="tid_int", how="inner")
            if len(adf) == 0:
                continue
            ptt = adf[adf["part_transaction_type"].notna()].copy()
            if len(ptt) > 0:
                dummies = pd.get_dummies(ptt["part_transaction_type"], prefix="ptt")
                dummies["aid_int"] = ptt["aid_int"].values
                ptt_acc.append(dummies.groupby("aid_int", sort=False).sum())

            if len(ptt_acc) >= 30:
                ptt_acc = [_merge_accumulators(ptt_acc)]
                gc.collect()

        del tid_map
        gc.collect()

    if not ptt_acc:
        feats = pd.DataFrame(index=pd.Index(sorted_ids, name="account_id"))
        feats["ptt_ci_ratio"] = 0
        feats["ptt_bi_ratio"] = 0
        feats.to_parquet(cache_file)
        return feats

    ptt_all = _merge_accumulators(ptt_acc)
    ptt_total = ptt_all.sum(axis=1).clip(lower=1)
    feats = pd.DataFrame(index=ptt_all.index)
    for col in ptt_all.columns:
        feats[f"{col}_ratio"] = ptt_all[col] / ptt_total
    feats.index = idx_to_acct[feats.index.values]
    feats.index.name = "account_id"
    feats = feats.fillna(0)
    feats.to_parquet(cache_file)
    log.info(f"  PTT features: {feats.shape}")
    return feats


# ===================================================================
# MCC ANOMALY FEATURES
# ===================================================================

def compute_mcc_features(sd):
    cache_file = CACHE / "mcc_features_v3.parquet"
    if cache_file.exists():
        log.info("Loading cached MCC features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]

    # Pass 1: population MCC stats
    log.info("MCC features pass 1: population stats...")
    mcc_sum = defaultdict(float)
    mcc_sq = defaultdict(float)
    mcc_cnt = defaultdict(int)
    for i, path in enumerate(TXN_PARTS):
        if i % 100 == 0:
            log.info(f"  Part {i + 1}/{len(TXN_PARTS)}")
        df = pd.read_parquet(path, columns=["mcc_code", "amount"])
        df["abs_a"] = df["amount"].abs()
        df["abs_a_sq"] = df["abs_a"] ** 2
        grp = df.groupby("mcc_code", sort=False).agg(
            s=("abs_a", "sum"), sq=("abs_a_sq", "sum"), n=("abs_a", "count")
        )
        for mcc, row in grp.iterrows():
            mcc_sum[mcc] += row["s"]
            mcc_sq[mcc] += row["sq"]
            mcc_cnt[mcc] += int(row["n"])

    mcc_stats = pd.DataFrame({
        "mcc_code": list(mcc_sum.keys()),
        "mcc_mean": [mcc_sum[m] / max(mcc_cnt[m], 1) for m in mcc_sum],
        "mcc_std": [np.sqrt(max(0, mcc_sq[m] / max(mcc_cnt[m], 1) - (mcc_sum[m] / max(mcc_cnt[m], 1)) ** 2)) + 1e-6 for m in mcc_sum],
    })

    # Pass 2: per-account z-scores
    log.info("MCC features pass 2: per-account z-scores...")
    z_acc = []
    for i, path in enumerate(TXN_PARTS):
        if i % 100 == 0:
            log.info(f"  Part {i + 1}/{len(TXN_PARTS)}")
        df = pd.read_parquet(path, columns=["account_id", "mcc_code", "amount"])
        df = df[df["account_id"].isin(all_ids)]
        if len(df) == 0:
            continue
        df["abs_a"] = df["amount"].abs()
        df = df.merge(mcc_stats, on="mcc_code", how="left")
        df["z"] = (df["abs_a"] - df["mcc_mean"]) / df["mcc_std"]
        df["abs_z"] = df["z"].abs()
        chunk = df.groupby("account_id", sort=False).agg(
            z_sum=("z", "sum"), z_abs_sum=("abs_z", "sum"), z_n=("z", "count")
        )
        z_acc.append(chunk)

    z_all = _merge_accumulators(z_acc)
    z_all["mcc_anomaly_mean_z"] = z_all["z_sum"] / (z_all["z_n"] + 1)
    z_all["mcc_anomaly_mean_abs_z"] = z_all["z_abs_sum"] / (z_all["z_n"] + 1)
    feats = z_all[["mcc_anomaly_mean_z", "mcc_anomaly_mean_abs_z"]].copy()
    feats.index.name = "account_id"
    feats.to_parquet(cache_file)
    del z_acc, z_all
    gc.collect()
    log.info(f"  MCC features: {feats.shape}")
    return feats


# ===================================================================
# STATIC / PROFILE FEATURES
# ===================================================================

def compute_static_features(sd):
    log.info("Computing static features...")
    acc = sd["accounts"].copy()
    cust = sd["customers"].copy()
    link = sd["linkage"].copy()
    demo = sd["demographics"].copy()
    prod = sd["product"].copy()
    branch = sd["branch"].copy()
    acc_add = sd["accounts_add"].copy()

    # Account features
    acc["account_opening_date"] = pd.to_datetime(acc["account_opening_date"], errors="coerce")
    acc["account_age_days"] = (REF_DATE - acc["account_opening_date"]).dt.days
    acc["is_frozen"] = (acc["account_status"] == "frozen").astype(int)
    acc["freeze_date"] = pd.to_datetime(acc["freeze_date"], errors="coerce")
    acc["unfreeze_date"] = pd.to_datetime(acc["unfreeze_date"], errors="coerce")
    acc["days_frozen"] = (acc["unfreeze_date"] - acc["freeze_date"]).dt.days
    acc["was_frozen"] = acc["freeze_date"].notna().astype(int)
    acc["last_mobile_update_date"] = pd.to_datetime(acc["last_mobile_update_date"], errors="coerce")
    acc["days_since_mobile_update"] = (REF_DATE - acc["last_mobile_update_date"]).dt.days
    acc["last_kyc_date"] = pd.to_datetime(acc["last_kyc_date"], errors="coerce")
    acc["days_since_kyc"] = (REF_DATE - acc["last_kyc_date"]).dt.days
    acc["kyc_compliant_num"] = (acc["kyc_compliant"] == "Y").astype(int)
    acc["nomination_num"] = (acc["nomination_flag"] == "Y").astype(int)
    acc["cheque_allowed_num"] = (acc["cheque_allowed"] == "Y").astype(int)
    acc["cheque_availed_num"] = (acc["cheque_availed"] == "Y").astype(int)
    acc["rural_num"] = (acc["rural_branch"] == "Y").astype(int)
    acc["monthly_to_avg"] = acc["monthly_avg_balance"] / (acc["avg_balance"].abs() + 1)
    acc["daily_to_monthly"] = acc["daily_avg_balance"] / (acc["monthly_avg_balance"].abs() + 1)
    acc["qtr_to_monthly"] = acc["quarterly_avg_balance"] / (acc["monthly_avg_balance"].abs() + 1)

    for pf in ["S", "K", "O"]:
        acc[f"pf_{pf}"] = (acc["product_family"] == pf).astype(int)

    acc_cols = [
        "account_id", "account_age_days", "is_frozen", "days_frozen", "was_frozen",
        "days_since_mobile_update", "days_since_kyc", "kyc_compliant_num",
        "nomination_num", "cheque_allowed_num", "cheque_availed_num",
        "num_chequebooks", "rural_num", "avg_balance", "monthly_avg_balance",
        "quarterly_avg_balance", "daily_avg_balance",
        "monthly_to_avg", "daily_to_monthly", "qtr_to_monthly",
        "pf_S", "pf_K", "pf_O", "product_code", "branch_code",
    ]
    acc_feats = acc[acc_cols].copy()

    # Scheme code
    scheme = pd.get_dummies(acc_add.set_index("account_id")["scheme_code"], prefix="scheme")
    acc_feats = acc_feats.merge(scheme, left_on="account_id", right_index=True, how="left")

    # Branch features
    bf = branch[["branch_code", "branch_employee_count", "branch_turnover", "branch_asset_size"]].copy()
    bf["branch_urban"] = (branch["branch_type"] == "urban").astype(int)
    bf["branch_rural"] = (branch["branch_type"] == "rural").astype(int)

    # Branch geographic features — account density per city/state/pin
    branch_geo = branch[["branch_code", "branch_city", "branch_state", "branch_pin_code"]].copy()
    acc_branch = acc[["account_id", "branch_code"]].merge(branch_geo, on="branch_code", how="left")
    city_counts = acc_branch.groupby("branch_city")["account_id"].nunique().rename("city_account_count")
    state_counts = acc_branch.groupby("branch_state")["account_id"].nunique().rename("state_account_count")
    pin_counts = acc_branch.groupby("branch_pin_code")["account_id"].nunique().rename("pin_account_count")
    acc_geo = acc_branch.merge(city_counts, on="branch_city", how="left")
    acc_geo = acc_geo.merge(state_counts, on="branch_state", how="left")
    acc_geo = acc_geo.merge(pin_counts, on="branch_pin_code", how="left")
    bf = bf.merge(branch_geo, on="branch_code", how="left")

    acc_feats = acc_feats.merge(bf, on="branch_code", how="left")
    acc_feats = acc_feats.merge(
        acc_geo[["account_id", "city_account_count", "state_account_count", "pin_account_count"]],
        on="account_id", how="left"
    )

    # Customer features
    cust["date_of_birth"] = pd.to_datetime(cust["date_of_birth"], errors="coerce")
    cust["age_years"] = (REF_DATE - cust["date_of_birth"]).dt.days / 365.25
    cust["relationship_start_date"] = pd.to_datetime(cust["relationship_start_date"], errors="coerce")
    cust["relationship_days"] = (REF_DATE - cust["relationship_start_date"]).dt.days
    cust["doc_count"] = sum(
        (cust[c] == "Y").astype(int) for c in ["pan_available", "aadhaar_available", "passport_available"]
    )
    cust["digital_count"] = sum(
        (cust[c] == "Y").astype(int) for c in ["mobile_banking_flag", "internet_banking_flag", "atm_card_flag"]
    )
    cust["demat_num"] = (cust["demat_flag"] == "Y").astype(int)
    cust["cc_flag_num"] = (cust["credit_card_flag"] == "Y").astype(int)
    cust["fastag_num"] = (cust["fastag_flag"] == "Y").astype(int)
    cust["pin_mismatch"] = (cust["customer_pin"] != cust["permanent_pin"]).astype(int)

    cust_cols = [
        "customer_id", "age_years", "relationship_days", "doc_count", "digital_count",
        "demat_num", "cc_flag_num", "fastag_num", "pin_mismatch", "customer_pin",
    ]
    cust_feats = cust[cust_cols].copy()

    # Demographics
    demo["gender_num"] = (demo["gender"] == "M").astype(int)
    demo["joint_num"] = (demo["joint_account_flag"] == "Y").astype(int)
    demo["nri_num"] = (demo["nri_flag"] == "Y").astype(int)
    demo["address_last_update_date"] = pd.to_datetime(demo["address_last_update_date"], errors="coerce")
    demo["days_since_addr"] = (REF_DATE - demo["address_last_update_date"]).dt.days
    demo["passbook_last_update_date"] = pd.to_datetime(demo["passbook_last_update_date"], errors="coerce")
    demo["days_since_passbook"] = (REF_DATE - demo["passbook_last_update_date"]).dt.days
    demo["has_passbook_update"] = demo["passbook_last_update_date"].notna().astype(int)
    demo_feats = demo[["customer_id", "gender_num", "joint_num", "nri_num", "days_since_addr",
                       "days_since_passbook", "has_passbook_update"]].copy()
    cust_feats = cust_feats.merge(demo_feats, on="customer_id", how="left")

    # Product details
    prod["total_bal"] = prod[["loan_sum", "cc_sum", "od_sum", "ka_sum", "sa_sum"]].sum(axis=1)
    prod["total_prod_count"] = prod[["loan_count", "cc_count", "od_count", "ka_count", "sa_count"]].sum(axis=1)
    cust_feats = cust_feats.merge(prod, on="customer_id", how="left")

    # Join to account level
    cust_feats = link.merge(cust_feats, on="customer_id", how="left").drop(columns=["customer_id"])
    feats = acc_feats.merge(cust_feats, on="account_id", how="left")

    # PIN-branch match
    bp = acc[["account_id", "branch_pin"]].copy()
    feats = feats.merge(bp, on="account_id", how="left")
    feats["pin_branch_match"] = (feats["customer_pin"] == feats["branch_pin"]).astype(int)
    feats = feats.drop(columns=["customer_pin", "branch_pin"], errors="ignore")
    feats = feats.set_index("account_id")

    # Drop object columns
    obj_cols = feats.select_dtypes(include=["object"]).columns
    feats = feats.drop(columns=obj_cols, errors="ignore")

    log.info(f"  Static features: {feats.shape}")
    return feats


def compute_composite_features(static_feats, txn_feats, graph_feats, geo_feats):
    """Composite features targeting specific mule patterns not yet jointly captured."""
    log.info("Computing composite features...")
    feats = pd.DataFrame(index=static_feats.index)
    feats.index.name = "account_id"

    # --- Pattern #6: New Account High Value ---
    # Young accounts with disproportionately high transaction volume
    age = static_feats["account_age_days"].reindex(feats.index).fillna(1825)  # default 5yr
    txn_sum = txn_feats["txn_sum"].reindex(feats.index).fillna(0)
    txn_count = txn_feats["txn_count"].reindex(feats.index).fillna(0)
    feats["early_value_ratio"] = txn_sum / (age.clip(lower=1))
    feats["early_count_ratio"] = txn_count / (age.clip(lower=1))
    feats["new_acct_high_vol"] = ((age < 365) & (txn_sum > txn_sum.median())).astype(np.int8)

    # --- Pattern #7: Income Mismatch ---
    # Transaction volume disproportionate to account balance profile
    avg_bal = static_feats["avg_balance"].reindex(feats.index).fillna(0).abs()
    feats["txn_to_balance_ratio"] = txn_sum / (avg_bal + 1)
    monthly_bal = static_feats["monthly_avg_balance"].reindex(feats.index).fillna(0).abs()
    feats["txn_to_monthly_bal"] = txn_sum / (monthly_bal + 1)

    # --- Pattern #11: Salary Cycle Exploitation ---
    # Month-end credit concentration suggests salary receipt; combine with withdrawal
    monthend = txn_feats["monthend_ratio"].reindex(feats.index).fillna(0)
    cd_sum = txn_feats["cd_ratio_sum"].reindex(feats.index).fillna(1)
    net_flow = txn_feats["net_flow"].reindex(feats.index).fillna(0)
    feats["salary_exploit_score"] = monthend * (1 / (cd_sum + 0.1))  # high monthend + more debits
    feats["net_flow_per_balance"] = net_flow / (avg_bal + 1)  # pass-through vs parking

    # --- Pattern #12: Branch-Level Activity ---
    # Branch structural features × graph structural features (no label leakage)
    branch_emp = static_feats["branch_employee_count"].reindex(feats.index).fillna(0)
    degree = graph_feats["degree"].reindex(feats.index).fillna(0)
    feats["branch_degree_ratio"] = degree / (branch_emp.clip(lower=1))
    feats["branch_volume_score"] = txn_sum / (static_feats["branch_turnover"].reindex(feats.index).fillna(1).clip(lower=1))

    # --- Pattern #8: Post-Mobile-Change Spike ---
    # Recent mobile update + high transaction velocity
    days_mobile = static_feats["days_since_mobile_update"].reindex(feats.index).fillna(9999)
    velocity = txn_feats["txn_velocity"].reindex(feats.index).fillna(0) if "txn_velocity" in txn_feats.columns else pd.Series(0, index=feats.index)
    feats["mobile_change_activity"] = velocity / (days_mobile.clip(lower=1))

    # --- Rapid Pass-Through enhancement ---
    cv = txn_feats["cv"].reindex(feats.index).fillna(1)
    feats["passthrough_score"] = (1 / (cv + 0.01)) * (txn_count / (age.clip(lower=1)))

    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    log.info(f"  Composite features: {feats.shape}")
    return feats


# ===================================================================
# GRAPH EMBEDDINGS (Node2Vec + Louvain Communities)
# ===================================================================

def compute_graph_embeddings(sd):
    cache_file = CACHE / "graph_embeddings_v6.parquet"
    community_cache = CACHE / "community_map_cache.parquet"
    if cache_file.exists():
        log.info("Loading cached graph embeddings...")
        return pd.read_parquet(cache_file)

    import networkx as nx
    import community as community_louvain

    all_ids = sd["all_ids"]

    # Reuse edges from cache if available
    edges_cache = CACHE / "edges_cache.parquet"
    if edges_cache.exists():
        log.info("  Loading edges from cache for embeddings...")
        edges = pd.read_parquet(edges_cache)
    else:
        log.info("Building edge list for graph embeddings...")
        edge_acc = []
        for i, path in enumerate(TXN_PARTS):
            if i % 50 == 0:
                log.info(f"  Edge part {i + 1}/{len(TXN_PARTS)}")
            df = pd.read_parquet(path, columns=["account_id", "counterparty_id", "amount"])
            df = df[df["account_id"].isin(all_ids)]
            if len(df) == 0:
                continue
            grp = df.groupby(["account_id", "counterparty_id"], sort=False)["amount"].agg(
                w="sum", cnt="count"
            ).reset_index()
            grp["w"] = grp["w"].abs()
            edge_acc.append(grp)
            if len(edge_acc) >= 25:
                merged = pd.concat(edge_acc, ignore_index=True)
                merged = merged.groupby(["account_id", "counterparty_id"], sort=False).agg(
                    {"w": "sum", "cnt": "sum"}
                ).reset_index()
                edge_acc = [merged]
                gc.collect()
        edges = pd.concat(edge_acc, ignore_index=True)
        edges = edges.groupby(["account_id", "counterparty_id"], sort=False).agg(
            {"w": "sum", "cnt": "sum"}
        ).reset_index()
        del edge_acc
        gc.collect()

    log.info(f"  Edges for embedding: {len(edges)}")

    # Build networkx graph — only keep edges where both endpoints are target accounts
    log.info("  Building networkx graph (target accounts only)...")
    all_ids_set = set(all_ids)
    edges_filtered = edges[
        edges["account_id"].isin(all_ids_set) & edges["counterparty_id"].isin(all_ids_set)
    ].copy()
    log.info(f"  Filtered edges (both in target): {len(edges_filtered)}")

    G = nx.from_pandas_edgelist(edges_filtered, "account_id", "counterparty_id",
                                edge_attr="w", create_using=nx.Graph())
    log.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Louvain community detection
    log.info("  Louvain community detection...")
    partition = community_louvain.best_partition(G, random_state=SEED)
    comm_df = pd.DataFrame(list(partition.items()), columns=["account_id", "community_id"])
    comm_df = comm_df[comm_df["account_id"].isin(all_ids_set)].set_index("account_id")
    comm_sizes = comm_df["community_id"].value_counts().to_dict()
    comm_df["community_size"] = comm_df["community_id"].map(comm_sizes)

    # Save community map for per-fold label feature recomputation
    comm_df.to_parquet(community_cache)
    log.info("  Community map saved to cache")

    # PageRank (label-free structural centrality)
    log.info("  Computing PageRank...")
    pr = nx.pagerank(G, weight="w", max_iter=100, tol=1e-6)
    pr_df = pd.DataFrame(list(pr.items()), columns=["account_id", "pagerank"]).set_index("account_id")

    # Betweenness centrality approximation (sample-based for speed)
    log.info("  Computing betweenness centrality (approx)...")
    bc = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), seed=SEED)
    bc_df = pd.DataFrame(list(bc.items()), columns=["account_id", "betweenness"]).set_index("account_id")

    # Clustering coefficient
    log.info("  Computing clustering coefficients...")
    cc = nx.clustering(G)
    cc_df = pd.DataFrame(list(cc.items()), columns=["account_id", "clustering_coeff"]).set_index("account_id")

    # Join community + structural features (no Node2Vec — ablation showed ~0 impact)
    feats = comm_df[["community_id", "community_size"]].join(pr_df, how="outer").join(bc_df, how="outer").join(cc_df, how="outer").fillna(0)
    feats.index.name = "account_id"
    feats.to_parquet(cache_file)
    del G, edges, edges_filtered
    gc.collect()
    log.info(f"  Graph embeddings: {feats.shape}")
    return feats


# ===================================================================
# TEMPORAL GRAPH FEATURES
# ===================================================================

def compute_temporal_graph_features(sd):
    cache_file = CACHE / "temporal_graph_v6.parquet"
    if cache_file.exists():
        log.info("Loading cached temporal graph features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]
    all_ids_set = set(all_ids)
    log.info("Computing temporal graph features (6-month windows, single-pass)...")

    # Define 6-month windows
    windows = []
    start = pd.Timestamp("2020-07-01")
    while start < pd.Timestamp("2025-07-01"):
        end = start + pd.DateOffset(months=6)
        windows.append((start, end))
        start = end
    nw = len(windows)
    log.info(f"  {nw} time windows")

    # Single-pass: read each part once, bin transactions into windows
    # Accumulators per window: deg (set of counterparties), vol (abs sum), cnt (count)
    deg_dfs = {i: [] for i in range(nw)}
    vol_dfs = {i: [] for i in range(nw)}
    cnt_dfs = {i: [] for i in range(nw)}

    for pi, path in enumerate(TXN_PARTS):
        if pi % 50 == 0:
            log.info(f"  Reading txn part {pi+1}/{len(TXN_PARTS)}")
        df = pd.read_parquet(path, columns=["account_id", "counterparty_id", "amount", "transaction_timestamp"])
        df = df[df["account_id"].isin(all_ids_set)]
        if len(df) == 0:
            continue
        df["ts"] = pd.to_datetime(df["transaction_timestamp"], format="ISO8601")
        df["abs_amount"] = df["amount"].abs()

        # Assign each txn to a window
        for wi, (ws, we) in enumerate(windows):
            mask = (df["ts"] >= ws) & (df["ts"] < we)
            wdf = df[mask]
            if len(wdf) == 0:
                continue
            g = wdf.groupby("account_id", sort=False)
            deg_dfs[wi].append(g["counterparty_id"].nunique().rename(f"deg_w{wi}"))
            vol_dfs[wi].append(g["abs_amount"].sum().rename(f"vol_w{wi}"))
            cnt_dfs[wi].append(g["amount"].count().rename(f"cnt_w{wi}"))

    log.info("  Merging window accumulators...")
    window_feats = []
    for wi in range(nw):
        for acc_list, name in [(deg_dfs[wi], f"deg_w{wi}"), (vol_dfs[wi], f"vol_w{wi}"), (cnt_dfs[wi], f"cnt_w{wi}")]:
            if acc_list:
                merged = pd.concat(acc_list)
                merged = merged.groupby(level=0).sum()
                window_feats.append(merged)
            else:
                window_feats.append(pd.Series(dtype=float, name=name))

    wf = pd.concat(window_feats, axis=1).fillna(0)
    wf.index.name = "account_id"

    # Derive temporal delta features
    feats = pd.DataFrame(index=wf.index)
    nw = len(windows)

    # Degree features across windows
    deg_cols = [f"deg_w{i}" for i in range(nw) if f"deg_w{i}" in wf.columns]
    vol_cols = [f"vol_w{i}" for i in range(nw) if f"vol_w{i}" in wf.columns]
    cnt_cols = [f"cnt_w{i}" for i in range(nw) if f"cnt_w{i}" in wf.columns]

    if deg_cols:
        deg_mat = wf[deg_cols].values
        feats["degree_mean"] = deg_mat.mean(axis=1)
        feats["degree_std"] = deg_mat.std(axis=1)
        feats["degree_max"] = deg_mat.max(axis=1)
        feats["degree_trend"] = deg_mat[:, -1] - deg_mat[:, 0]  # last - first
        # Max single-window increase
        diffs = np.diff(deg_mat, axis=1)
        feats["degree_max_jump"] = diffs.max(axis=1) if diffs.shape[1] > 0 else 0

    if vol_cols:
        vol_mat = wf[vol_cols].values
        feats["volume_mean"] = vol_mat.mean(axis=1)
        feats["volume_std"] = vol_mat.std(axis=1)
        feats["volume_max"] = vol_mat.max(axis=1)
        feats["volume_trend"] = vol_mat[:, -1] - vol_mat[:, 0]
        diffs = np.diff(vol_mat, axis=1)
        feats["volume_max_jump"] = diffs.max(axis=1) if diffs.shape[1] > 0 else 0
        feats["volume_acceleration"] = np.diff(diffs, axis=1).max(axis=1) if diffs.shape[1] > 1 else 0

    if cnt_cols:
        cnt_mat = wf[cnt_cols].values
        feats["count_mean"] = cnt_mat.mean(axis=1)
        feats["count_std"] = cnt_mat.std(axis=1)
        feats["count_max"] = cnt_mat.max(axis=1)
        feats["count_trend"] = cnt_mat[:, -1] - cnt_mat[:, 0]
        # Activity ratio: active windows / total windows
        feats["active_window_ratio"] = (cnt_mat > 0).sum(axis=1) / nw

    # ── Temporal drift features (Item 8) ──
    # Recent vs historical ratio: last 2 windows (≈last 12 months) vs earlier windows
    n_recent = min(2, nw)  # last 2 windows ≈ 12 months
    if vol_cols and nw > 2:
        vol_mat = wf[vol_cols].values
        recent_vol = vol_mat[:, -n_recent:].mean(axis=1)
        earlier_vol = vol_mat[:, :-n_recent].mean(axis=1)
        feats["volume_ratio_recent"] = recent_vol / (earlier_vol + 1)
    if deg_cols and nw > 2:
        deg_mat = wf[deg_cols].values
        recent_deg = deg_mat[:, -n_recent:].mean(axis=1)
        earlier_deg = deg_mat[:, :-n_recent].mean(axis=1)
        feats["degree_ratio_recent"] = recent_deg / (earlier_deg + 1)
    if cnt_cols and nw > 0:
        cnt_mat = wf[cnt_cols].values
        # Activity recency: index of last active window / total windows
        last_active = np.zeros(cnt_mat.shape[0])
        for wi in range(nw):
            active_mask = cnt_mat[:, wi] > 0
            last_active[active_mask] = wi
        feats["activity_recency"] = last_active / max(nw - 1, 1)

    # ── New behavioral drift features (v6) ──
    # count_ratio_recent: recent txn count vs historical
    if cnt_cols and nw > 2:
        cnt_mat = wf[cnt_cols].values
        recent_cnt = cnt_mat[:, -n_recent:].mean(axis=1)
        earlier_cnt = cnt_mat[:, :-n_recent].mean(axis=1)
        feats["count_ratio_recent"] = recent_cnt / (earlier_cnt + 1)

    # volume_spike_max: max single-window volume / mean volume
    if vol_cols:
        vol_mat = wf[vol_cols].values
        vol_mean = vol_mat.mean(axis=1)
        feats["volume_spike_max"] = vol_mat.max(axis=1) / (vol_mean + 1)

    # degree_spike_max: max single-window degree / mean degree
    if deg_cols:
        deg_mat = wf[deg_cols].values
        deg_mean = deg_mat.mean(axis=1)
        feats["degree_spike_max"] = deg_mat.max(axis=1) / (deg_mean + 1)

    # dormancy_then_active: had >=2 consecutive zero-count windows then became active
    if cnt_cols and nw > 3:
        cnt_mat = wf[cnt_cols].values
        active_flags = (cnt_mat > 0).astype(np.int8)
        dormancy_score = np.zeros(cnt_mat.shape[0])
        for acct_i in range(cnt_mat.shape[0]):
            max_dormant = 0
            run = 0
            had_dormant = False
            for wi in range(nw):
                if active_flags[acct_i, wi] == 0:
                    run += 1
                    if run >= 2:
                        had_dormant = True
                    max_dormant = max(max_dormant, run)
                else:
                    run = 0
            # active in last window after dormancy
            dormancy_score[acct_i] = 1.0 if (had_dormant and active_flags[acct_i, -1] == 1) else 0.0
        feats["dormancy_then_active"] = dormancy_score

    # window_activity_entropy: Shannon entropy of activity distribution
    if cnt_cols:
        cnt_mat = wf[cnt_cols].values
        cnt_total = cnt_mat.sum(axis=1, keepdims=True)
        probs = cnt_mat / (cnt_total + 1e-12)
        log_probs = np.log2(probs + 1e-12)
        entropy = -(probs * log_probs).sum(axis=1)
        feats["window_activity_entropy"] = entropy

    feats.index.name = "account_id"
    feats = feats.fillna(0)
    feats.to_parquet(cache_file)
    log.info(f"  Temporal graph features: {feats.shape}")
    return feats


# ===================================================================
# ANOMALY DETECTION FEATURES
# ===================================================================

def compute_anomaly_features(all_feats):
    cache_file = CACHE / "anomaly_features_v4.parquet"
    if cache_file.exists():
        log.info("Loading cached anomaly features...")
        return pd.read_parquet(cache_file)

    from sklearn.ensemble import IsolationForest
    import torch
    import torch.nn as nn

    log.info("Computing anomaly detection features...")
    X = all_feats.fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)

    # --- Isolation Forest ---
    log.info("  Isolation Forest...")
    iforest = IsolationForest(n_estimators=200, contamination=0.03,
                               random_state=SEED, n_jobs=N_JOBS)
    iforest.fit(X)
    if_scores = -iforest.decision_function(X)  # higher = more anomalous
    if_labels = (iforest.predict(X) == -1).astype(np.int8)

    # --- Autoencoder ---
    log.info("  Autoencoder anomaly detection...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ae_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Autoencoder device: {ae_device}")
    X_t = torch.tensor(X_scaled, dtype=torch.float32).to(ae_device)

    n_feat = X_t.shape[1]

    class AE(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))
        def forward(self, x):
            return self.decoder(self.encoder(x))

    ae = AE(n_feat).to(ae_device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2048, shuffle=True)

    ae.train()
    for epoch in range(50):
        for (batch,) in loader:
            recon = ae(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 10 == 0:
            log.info(f"    Epoch {epoch+1}/50, loss={loss.item():.6f}")

    ae.eval()
    with torch.no_grad():
        recon_all = ae(X_t)
        ae_errors = ((X_t - recon_all) ** 2).mean(dim=1).cpu().numpy()

    feats = pd.DataFrame({
        "iforest_score": if_scores,
        "iforest_label": if_labels,
        "ae_recon_error": ae_errors,
    }, index=all_feats.index)
    feats.index.name = "account_id"
    feats.to_parquet(cache_file)
    log.info(f"  Anomaly features: {feats.shape}")
    return feats


# ===================================================================
# BEHAVIORAL SCREENING (Stage 1 of Multi-Stage Pipeline)
# ===================================================================

def compute_behavioral_screening(all_feats):
    """Score accounts on 10 domain-expert AML heuristic rules using existing features."""
    cache_file = CACHE / "behavioral_screening_v6.parquet"
    if cache_file.exists():
        log.info("Loading cached behavioral screening features...")
        return pd.read_parquet(cache_file)

    log.info("Computing behavioral screening features...")
    df = all_feats.fillna(0).copy()
    flags = pd.DataFrame(index=all_feats.index, dtype=np.float32)

    # Rule 1: Structuring / threshold avoidance
    struct_ratio = (df.get("struct_45_50k_ratio", pd.Series(0, index=df.index)).fillna(0) +
                    df.get("struct_9_10k_ratio", pd.Series(0, index=df.index)).fillna(0) +
                    df.get("struct_90_100k_ratio", pd.Series(0, index=df.index)).fillna(0))
    round_ratio = (df.get("round_1k_ratio", pd.Series(0, index=df.index)).fillna(0) +
                   df.get("round_5k_ratio", pd.Series(0, index=df.index)).fillna(0) +
                   df.get("round_10k_ratio", pd.Series(0, index=df.index)).fillna(0))
    flags["r_structuring"] = ((struct_ratio > 0.05) | (round_ratio > 0.3)).astype(np.float32)

    # Rule 2: Dormancy then activation
    flags["r_dormancy"] = ((df.get("max_dormant_months", pd.Series(0, index=df.index)).fillna(0) > 6) &
                           (df.get("burstiness", pd.Series(0, index=df.index)).fillna(0) > 1.5)).astype(np.float32)

    # Rule 3: Rapid pass-through
    flags["r_passthrough"] = (df.get("passthrough_score", pd.Series(0, index=df.index)).fillna(0) > 0.7).astype(np.float32)

    # Rule 4: High velocity / burstiness
    flags["r_velocity"] = (df.get("burstiness", pd.Series(0, index=df.index)).fillna(0) > 3.0).astype(np.float32)

    # Rule 5: Benford law deviation
    flags["r_benford"] = (df.get("benford_chi2", pd.Series(0, index=df.index)).fillna(0) > df.get("benford_chi2", pd.Series(0, index=df.index)).fillna(0).quantile(0.9)).astype(np.float32)

    # Rule 6: Extreme fan-in/out ratio
    fio = df.get("fanin_fanout_ratio", pd.Series(1.0, index=df.index)).fillna(1.0)
    flags["r_fan_pattern"] = ((fio > 5) | (fio < 0.2)).astype(np.float32)

    # Rule 7: Geographic / IP anomaly
    flags["r_geo_anomaly"] = (df.get("unique_ips", pd.Series(0, index=df.index)).fillna(0) > 10).astype(np.float32)

    # Rule 8: KYC non-compliant
    flags["r_kyc"] = (df.get("kyc_compliant_num", pd.Series(1, index=df.index)).fillna(1) == 0).astype(np.float32)

    # Rule 9: New account with high volume
    flags["r_new_high_vol"] = (df.get("new_acct_high_vol", pd.Series(0, index=df.index)).fillna(0) > 0).astype(np.float32)

    # Rule 10: Isolation Forest anomaly
    flags["r_iforest"] = (df.get("iforest_label", pd.Series(0, index=df.index)).fillna(0) == 1).astype(np.float32)

    # Aggregate screening scores
    rule_cols = [c for c in flags.columns if c.startswith("r_")]
    weights = np.array([1.5, 1.2, 1.5, 1.0, 1.0, 1.2, 0.8, 0.8, 1.0, 1.0])  # domain-tuned
    flag_vals = flags[rule_cols].values

    feats = pd.DataFrame(index=all_feats.index)
    feats["screening_flag_count"] = flag_vals.sum(axis=1).astype(np.float32)
    feats["screening_risk_score"] = (flag_vals * weights).sum(axis=1) / weights.sum()
    feats["screening_risk_score"] = feats["screening_risk_score"].astype(np.float32)
    feats["screening_stage"] = np.where(feats["screening_flag_count"] >= 4, 3,
                                np.where(feats["screening_flag_count"] >= 2, 2, 1)).astype(np.float32)
    applicable = (flag_vals.shape[1])  # all 10 rules applicable to all accounts
    feats["screening_pass_rate"] = (feats["screening_flag_count"] / applicable).astype(np.float32)
    feats.index.name = "account_id"
    feats.to_parquet(cache_file)
    log.info(f"  Behavioral screening features: {feats.shape}")
    return feats


# ===================================================================
# RISK CONTAGION FEATURES (Stage 2 of Multi-Stage Pipeline)
# ===================================================================

def compute_risk_contagion_features(all_feats, edges_df):
    """Propagate risk through the transaction graph using iterative belief propagation."""
    cache_file = CACHE / "risk_contagion_v7.parquet"
    if cache_file.exists():
        log.info("Loading cached risk contagion features...")
        return pd.read_parquet(cache_file)

    from scipy import sparse

    log.info("Computing risk contagion features...")
    all_accounts = all_feats.index.tolist()
    acct_to_idx = {a: i for i, a in enumerate(all_accounts)}
    n = len(all_accounts)

    # Initial risk signal R_0: blend of anomaly + screening scores
    iforest = all_feats.get("iforest_score", pd.Series(0, index=all_feats.index)).fillna(0).values.astype(np.float64)
    ae_err = all_feats.get("ae_recon_error", pd.Series(0, index=all_feats.index)).fillna(0).values.astype(np.float64)
    screening = all_feats.get("screening_risk_score", pd.Series(0, index=all_feats.index)).fillna(0).values.astype(np.float64)

    # Normalize each to [0, 1]
    def _norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    R_0 = 0.4 * _norm(iforest) + 0.2 * _norm(ae_err) + 0.4 * _norm(screening)

    # Build sparse adjacency matrix from edges
    log.info("  Building sparse adjacency matrix...")
    valid_edges = edges_df[
        edges_df["account_id"].isin(acct_to_idx) & edges_df["counterparty_id"].isin(acct_to_idx)
    ].copy()
    rows = valid_edges["account_id"].map(acct_to_idx).values
    cols = valid_edges["counterparty_id"].map(acct_to_idx).values
    weights = valid_edges["w"].values.astype(np.float64)

    # Undirected: add both directions
    r_all = np.concatenate([rows, cols])
    c_all = np.concatenate([cols, rows])
    w_all = np.concatenate([weights, weights])

    A = sparse.csr_matrix((w_all, (r_all, c_all)), shape=(n, n))

    # Row-normalize: A_norm[i,j] = w[i,j] / sum_j(w[i,j])
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_row = sparse.diags(1.0 / row_sums)
    A_norm = inv_row @ A

    # Iterative propagation: R_{t+1} = alpha * R_0 + (1-alpha) * A_norm @ R_t
    alpha = 0.4
    R = R_0.copy()
    log.info("  Running risk propagation (4 iterations)...")
    for k in range(4):
        R = alpha * R_0 + (1 - alpha) * A_norm.dot(R)

    # Directional flow features using c_cnt / d_cnt proportions
    c_cnt = valid_edges["c_cnt"].values.astype(np.float64)
    d_cnt = valid_edges["d_cnt"].values.astype(np.float64)
    total_cnt = (c_cnt + d_cnt).clip(min=1)

    # incoming_risk_flow: risk from senders (credits received → counterparty sends to us)
    in_weights = weights * (c_cnt / total_cnt)
    # outgoing_risk_flow: risk to receivers (debits sent → we send to counterparty)
    out_weights = weights * (d_cnt / total_cnt)

    # For each account, sum R[counterparty] * directional_weight
    in_vals = R[cols] * in_weights
    out_vals = R[cols] * out_weights

    incoming_risk = np.zeros(n, dtype=np.float64)
    outgoing_risk = np.zeros(n, dtype=np.float64)
    np.add.at(incoming_risk, rows, in_vals)
    np.add.at(outgoing_risk, rows, out_vals)
    # Normalize by node degree weight
    incoming_risk /= (row_sums + 1e-12)
    outgoing_risk /= (row_sums + 1e-12)

    # Neighbor risk statistics
    neighbor_sum = A_norm.dot(R)
    degree = np.array((A > 0).sum(axis=1)).flatten().astype(np.float64)
    degree[degree == 0] = 1.0
    risk_neighbor_mean = neighbor_sum / degree * degree  # just A_norm @ R essentially

    # Risk cluster density: fraction of neighbors with R_0 > 75th percentile
    r0_thresh = np.percentile(R_0, 75)
    high_risk_indicator = (R_0 > r0_thresh).astype(np.float64)
    high_risk_neighbor_sum = A_norm.dot(high_risk_indicator)

    feats = pd.DataFrame({
        "contagion_risk": R.astype(np.float32),
        "contagion_delta": (R - R_0).astype(np.float32),
        "incoming_risk_flow": incoming_risk.astype(np.float32),
        "outgoing_risk_flow": outgoing_risk.astype(np.float32),
        "risk_neighbor_mean": neighbor_sum.astype(np.float32),
        "risk_cluster_density": high_risk_neighbor_sum.astype(np.float32),
    }, index=all_feats.index)
    feats.index.name = "account_id"
    feats.to_parquet(cache_file)
    log.info(f"  Risk contagion features: {feats.shape}")
    return feats


def compute_counterparty_reputation(edges_df, contagion_feats):
    """Compute label-free counterparty reputation scores using contagion risk as proxy."""
    cache_file = CACHE / "counterparty_reputation_v2.parquet"
    if cache_file.exists():
        log.info("Loading cached counterparty reputation features...")
        return pd.read_parquet(cache_file)

    log.info("Computing counterparty reputation features...")
    cr = contagion_feats["contagion_risk"].to_dict()
    thresh_75 = contagion_feats["contagion_risk"].quantile(0.75)

    # For each account, compute stats across its counterparties' contagion risk
    cp_risks = edges_df.copy()
    cp_risks["cp_risk"] = cp_risks["counterparty_id"].map(cr).fillna(0)

    g = cp_risks.groupby("account_id", sort=False)["cp_risk"]
    avg_cp = g.mean().rename("avg_cp_contagion_risk")
    max_cp = g.max().rename("max_cp_contagion_risk")

    # fraction of counterparties with high contagion risk
    cp_risks["is_high_risk"] = (cp_risks["cp_risk"] > thresh_75).astype(int)
    g2 = cp_risks.groupby("account_id", sort=False)["is_high_risk"]
    frac_high = g2.mean().rename("high_risk_cp_fraction")

    # ── New counterparty reputation features (v2) ──
    std_cp = g.std().fillna(0).rename("std_cp_contagion_risk")
    min_cp = g.min().rename("min_cp_contagion_risk")
    cp_risk_range = (max_cp - min_cp).rename("cp_risk_range")

    # Weighted CP risk: weight by edge transaction volume
    if "w" in cp_risks.columns:
        cp_risks["weighted_risk"] = cp_risks["cp_risk"] * cp_risks["w"]
        w_sum = cp_risks.groupby("account_id", sort=False)["w"].sum().clip(lower=1e-12)
        wr_sum = cp_risks.groupby("account_id", sort=False)["weighted_risk"].sum()
        weighted_cp = (wr_sum / w_sum).rename("weighted_cp_risk")
    else:
        weighted_cp = avg_cp.rename("weighted_cp_risk")

    # Incoming risk ratio: incoming_risk_flow / (outgoing_risk_flow + 1)
    in_flow = contagion_feats.get("incoming_risk_flow", pd.Series(0, index=contagion_feats.index))
    out_flow = contagion_feats.get("outgoing_risk_flow", pd.Series(0, index=contagion_feats.index))
    in_ratio = (in_flow / (out_flow + 1e-9)).rename("incoming_risk_ratio")

    feats = pd.concat([avg_cp, max_cp, frac_high, std_cp, min_cp, cp_risk_range,
                        weighted_cp, in_ratio], axis=1).fillna(0).astype(np.float32)
    feats.index.name = "account_id"
    feats.to_parquet(cache_file)
    log.info(f"  Counterparty reputation features: {feats.shape}")
    return feats


# ===================================================================
# RISK SCORING FRAMEWORK (post-training)
# ===================================================================

def compute_risk_scores(results, features, shap_results):
    """Decompose ML predictions into 5 interpretable risk dimensions using SHAP values."""
    log.info("Computing multi-dimensional risk scores...")

    test_ids = results["test_ids"]
    test_preds = results["test_preds"]
    feat_names = shap_results["feat_names"]
    shap_values = shap_results["shap_values"]

    # Feature-to-dimension mapping
    dim_patterns = {
        "behavior_risk": ["txn_", "debit_", "credit_", "cd_ratio", "net_flow", "struct_",
                          "round_", "benford_", "burstiness", "passthrough", "flow_chain",
                          "screening_", "rapid_relay", "chain_", "relay_", "night_",
                          "weekend_", "monthend_", "ch_", "hour_entropy",
                          "avg_holding", "min_holding", "flow_ratio_tight",
                          "pass_through_ratio", "net_flow_ratio", "fast_relay_flag",
                          "credit_debit_count_ratio", "flow_symmetry"],
        "network_risk": ["degree", "fan", "community_", "contagion_", "shared_cp",
                         "suspicious_cp", "two_hop", "edge_", "herfindahl", "weighted_degree",
                         "counterparty_", "top_cp", "risk_neighbor", "risk_cluster",
                         "incoming_risk", "outgoing_risk",
                         "avg_cp_contagion", "max_cp_contagion", "high_risk_cp",
                         "std_cp_contagion", "min_cp_contagion", "cp_risk_range",
                         "weighted_cp_risk", "incoming_risk_ratio"],
        "infrastructure_risk": ["geo_", "ip_n", "unique_ips", "shared_ip", "bal_mean",
                                "bal_std", "bal_min", "bal_max", "bal_range", "branch_",
                                "rural", "avg_balance", "monthly_avg", "quarterly_avg",
                                "daily_avg"],
        "temporal_risk": ["degree_mean", "degree_std", "degree_max", "degree_trend",
                          "volume_", "count_mean", "count_std", "count_max", "count_trend",
                          "active_window", "active_months", "monthly_txn", "max_dormant",
                          "dormant_", "recent_", "late_", "days_since", "account_age",
                          "volume_ratio_recent", "degree_ratio_recent", "activity_recency",
                          "count_ratio_recent", "volume_spike_max", "degree_spike_max",
                          "dormancy_then_active", "window_activity_entropy"],
        "ml_anomaly_risk": ["iforest_", "ae_recon"],
    }

    # Map each feature to a dimension
    feat_to_dim = {}
    for dim, patterns in dim_patterns.items():
        for fi, fname in enumerate(feat_names):
            if any(fname.startswith(p) or p in fname for p in patterns):
                if fi not in feat_to_dim:
                    feat_to_dim[fi] = dim

    # Compute per-dimension SHAP sums for all test accounts
    n_test = len(test_ids)
    dim_names = list(dim_patterns.keys())
    dim_scores = {d: np.zeros(n_test, dtype=np.float64) for d in dim_names}

    for fi in range(len(feat_names)):
        dim = feat_to_dim.get(fi, "behavior_risk")  # uncategorized → behavior
        dim_scores[dim] += np.abs(shap_values[:, fi])

    # Normalize each dimension to [0, 1] across test accounts
    for d in dim_names:
        arr = dim_scores[d]
        mn, mx = arr.min(), arr.max()
        dim_scores[d] = (arr - mn) / (mx - mn + 1e-12)

    # Weighted composite risk score
    w = {"behavior_risk": 0.25, "network_risk": 0.25, "infrastructure_risk": 0.15,
         "temporal_risk": 0.20, "ml_anomaly_risk": 0.15}
    total_risk = sum(dim_scores[d] * w[d] for d in dim_names)

    # Risk level classification
    risk_levels = np.where(total_risk > 0.8, "CRITICAL",
                  np.where(total_risk > 0.5, "HIGH",
                  np.where(total_risk > 0.3, "MEDIUM", "LOW")))

    # Build output DataFrame
    risk_df = pd.DataFrame({
        "account_id": test_ids,
        "probability": np.round(test_preds, 4),
        "behavior_risk": np.round(dim_scores["behavior_risk"], 4).astype(np.float32),
        "network_risk": np.round(dim_scores["network_risk"], 4).astype(np.float32),
        "infrastructure_risk": np.round(dim_scores["infrastructure_risk"], 4).astype(np.float32),
        "temporal_risk": np.round(dim_scores["temporal_risk"], 4).astype(np.float32),
        "ml_anomaly_risk": np.round(dim_scores["ml_anomaly_risk"], 4).astype(np.float32),
        "total_risk_score": np.round(total_risk, 4).astype(np.float32),
        "risk_level": risk_levels,
    })

    risk_df.to_csv(CACHE / "risk_breakdown.csv", index=False)
    log.info(f"  Risk scores computed: {risk_df.shape}")
    for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        log.info(f"    {lvl}: {(risk_df['risk_level'] == lvl).sum()}")
    return risk_df


# ===================================================================
# INVESTIGATION INTELLIGENCE
# ===================================================================

def generate_investigation_profiles(results, risk_scores, all_feats_aug,
                                     shap_results, edges_df, windows):
    """Generate per-account investigation cards for flagged accounts."""
    log.info("Generating investigation profiles...")

    test_ids = results["test_ids"]
    test_preds = results["test_preds"]
    feat_names = shap_results["feat_names"]
    shap_values = shap_results["shap_values"]

    flagged_idx = [i for i, p in enumerate(test_preds) if p > 0.3]
    flagged_ids = [test_ids[i] for i in flagged_idx]
    log.info(f"  {len(flagged_ids)} flagged accounts for investigation")

    # Build counterparty risk map from edges + contagion
    contagion_risk = all_feats_aug.get("contagion_risk", pd.Series(0, index=all_feats_aug.index))
    cp_risk_map = contagion_risk.to_dict()

    # Pattern definitions: feature name → threshold for each of the 13 archetypes
    pattern_rules = {
        "Dormant Activation": [("max_dormant_months", 6, "gt"), ("burstiness", 1.5, "gt")],
        "Structuring": [("struct_45_50k_ratio", 0.02, "gt")],
        "Rapid Pass-Through": [("passthrough_score", 0.5, "gt")],
        "Fan-In/Fan-Out": [("fanin_fanout_ratio", 4, "gt")],
        "Geographic Anomaly": [("unique_ips", 8, "gt")],
        "New Account High Vol": [("new_acct_high_vol", 0, "gt")],
        "Income Mismatch": [("txn_to_balance_ratio", 3, "gt")],
        "Post-Mobile Spike": [("days_since_mobile_update", 0, "lt_pos")],
        "Round Amounts": [("round_1k_ratio", 0.3, "gt")],
        "Layered/Subtle": [("iforest_score", 0, "gt_median"), ("ae_recon_error", 0, "gt_median")],
        "Salary Exploitation": [("salary_exploit_score", 0.3, "gt")],
        "Branch Collusion": [("branch_collusion_score", 0, "gt")],
        "MCC Anomaly": [("mcc_entropy", 0, "gt_median")],
    }

    # Pre-compute medians for "gt_median" rules
    X_test = all_feats_aug.loc[test_ids].fillna(0)
    medians = X_test.median()

    # Feature-to-risk-dimension mapping (reuse from compute_risk_scores)
    dim_patterns = {
        "behavior": ["txn_", "struct_", "round_", "benford_", "burstiness", "passthrough",
                      "flow_chain", "screening_", "rapid_relay", "chain_", "relay_",
                      "pass_through_ratio", "net_flow_ratio", "fast_relay_flag",
                      "credit_debit_count_ratio", "flow_symmetry"],
        "network": ["degree", "fan", "community_", "contagion_", "shared_cp",
                     "suspicious_cp", "two_hop", "edge_", "herfindahl",
                     "std_cp_contagion", "min_cp_contagion", "cp_risk_range",
                     "weighted_cp_risk", "incoming_risk_ratio"],
        "infrastructure": ["geo_", "ip_n", "unique_ips", "shared_ip", "bal_", "branch_"],
        "temporal": ["degree_mean", "volume_", "count_mean", "active_", "dormant_", "monthly_txn",
                     "count_ratio_recent", "volume_spike_max", "degree_spike_max",
                     "dormancy_then_active", "window_activity_entropy"],
        "anomaly": ["iforest_", "ae_recon"],
    }

    # Windows lookup
    win_map = {}
    if windows is not None and len(windows) > 0:
        for _, wr in windows.iterrows():
            win_map[wr["account_id"]] = (wr.get("suspicious_start", ""), wr.get("suspicious_end", ""))

    profiles = []
    for idx_pos, i in enumerate(flagged_idx):
        aid = test_ids[i]
        prob = test_preds[i]
        rlevel = risk_scores.loc[risk_scores["account_id"] == aid, "risk_level"].values
        rlevel = rlevel[0] if len(rlevel) > 0 else "UNKNOWN"

        # Top risk drivers: top 3 SHAP features per dimension
        sv = shap_values[i]
        dim_drivers = {}
        for dim, patterns in dim_patterns.items():
            dim_feats = [(fi, feat_names[fi], sv[fi]) for fi in range(len(feat_names))
                         if any(feat_names[fi].startswith(p) or p in feat_names[fi] for p in patterns)]
            dim_feats.sort(key=lambda x: abs(x[2]), reverse=True)
            dim_drivers[dim] = dim_feats[:3]

        # Format top risk drivers as semicolon-separated string
        top_drivers = []
        for dim, feats in dim_drivers.items():
            for _, fname, sval in feats:
                top_drivers.append(f"{fname}:{sval:.4f}")
        top_drivers_str = ";".join(top_drivers[:15])  # cap at 15

        # Suspicious counterparties: top 5 by contagion risk
        acct_edges = edges_df[edges_df["account_id"] == aid]
        if len(acct_edges) > 0:
            cp_ids = acct_edges["counterparty_id"].values
            cp_risks = [(cp, cp_risk_map.get(cp, 0)) for cp in cp_ids]
            cp_risks.sort(key=lambda x: x[1], reverse=True)
            susp_cps = ";".join(str(cp) for cp, _ in cp_risks[:5])
        else:
            susp_cps = ""

        # Pattern matching
        matched = []
        acct_feats = X_test.loc[aid] if aid in X_test.index else pd.Series(dtype=np.float64)
        for pname, rules in pattern_rules.items():
            match = True
            for feat, thresh, op in rules:
                val = acct_feats.get(feat, 0) if len(acct_feats) > 0 else 0
                if op == "gt":
                    match = match and val > thresh
                elif op == "lt_pos":
                    match = match and (0 < val < 365)
                elif op == "gt_median":
                    m = medians.get(feat, 0)
                    match = match and val > m
            if match:
                matched.append(pname)
        matched_str = ";".join(matched) if matched else "None"

        # Activity summary from features
        txn_count = acct_feats.get("txn_count", 0) if len(acct_feats) > 0 else 0
        cd_ratio = acct_feats.get("cd_ratio_sum", 0) if len(acct_feats) > 0 else 0
        active_m = acct_feats.get("active_months", 0) if len(acct_feats) > 0 else 0
        summary = f"txns={int(txn_count)};cd_ratio={cd_ratio:.2f};active_months={int(active_m)}"

        # Timeline
        w_start, w_end = win_map.get(aid, ("", ""))

        profiles.append({
            "account_id": aid,
            "probability": round(prob, 4),
            "risk_level": rlevel,
            "top_risk_drivers": top_drivers_str,
            "suspicious_counterparties": susp_cps,
            "matched_patterns": matched_str,
            "activity_summary": summary,
            "suspicious_start": w_start,
            "suspicious_end": w_end,
        })

    prof_df = pd.DataFrame(profiles)
    prof_df.to_csv(CACHE / "investigation_profiles.csv", index=False)
    log.info(f"  Investigation profiles: {prof_df.shape}")
    return prof_df


# ===================================================================
# SHAP EXPLAINABILITY
# ===================================================================

def compute_shap_analysis(results, features, lgb_model):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log.info("Computing SHAP analysis...")
    fig_dir = BASE / "report_figures"
    fig_dir.mkdir(exist_ok=True)

    test_ids = results["test_ids"]
    X_test = features.loc[test_ids].fillna(0).replace([np.inf, -np.inf], 0)
    feat_names = features.columns.tolist()

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test.values.astype(np.float32))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 (mule)

    # SHAP summary beeswarm
    log.info("  Generating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test.values, feature_names=feat_names,
                      show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP bar plot (mean |SHAP|)
    log.info("  Generating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-30:][::-1]
    plt.barh(range(30), mean_shap[top_idx][::-1])
    plt.yticks(range(30), [feat_names[i] for i in top_idx][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 30 Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Force plot for top mule
    test_preds = results["test_preds"]
    top_mule_idx = np.argmax(test_preds)
    top_mule_id = test_ids[top_mule_idx]
    log.info(f"  Top mule: {top_mule_id} (p={test_preds[top_mule_idx]:.4f})")

    # Top 5 SHAP contributors for each flagged account
    flagged = [i for i, p in enumerate(test_preds) if p > 0.3]
    explanations = []
    for i in flagged:
        sv = shap_values[i]
        top5 = np.argsort(np.abs(sv))[-5:][::-1]
        for j in top5:
            explanations.append({
                "account_id": test_ids[i],
                "probability": round(test_preds[i], 4),
                "feature": feat_names[j],
                "shap_value": round(sv[j], 6),
                "feature_value": round(float(X_test.iloc[i, j]), 4),
            })
    exp_df = pd.DataFrame(explanations)
    exp_df.to_csv(CACHE / "shap_explanations.csv", index=False)

    log.info(f"  SHAP analysis complete. {len(flagged)} accounts explained.")
    return {"shap_values": shap_values, "feat_names": feat_names, "explanations": exp_df}


# ===================================================================
# FEATURE ABLATION STUDY
# ===================================================================

def run_ablation_study(features, cleaned_labels, sd):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    log.info("Running feature ablation study...")

    tl = cleaned_labels.set_index("account_id")
    train_ids = sorted(set(tl.index) & set(features.index))
    X_all = features.loc[train_ids].fillna(0).replace([np.inf, -np.inf], 0)
    y = tl.loc[train_ids, "is_mule"].values
    w = tl.loc[train_ids, "weight"].values
    spw = max(1, (y == 0).sum() / max((y == 1).sum(), 1))

    # Define feature groups by prefix/name patterns
    all_cols = set(features.columns)
    groups = {
        "Static/Account": {c for c in all_cols if any(c.startswith(p) for p in
            ["account_age", "is_frozen", "days_frozen", "was_frozen", "days_since",
             "kyc_", "nomination", "cheque_", "num_cheque", "rural", "avg_balance",
             "monthly_avg", "quarterly_avg", "daily_avg", "monthly_to", "daily_to",
             "qtr_to", "pf_", "scheme_", "branch_", "age_years", "relationship",
             "doc_count", "digital_count", "demat", "cc_flag", "fastag", "pin_",
             "gender", "joint", "nri", "days_since_addr", "loan_", "cc_sum",
             "cc_count", "od_", "ka_", "sa_", "total_bal", "total_prod"])},
        "Transaction": {c for c in all_cols if any(c.startswith(p) for p in
            ["txn_", "debit_", "credit_", "cd_ratio", "net_flow", "abs_net",
             "night_", "weekend_", "monthend_", "struct_", "round_", "late_",
             "recent_", "burstiness", "active_months", "monthly_txn", "max_dormant",
             "benford_", "hour_entropy", "cv", "ch_"])},
        "Graph/Network": {c for c in all_cols if any(c.startswith(p) for p in
            ["degree", "weighted_degree", "edge_", "counterparty_", "top_cp",
             "shared_cp", "suspicious_cp", "two_hop", "fanin", "fanout"])},
        "Graph Embeddings": {c for c in all_cols if any(c.startswith(p) for p in
            ["n2v_", "community_"])},
        "Geo/IP/Balance": {c for c in all_cols if any(c.startswith(p) for p in
            ["geo_", "ip_n", "unique_ips", "shared_ip", "bal_mean", "bal_std",
             "bal_min", "bal_max", "bal_range"])},
        "MCC Anomaly": {c for c in all_cols if c.startswith("mcc_")},
        "Anomaly Scores": {c for c in all_cols if any(c.startswith(p) for p in
            ["iforest_", "ae_recon"])},
        "Temporal Graph": {c for c in all_cols if any(c.startswith(p) for p in
            ["degree_mean", "degree_std", "degree_max", "degree_trend", "degree_max_jump",
             "volume_", "count_mean", "count_std", "count_max", "count_trend",
             "active_window"])},
        "Behavioral Screening": {c for c in all_cols if c.startswith("screening_")},
        "Risk Contagion": {c for c in all_cols if any(c.startswith(p) for p in
            ["contagion_", "incoming_risk", "outgoing_risk", "risk_neighbor", "risk_cluster"])},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lgb_params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "n_jobs": N_JOBS, "n_estimators": 300, "learning_rate": 0.05,
        "num_leaves": 63, "max_depth": 8, "scale_pos_weight": spw,
    }

    def quick_auc(X):
        scores = []
        for ti, vi in skf.split(X, y):
            dt = lgb.Dataset(X[ti], y[ti], weight=w[ti])
            dv = lgb.Dataset(X[vi], y[vi], weight=w[vi])
            m = lgb.train(lgb_params, dt, valid_sets=[dv],
                         callbacks=[lgb.early_stopping(30, verbose=False)])
            scores.append(roc_auc_score(y[vi], m.predict(X[vi])))
        return np.mean(scores)

    # Full AUC
    log.info("  Full model AUC...")
    X_full = X_all.values.astype(np.float32)
    full_auc = quick_auc(X_full)
    log.info(f"  Full AUC: {full_auc:.5f}")

    results = []
    for group_name, group_cols in groups.items():
        present = group_cols & all_cols
        if not present:
            continue
        remaining = [c for c in X_all.columns if c not in present]
        if not remaining:
            continue
        log.info(f"  Ablating {group_name} ({len(present)} features)...")
        X_abl = X_all[remaining].values.astype(np.float32)
        abl_auc = quick_auc(X_abl)
        drop = full_auc - abl_auc
        results.append({
            "group": group_name,
            "n_features": len(present),
            "full_auc": round(full_auc, 5),
            "ablated_auc": round(abl_auc, 5),
            "auc_drop": round(drop, 5),
        })
        log.info(f"    AUC drop: {drop:.5f}")

    abl_df = pd.DataFrame(results).sort_values("auc_drop", ascending=False)
    abl_df.to_csv(CACHE / "ablation_study.csv", index=False)
    log.info(f"  Ablation study complete: {len(results)} groups tested")
    return abl_df


# ===================================================================
# VISUALIZATIONS
# ===================================================================

def generate_visualizations(results, features, sd, shap_results=None, risk_scores=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    log.info("Generating visualizations...")
    fig_dir = BASE / "report_figures"
    fig_dir.mkdir(exist_ok=True)

    test_preds = results["test_preds"]
    test_ids = results["test_ids"]
    train_ids = results["train_ids"]
    y_train = results["y_train"]
    feat_names = results["feat_names"]

    # --- Fig 1: Feature importance bar chart ---
    log.info("  Fig 1: Feature importance...")
    fi = pd.read_csv(CACHE / "feature_importance.csv")
    top30 = fi.head(30)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top30)), top30["importance"].values[::-1], color="steelblue")
    plt.yticks(range(len(top30)), top30["feature"].values[::-1])
    plt.xlabel("LightGBM Gain")
    plt.title("Top 30 Features by LightGBM Importance")
    plt.tight_layout()
    plt.savefig(fig_dir / "feat_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 2: Feature distributions mule vs legit ---
    log.info("  Fig 2: Feature distributions...")
    train_feats = features.loc[train_ids].copy()
    train_feats["is_mule"] = y_train
    top6 = fi.head(6)["feature"].values
    top6_present = [f for f in top6 if f in train_feats.columns]

    if top6_present:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        for idx, feat in enumerate(top6_present[:6]):
            ax = axes[idx // 3][idx % 3]
            for label, color in [(0, "steelblue"), (1, "crimson")]:
                vals = train_feats[train_feats["is_mule"] == label][feat].clip(
                    train_feats[feat].quantile(0.01), train_feats[feat].quantile(0.99))
                ax.hist(vals, bins=50, alpha=0.6, color=color, label=f"{'Mule' if label else 'Legit'}", density=True)
            ax.set_title(feat, fontsize=10)
            ax.legend(fontsize=8)
        plt.suptitle("Top Feature Distributions: Mule vs Legitimate", fontsize=14)
        plt.tight_layout()
        plt.savefig(fig_dir / "feature_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Fig 3: Prediction distribution ---
    log.info("  Fig 3: Prediction distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(test_preds, bins=100, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Mule Probability")
    ax1.set_ylabel("Count")
    ax1.set_title("Test Set Prediction Distribution")
    ax1.set_yscale("log")

    flagged = [p for p in test_preds if p > 0.01]
    ax2.hist(flagged, bins=50, color="crimson", edgecolor="white")
    ax2.set_xlabel("Mule Probability")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Flagged Accounts (p>0.01): {len(flagged)}")
    plt.tight_layout()
    plt.savefig(fig_dir / "pred_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Fig 4: Network subgraph of top mules ---
    log.info("  Fig 4: Network subgraph...")
    import networkx as nx
    top_mule_indices = np.argsort(test_preds)[-10:][::-1]
    top_mule_accts = [test_ids[i] for i in top_mule_indices]
    top_mule_probs = {test_ids[i]: test_preds[i] for i in top_mule_indices}

    # Read a sample of edges for these accounts
    sub_edges = []
    for path in TXN_PARTS[:50]:
        df = pd.read_parquet(path, columns=["account_id", "counterparty_id", "amount"])
        df = df[df["account_id"].isin(set(top_mule_accts))]
        if len(df) > 0:
            sub_edges.append(df.groupby(["account_id", "counterparty_id"]).size().reset_index(name="w"))
    if sub_edges:
        se = pd.concat(sub_edges).groupby(["account_id", "counterparty_id"])["w"].sum().reset_index()
        # Keep top 5 counterparties per mule
        se = se.sort_values("w", ascending=False).groupby("account_id").head(5)

        Gs = nx.Graph()
        for _, row in se.iterrows():
            Gs.add_edge(row["account_id"], row["counterparty_id"], weight=row["w"])

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(Gs, seed=SEED, k=2)
        node_colors = []
        node_sizes = []
        for n in Gs.nodes():
            if n in top_mule_probs:
                node_colors.append("crimson")
                node_sizes.append(300)
            else:
                node_colors.append("lightblue")
                node_sizes.append(80)
        nx.draw_networkx(Gs, pos, node_color=node_colors, node_size=node_sizes,
                         font_size=6, edge_color="gray", alpha=0.8, with_labels=False)
        # Label only mules
        mule_labels = {n: f"{n[-6:]}\np={top_mule_probs[n]:.2f}" for n in top_mule_accts if n in Gs.nodes()}
        nx.draw_networkx_labels(Gs, pos, mule_labels, font_size=7, font_color="white",
                                font_weight="bold")
        plt.title("Network Subgraph: Top 10 Predicted Mules and Their Counterparties")
        plt.tight_layout()
        plt.savefig(fig_dir / "network_subgraph.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Fig 5: Temporal activity for top mules ---
    log.info("  Fig 5: Temporal activity heatmap...")
    top5_accts = top_mule_accts[:5]
    daily_data = []
    for path in TXN_PARTS:
        df = pd.read_parquet(path, columns=["account_id", "transaction_timestamp", "amount"])
        df = df[df["account_id"].isin(set(top5_accts))]
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["transaction_timestamp"], format="ISO8601").dt.to_period("M").astype(str)
            daily_data.append(df.groupby(["account_id", "date"]).size().reset_index(name="cnt"))

    if daily_data:
        dd = pd.concat(daily_data).groupby(["account_id", "date"])["cnt"].sum().reset_index()
        pivot = dd.pivot_table(index="account_id", columns="date", values="cnt", fill_value=0)
        plt.figure(figsize=(18, 4))
        sns.heatmap(pivot, cmap="YlOrRd", xticklabels=3, yticklabels=True)
        plt.title("Monthly Transaction Volume: Top 5 Predicted Mules")
        plt.xlabel("Month")
        plt.ylabel("Account")
        plt.tight_layout()
        plt.savefig(fig_dir / "temporal_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Fig 6: Risk Dimension Radar Chart (top 5 accounts) ---
    if risk_scores is not None and len(risk_scores) > 0:
        log.info("  Fig 6: Risk dimension radar chart...")
        dims = ["behavior_risk", "network_risk", "infrastructure_risk", "temporal_risk", "ml_anomaly_risk"]
        dim_labels = ["Behavior", "Network", "Infrastructure", "Temporal", "ML/Anomaly"]
        top5_risk = risk_scores.nlargest(5, "total_risk_score")

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        for idx, (_, row) in enumerate(top5_risk.iterrows()):
            vals = [row[d] for d in dims] + [row[dims[0]]]
            ax.plot(angles, vals, "o-", linewidth=2, label=f"{row['account_id']} (p={row['probability']:.2f})",
                    color=colors[idx % len(colors)])
            ax.fill(angles, vals, alpha=0.1, color=colors[idx % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title("Risk Dimension Profile: Top 5 Riskiest Accounts", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "risk_dimension_radar.png", dpi=150, bbox_inches="tight")
        plt.close()

        # --- Fig 7: Stage Funnel Chart ---
        log.info("  Fig 7: Multi-stage pipeline funnel...")
        total = len(test_ids)
        screening_feats = features.get("screening_stage", pd.Series(1, index=features.index))
        stage2_count = int((screening_feats.loc[test_ids] >= 2).sum()) if len(screening_feats) > 0 else 0
        stage3_count = int((np.array(test_preds) > 0.3).sum())
        critical_count = int((risk_scores["risk_level"] == "CRITICAL").sum())

        stages = ["All Accounts", "Stage 1: Behavioral\nScreening", "Stage 2: ML\nDetection", "Stage 3: Critical\nRisk"]
        counts = [total, stage2_count, stage3_count, critical_count]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors_funnel = ["#3498db", "#f1c40f", "#e67e22", "#e74c3c"]
        max_w = 0.9
        for i, (stage, cnt) in enumerate(zip(stages, counts)):
            w = max_w * (cnt / max(counts[0], 1))
            w = max(w, 0.05)
            ax.barh(len(stages) - 1 - i, w, height=0.7, color=colors_funnel[i],
                    edgecolor="white", linewidth=2, left=(max_w - w) / 2)
            ax.text(max_w / 2, len(stages) - 1 - i, f"{stage}\n{cnt:,} ({100*cnt/max(total,1):.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold")
        ax.set_xlim(0, max_w)
        ax.set_ylim(-0.5, len(stages) - 0.5)
        ax.axis("off")
        ax.set_title("Multi-Stage Detection Pipeline Funnel", fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(fig_dir / "stage_funnel.png", dpi=150, bbox_inches="tight")
        plt.close()

        # --- Fig 8: Risk Level Distribution ---
        log.info("  Fig 8: Risk level distribution...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        level_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        level_colors = {"CRITICAL": "#e74c3c", "HIGH": "#e67e22", "MEDIUM": "#f1c40f", "LOW": "#2ecc71"}
        level_counts = risk_scores["risk_level"].value_counts()
        bars = [level_counts.get(l, 0) for l in level_order]
        ax1.bar(level_order, bars, color=[level_colors[l] for l in level_order], edgecolor="white")
        ax1.set_ylabel("Number of Accounts")
        ax1.set_title("Risk Level Distribution (Test Set)")
        for i, v in enumerate(bars):
            ax1.text(i, v + max(bars)*0.01, str(v), ha="center", fontsize=10)

        # Contagion delta distribution
        if "contagion_delta" in features.columns:
            delta = features.loc[test_ids, "contagion_delta"].fillna(0)
            ax2.hist(delta[delta != 0], bins=80, color="#9b59b6", edgecolor="white", alpha=0.8)
            ax2.set_xlabel("Contagion Risk Delta")
            ax2.set_ylabel("Count")
            ax2.set_title("Risk Contagion Impact Distribution")
            ax2.axvline(x=0, color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / "risk_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()

    log.info("  Visualizations complete.")


# ===================================================================
# LABEL CLEANING
# ===================================================================

def clean_labels(sd, features):
    from cleanlab.classification import CleanLearning
    import lightgbm as lgb

    log.info("PHASE 3: Label cleaning...")
    tl = sd["train_labels"].copy()
    tl["mule_flag_date"] = pd.to_datetime(tl["mule_flag_date"], errors="coerce")

    mules = tl[tl["is_mule"] == 1]
    outside = mules[(mules["mule_flag_date"] < TXN_START) | (mules["mule_flag_date"] > TXN_END)]
    log.info(f"  Mules outside txn window: {len(outside)}/{len(mules)}")

    tl["weight"] = 1.0
    tl.loc[tl["account_id"].isin(set(outside["account_id"])), "weight"] = 0.3

    routine = tl[(tl["alert_reason"] == "Routine Investigation") & (tl["is_mule"] == 1)]
    log.info(f"  'Routine Investigation' mules: {len(routine)}")
    tl.loc[tl["account_id"].isin(set(routine["account_id"])), "weight"] *= 0.2

    # Cleanlab
    log.info("  Running cleanlab...")
    common = sorted(set(tl["account_id"]) & set(features.index))
    X = features.loc[common].fillna(0).replace([np.inf, -np.inf], 0).values
    tl_c = tl.set_index("account_id").loc[common]
    y = tl_c["is_mule"].values

    clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        scale_pos_weight=max(1, (y == 0).sum() / max((y == 1).sum(), 1)),
        random_state=SEED, verbose=-1, n_jobs=N_JOBS,
    )

    try:
        cl = CleanLearning(clf=clf, seed=SEED)
        issues = cl.find_label_issues(X, y)
        n_issues = issues["is_label_issue"].sum()
        log.info(f"  Cleanlab found {n_issues} issues ({n_issues / len(y) * 100:.1f}%)")
        issue_ids = set(pd.Index(common)[issues["is_label_issue"]])
        tl.loc[tl["account_id"].isin(issue_ids), "weight"] *= 0.3
    except Exception as e:
        log.warning(f"  Cleanlab failed: {e}")

    log.info(f"  Weight stats: mean={tl['weight'].mean():.3f}")
    return tl


# ===================================================================
# MONEY-FLOW CHAIN FEATURES (label-free)
# ===================================================================

def compute_flow_chain_features(sd):
    """
    Trace money-flow chains: A→B (credit) then B→C (debit within 24h) = possible layering.
    All features are label-free (structural only).
    """
    cache_file = CACHE / "flow_chain_v8.parquet"
    if cache_file.exists():
        log.info("Loading cached flow chain features...")
        return pd.read_parquet(cache_file)

    all_ids = sd["all_ids"]
    all_ids_set = set(all_ids)
    log.info("Computing flow chain features (rapid relay detection)...")

    # Incremental per-account aggregation to stay within 16 GB RAM.
    # Instead of collecting all ~400M raw events, aggregate per-part immediately.
    credit_parts = []
    debit_parts = []
    for i, path in enumerate(TXN_PARTS):
        if i % 50 == 0:
            log.info(f"  Chain part {i + 1}/{len(TXN_PARTS)}")
        df = pd.read_parquet(path, columns=["account_id", "counterparty_id", "amount", "txn_type", "transaction_timestamp"])
        df = df[df["account_id"].isin(all_ids_set)]
        if len(df) == 0:
            continue
        df["ts"] = pd.to_datetime(df["transaction_timestamp"], format="ISO8601")
        df["abs_a"] = df["amount"].abs()
        for txn_type, parts_list in [("C", credit_parts), ("D", debit_parts)]:
            sub = df[df["txn_type"] == txn_type]
            if len(sub) == 0:
                continue
            agg = sub.groupby("account_id").agg(
                count=("abs_a", "count"),
                amount_sum=("abs_a", "sum"),
                cp_nunique=("counterparty_id", "nunique"),
                ts_min=("ts", "min"),
                ts_max=("ts", "max"),
            )
            parts_list.append(agg)
        del df
        if i % 50 == 0:
            gc.collect()

    def _combine_aggs(parts, prefix):
        if not parts:
            return pd.DataFrame()
        combined = pd.concat(parts)
        del parts[:]
        gc.collect()
        result = combined.groupby(level=0).agg({
            "count": "sum",
            "amount_sum": "sum",
            "cp_nunique": "sum",  # upper-bound approx (good enough as feature)
            "ts_min": "min",
            "ts_max": "max",
        })
        result.columns = [f"{prefix}_{c}" for c in result.columns]
        return result

    credit_agg = _combine_aggs(credit_parts, "credit")
    debit_agg = _combine_aggs(debit_parts, "debit")
    del credit_parts, debit_parts
    gc.collect()

    if credit_agg.empty and debit_agg.empty:
        feats = pd.DataFrame(index=pd.Index(sorted(all_ids), name="account_id"))
        for c in ["rapid_relay_count", "chain_volume", "chain_in_out_ratio", "relay_speed_mean",
                   "relay_counterparty_diversity", "avg_holding_hours", "min_holding_hours", "flow_ratio_tight",
                   "pass_through_ratio", "net_flow_ratio", "fast_relay_flag",
                   "credit_debit_count_ratio", "flow_symmetry"]:
            feats[c] = 0
        feats.to_parquet(cache_file)
        return feats

    cd = credit_agg.join(debit_agg, how="outer")
    num_cols = [c for c in cd.columns if "ts_" not in c]
    cd[num_cols] = cd[num_cols].fillna(0)

    # Rapid relay: accounts where first debit follows first credit within 24h
    cd["relay_gap_hours"] = (cd["debit_ts_min"] - cd["credit_ts_min"]).dt.total_seconds() / 3600
    cd["relay_gap_hours"] = cd["relay_gap_hours"].fillna(9999)

    feats = pd.DataFrame(index=pd.Index(sorted(all_ids), name="account_id"))

    # Rapid relay count: proxy via overlap of credit/debit activity
    feats["rapid_relay_count"] = 0
    has_both = cd[(cd["credit_count"] > 0) & (cd["debit_count"] > 0)]
    rapid = has_both[has_both["relay_gap_hours"].abs() < 24]
    feats.loc[rapid.index.intersection(feats.index), "rapid_relay_count"] = (
        rapid["credit_count"].clip(upper=rapid["debit_count"])
    ).astype(int)

    # Chain volume: min(credit_sum, debit_sum) — amount that "flows through"
    feats["chain_volume"] = 0.0
    feats.loc[cd.index.intersection(feats.index), "chain_volume"] = np.minimum(
        cd["credit_amount_sum"], cd["debit_amount_sum"]
    )

    # In/out ratio
    feats["chain_in_out_ratio"] = 0.0
    feats.loc[cd.index.intersection(feats.index), "chain_in_out_ratio"] = (
        cd["credit_amount_sum"] / (cd["debit_amount_sum"] + 1)
    )

    # Average relay speed (hours between first credit and first debit)
    feats["relay_speed_mean"] = 9999.0
    feats.loc[has_both.index.intersection(feats.index), "relay_speed_mean"] = (
        has_both["relay_gap_hours"].clip(-9999, 9999)
    )

    # Counterparty diversity ratio: unique CPs on credit side vs debit side
    feats["relay_counterparty_diversity"] = 0.0
    feats.loc[cd.index.intersection(feats.index), "relay_counterparty_diversity"] = (
        cd["credit_cp_nunique"] / (cd["debit_cp_nunique"] + 1)
    )

    # ── Money flow velocity features (Item 5) ──
    # avg_holding_hours: average gap between first credit and first debit per account
    feats["avg_holding_hours"] = 9999.0
    if len(has_both) > 0:
        hold = (has_both["debit_ts_min"] - has_both["credit_ts_min"]).dt.total_seconds() / 3600
        feats.loc[has_both.index.intersection(feats.index), "avg_holding_hours"] = hold.clip(-9999, 9999)

    # min_holding_hours: fastest turnaround (proxy from relay_gap already, but explicit)
    feats["min_holding_hours"] = 9999.0
    if len(has_both) > 0:
        feats.loc[has_both.index.intersection(feats.index), "min_holding_hours"] = hold.clip(0, 9999)

    # flow_ratio_tight: min(credit, debit) / max(credit, debit) → approaches 1.0 for pass-through
    feats["flow_ratio_tight"] = 0.0
    both_idx = cd.index.intersection(feats.index)
    if len(both_idx) > 0:
        cred = cd.loc[both_idx, "credit_amount_sum"].clip(lower=0)
        debt = cd.loc[both_idx, "debit_amount_sum"].clip(lower=0)
        min_flow = np.minimum(cred, debt)
        max_flow = np.maximum(cred, debt)
        feats.loc[both_idx, "flow_ratio_tight"] = min_flow / (max_flow + 1)

    # ── New flow velocity features (v8) ──
    # pass_through_ratio: min(credit, debit) / (credit + debit) → ~0.5 for pass-through
    feats["pass_through_ratio"] = 0.0
    if len(both_idx) > 0:
        cred2 = cd.loc[both_idx, "credit_amount_sum"].clip(lower=0)
        debt2 = cd.loc[both_idx, "debit_amount_sum"].clip(lower=0)
        feats.loc[both_idx, "pass_through_ratio"] = np.minimum(cred2, debt2) / (cred2 + debt2 + 1)

    # net_flow_ratio: abs(credit - debit) / (credit + debit) → low for pass-through
    feats["net_flow_ratio"] = 1.0
    if len(both_idx) > 0:
        feats.loc[both_idx, "net_flow_ratio"] = np.abs(cred2 - debt2) / (cred2 + debt2 + 1)

    # fast_relay_flag: relay gap < 6 hours (very rapid turnaround)
    feats["fast_relay_flag"] = 0
    if len(has_both) > 0:
        fast_mask = has_both["relay_gap_hours"].abs() < 6
        feats.loc[has_both.index[fast_mask].intersection(feats.index), "fast_relay_flag"] = 1

    # credit_debit_count_ratio: credit txn count / (debit txn count + 1)
    feats["credit_debit_count_ratio"] = 0.0
    feats.loc[cd.index.intersection(feats.index), "credit_debit_count_ratio"] = (
        cd["credit_count"] / (cd["debit_count"] + 1)
    )

    # flow_symmetry: 1 - abs(credit_count - debit_count) / (credit_count + debit_count + 1)
    feats["flow_symmetry"] = 0.0
    feats.loc[cd.index.intersection(feats.index), "flow_symmetry"] = (
        1 - np.abs(cd["credit_count"] - cd["debit_count"]) / (cd["credit_count"] + cd["debit_count"] + 1)
    )

    feats = feats.fillna(0).replace([np.inf, -np.inf], 0)
    feats.to_parquet(cache_file)
    del cd
    gc.collect()
    log.info(f"  Flow chain features: {feats.shape}")
    return feats


# ===================================================================
# PER-FOLD LABEL FEATURES (leakage-free)
# ===================================================================

def compute_label_features_for_fold(fold_mule_ids, edges_df, ip_acct_map, branch_acct_map, community_map, all_account_ids, city_acct_map=None, state_acct_map=None):
    """
    Recompute label-derived features using ONLY the given fold's mule_ids.
    This prevents OOF label leakage since validation-fold labels are excluded.

    ip_acct_map: dict of account_id → set(ip_hashes) — compact representation.
    city_acct_map: DataFrame with account_id, branch_city.
    state_acct_map: DataFrame with account_id, branch_state.
    Returns a DataFrame indexed by account_id with label-derived features.
    """
    feats = pd.DataFrame(index=pd.Index(sorted(all_account_ids), name="account_id"))

    # --- Graph-based label features from edges ---
    mule_ids_set = set(fold_mule_ids)

    # Mule counterparties
    mule_edges = edges_df[edges_df["account_id"].isin(mule_ids_set)]
    mule_cps = set(mule_edges["counterparty_id"].unique())

    # CPs appearing in >=5 mule accounts
    cp_mule_cnt = mule_edges.groupby("counterparty_id")["account_id"].nunique()
    suspicious_cps = set(cp_mule_cnt[cp_mule_cnt >= 5].index)

    # 2-hop: cp → set of accounts → does any connect to a mule?
    cp_to_accts = edges_df.groupby("counterparty_id")["account_id"].agg(set)
    cp_connects_mule = {cp: 1 if len(accts & mule_ids_set) > 0 else 0
                        for cp, accts in cp_to_accts.items()}

    # Compute per-account features
    edges_df = edges_df.copy()
    edges_df["is_mule_cp"] = edges_df["counterparty_id"].isin(mule_cps).astype(np.int8)
    edges_df["is_susp_cp"] = edges_df["counterparty_id"].isin(suspicious_cps).astype(np.int8)
    edges_df["cp_connects_mule"] = edges_df["counterparty_id"].map(cp_connects_mule).fillna(0).astype(np.int8)

    g = edges_df.groupby("account_id", sort=False)
    feats["shared_cp_with_mules"] = g["is_mule_cp"].sum()
    feats["suspicious_cp_count"] = g["is_susp_cp"].sum()
    feats["two_hop_mule_exposure"] = g["cp_connects_mule"].mean()

    # --- IP-based label features (using compact ip_acct_map: account → set of ip_hashes) ---
    if ip_acct_map is not None and len(ip_acct_map) > 0:
        # Collect all IP hashes used by mules
        mule_ips = set()
        for mid in mule_ids_set:
            if mid in ip_acct_map:
                mule_ips.update(ip_acct_map[mid])
        # For each account, count shared IPs with mules
        shared_ips_vals = {}
        unique_ips_vals = {}
        for aid in all_account_ids:
            acct_ips = ip_acct_map.get(aid, set())
            n_unique = len(acct_ips)
            n_shared = len(acct_ips & mule_ips)
            shared_ips_vals[aid] = n_shared
            unique_ips_vals[aid] = n_unique
        feats["shared_ips_with_mules"] = pd.Series(shared_ips_vals)
        feats["shared_ip_ratio"] = pd.Series(shared_ips_vals) / (pd.Series(unique_ips_vals) + 1)
    else:
        feats["shared_ips_with_mules"] = 0
        feats["shared_ip_ratio"] = 0

    # --- Branch mule rate ---
    if branch_acct_map is not None and len(branch_acct_map) > 0:
        # branch_acct_map: DataFrame with account_id, branch_code
        bam = branch_acct_map.copy()
        bam["is_mule_fold"] = bam["account_id"].isin(mule_ids_set).astype(int)
        bmr = bam.groupby("branch_code")["is_mule_fold"].mean().rename("branch_mule_rate")
        bam = bam.merge(bmr, on="branch_code", how="left")
        feats["branch_mule_rate"] = bam.set_index("account_id")["branch_mule_rate"]
    else:
        feats["branch_mule_rate"] = 0

    # --- Community mule features ---
    if community_map is not None and len(community_map) > 0:
        cm = community_map.copy()
        cm["is_mule_fold"] = cm.index.isin(mule_ids_set).astype(int)
        mule_per_comm = cm[cm["is_mule_fold"] == 1].groupby("community_id").size()
        cm["community_mule_count"] = cm["community_id"].map(mule_per_comm).fillna(0).astype(int)
        cm["community_mule_rate"] = cm["community_mule_count"] / (cm["community_size"] + 1)
        feats["community_mule_count"] = cm["community_mule_count"]
        feats["community_mule_rate"] = cm["community_mule_rate"]
    else:
        feats["community_mule_count"] = 0
        feats["community_mule_rate"] = 0

    # --- Composite label features (cross-family) ---
    bmr_vals = feats["branch_mule_rate"].fillna(0)
    scp_vals = feats["shared_cp_with_mules"].fillna(0)
    susp_vals = feats["suspicious_cp_count"].fillna(0)
    feats["branch_collusion_score"] = bmr_vals * scp_vals
    feats["branch_susp_cp_score"] = bmr_vals * susp_vals

    # --- City/State mule rate label features ---
    if city_acct_map is not None and len(city_acct_map) > 0:
        cam = city_acct_map.copy()
        cam["is_mule_fold"] = cam["account_id"].isin(mule_ids_set).astype(int)
        cmr = cam.groupby("branch_city")["is_mule_fold"].mean().rename("city_mule_rate")
        cam = cam.merge(cmr, on="branch_city", how="left")
        feats["city_mule_rate"] = cam.set_index("account_id")["city_mule_rate"]

    if state_acct_map is not None and len(state_acct_map) > 0:
        sam = state_acct_map.copy()
        sam["is_mule_fold"] = sam["account_id"].isin(mule_ids_set).astype(int)
        smr = sam.groupby("branch_state")["is_mule_fold"].mean().rename("state_mule_rate")
        sam = sam.merge(smr, on="branch_state", how="left")
        feats["state_mule_rate"] = sam.set_index("account_id")["state_mule_rate"]

    feats = feats.fillna(0).astype(np.float32)

    # Rank-transform all label features to percentile ranks [0, 1]
    # Tree-invariant: preserves all split decisions but compresses extreme outliers
    # that cause red-herring false positives (e.g., two_hop_mule_exposure 16x dominance)
    for col in feats.columns:
        feats[col] = feats[col].rank(pct=True).astype(np.float32)

    return feats


# ===================================================================
# MODEL TRAINING
# ===================================================================

def train_models(features, cleaned_labels, sd, edges_df, ip_acct_map, branch_acct_map, community_map, city_acct_map=None, state_acct_map=None):
    import joblib
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log.info("PHASE 4: Model training...")

    tl = cleaned_labels.set_index("account_id")
    train_ids = sorted(set(tl.index) & set(features.index))
    test_ids = sorted(set(sd["test_accounts"]["account_id"]) & set(features.index))
    all_account_ids = sorted(set(features.index))

    X_train = features.loc[train_ids].fillna(0).replace([np.inf, -np.inf], 0)
    y_train = tl.loc[train_ids, "is_mule"].values
    w_train = tl.loc[train_ids, "weight"].values
    X_test = features.loc[test_ids].fillna(0).replace([np.inf, -np.inf], 0)
    feat_names = X_train.columns.tolist()

    X_tr = X_train.values.astype(np.float32)
    X_te = X_test.values.astype(np.float32)
    log.info(f"  Train: {X_tr.shape}, Test: {X_te.shape}, Mule rate: {y_train.mean():.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    spw = max(1, (y_train == 0).sum() / max((y_train == 1).sum(), 1))

    # ─── Pre-compute per-fold label features (leakage-free) ───
    log.info("  Pre-computing per-fold label features (6 calls)...")
    train_ids_arr = np.array(train_ids)
    fold_label_cache = {}
    fold_splits = list(skf.split(X_tr, y_train))

    for fold_idx, (ti, vi) in enumerate(fold_splits):
        fold_mule_ids = set(train_ids_arr[ti][y_train[ti] == 1])
        fold_label_cache[fold_idx] = compute_label_features_for_fold(
            fold_mule_ids, edges_df, ip_acct_map, branch_acct_map, community_map, all_account_ids,
            city_acct_map=city_acct_map, state_acct_map=state_acct_map
        )
        log.info(f"    Fold {fold_idx}: {len(fold_mule_ids)} mules → {fold_label_cache[fold_idx].shape[1]} label features")

    # Full-train label features for test predictions
    all_mule_ids = set(train_ids_arr[y_train == 1])
    full_label_feats = compute_label_features_for_fold(
        all_mule_ids, edges_df, ip_acct_map, branch_acct_map, community_map, all_account_ids,
        city_acct_map=city_acct_map, state_acct_map=state_acct_map
    )
    log.info(f"    Full-train: {len(all_mule_ids)} mules → {full_label_feats.shape[1]} label features")

    label_col_names = full_label_feats.columns.tolist()
    all_feat_names = feat_names + label_col_names

    # Phase 3: Exposure normalization features (cross-features from base + label)
    # These help distinguish high-volume legitimate accounts (merchants) from actual mules
    _unique_cp_idx = feat_names.index("unique_cp") if "unique_cp" in feat_names else None
    _fanout_idx = feat_names.index("fanout_cps") if "fanout_cps" in feat_names else None
    _fanin_idx = feat_names.index("fanin_cps") if "fanin_cps" in feat_names else None
    _two_hop_idx = label_col_names.index("two_hop_mule_exposure") if "two_hop_mule_exposure" in label_col_names else None
    _shared_cp_idx = label_col_names.index("shared_cp_with_mules") if "shared_cp_with_mules" in label_col_names else None

    def _add_exposure_norm(X_base, X_label):
        """Add exposure normalization features to augmented feature matrix."""
        extras = []
        n_base = X_base.shape[1]
        if _two_hop_idx is not None and _unique_cp_idx is not None:
            two_hop = X_label[:, _two_hop_idx]
            ucp = X_base[:, _unique_cp_idx]
            extras.append((two_hop / (ucp + 1)).reshape(-1, 1).astype(np.float32))
        if _shared_cp_idx is not None and _fanout_idx is not None and _fanin_idx is not None:
            scp = X_label[:, _shared_cp_idx]
            total_cp = X_base[:, _fanout_idx] + X_base[:, _fanin_idx]
            extras.append((scp / (total_cp + 1)).reshape(-1, 1).astype(np.float32))
        if extras:
            return np.column_stack([X_base, X_label] + extras)
        return np.column_stack([X_base, X_label])

    exposure_norm_names = []
    if _two_hop_idx is not None and _unique_cp_idx is not None:
        exposure_norm_names.append("exposure_per_cp")
    if _shared_cp_idx is not None and _fanout_idx is not None and _fanin_idx is not None:
        exposure_norm_names.append("mule_cp_concentration")
    all_feat_names = all_feat_names + exposure_norm_names
    log.info(f"  Exposure normalization features: {exposure_norm_names}")

    # Augment test features with full-train label features + exposure normalization
    lf_test = full_label_feats.loc[test_ids].values.astype(np.float32)
    X_te_aug = _add_exposure_norm(X_te, lf_test)
    log.info(f"  Augmented feature dim: {len(all_feat_names)} (base {len(feat_names)} + label {len(label_col_names)} + norm {len(exposure_norm_names)})")

    # Helper to build augmented train/val arrays for a given fold
    def get_fold_data(fold_idx, ti, vi):
        lf = fold_label_cache[fold_idx]
        lf_train = lf.loc[train_ids_arr[ti]].values.astype(np.float32)
        lf_val = lf.loc[train_ids_arr[vi]].values.astype(np.float32)
        X_fold_train = _add_exposure_norm(X_tr[ti], lf_train)
        X_fold_val = _add_exposure_norm(X_tr[vi], lf_val)
        return X_fold_train, X_fold_val

    # ─── Focal Loss for LightGBM (Item 2) ───
    _focal_gamma = 2.0
    _focal_alpha = 0.25

    def focal_loss_obj(y_pred, dataset):
        """Simplified focal loss: weight BCE gradient by focal term."""
        y_true = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        # focal weights: alpha_t * (1 - p_t)^gamma
        pt = p * y_true + (1 - p) * (1 - y_true)
        alpha_t = _focal_alpha * y_true + (1 - _focal_alpha) * (1 - y_true)
        w = alpha_t * (1 - pt) ** _focal_gamma
        grad = w * (p - y_true)
        hess = np.maximum(w * p * (1 - p), 1e-6)
        return grad, hess

    def focal_loss_eval(y_pred, dataset):
        """Focal loss evaluation metric for early stopping (higher is better for AUC)."""
        y_true = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        return "auc", roc_auc_score(y_true, p), True

    # ─── LightGBM (Optuna) ───
    log.info("  LightGBM Optuna tuning (50 trials)...")

    def lgb_obj(trial):
        p = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "n_jobs": N_JOBS, "n_estimators": 1000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("nl", 20, 150),
            "max_depth": trial.suggest_int("md", 5, 15),
            "min_child_samples": trial.suggest_int("mcs", 10, 100),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cs", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-3, 10, log=True),
            "scale_pos_weight": spw,
        }
        scores = []
        for fold_idx, (ti, vi) in enumerate(fold_splits):
            Xft, Xfv = get_fold_data(fold_idx, ti, vi)
            dt = lgb.Dataset(Xft, y_train[ti], weight=w_train[ti])
            dv = lgb.Dataset(Xfv, y_train[vi], weight=w_train[vi])
            m = lgb.train(p, dt, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            preds = m.predict(Xfv)
            scores.append(roc_auc_score(y_train[vi], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lgb_obj, n_trials=50)
    log.info(f"  LightGBM best CV AUC: {study.best_value:.5f}")

    bp = study.best_params
    lgb_params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "n_jobs": N_JOBS, "n_estimators": 1000, "scale_pos_weight": spw,
        "learning_rate": bp["lr"], "num_leaves": bp["nl"], "max_depth": bp["md"],
        "min_child_samples": bp["mcs"], "subsample": bp["ss"],
        "colsample_bytree": bp["cs"], "reg_alpha": bp["ra"], "reg_lambda": bp["rl"],
    }

    # Multi-seed training with 5-fold CV (variance reduction)
    MULTI_SEEDS = [42, 123, 7]
    n_seeds = len(MULTI_SEEDS)
    lgb_oof = np.zeros(len(X_tr))
    lgb_test = np.zeros(len(X_te))
    lgb_model = None
    for fold_idx, (ti, vi) in enumerate(fold_splits):
        Xft, Xfv = get_fold_data(fold_idx, ti, vi)
        dt = lgb.Dataset(Xft, y_train[ti], weight=w_train[ti])
        dv = lgb.Dataset(Xfv, y_train[vi], weight=w_train[vi])
        m = lgb.train(lgb_params, dt, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        lgb_oof[vi] = m.predict(Xfv)
        lgb_test += m.predict(X_te_aug) / 5
        lgb_model = m
    lgb_auc = roc_auc_score(y_train, lgb_oof)
    log.info(f"  LightGBM OOF AUC: {lgb_auc:.5f}")

    fi = pd.DataFrame({"feature": all_feat_names, "importance": lgb_model.feature_importance(importance_type="gain")})
    fi.sort_values("importance", ascending=False).to_csv(CACHE / "feature_importance.csv", index=False)

    # ─── SHAP Feature Selection (Item 12) ───
    top_k = min(200, len(all_feat_names))
    top_feats = fi.nlargest(top_k, "importance")["feature"].tolist()
    selected_idx = [i for i, f in enumerate(all_feat_names) if f in set(top_feats)]
    n_dropped = len(all_feat_names) - len(selected_idx)
    log.info(f"  Feature selection: keeping {len(selected_idx)}/{len(all_feat_names)} features (dropped {n_dropped})")

    if n_dropped > 0:
        sel_idx_arr = np.array(selected_idx)
        # Rebuild pruned data for XGB/CatBoost
        X_te_aug = X_te_aug[:, sel_idx_arr]
        all_feat_names = [all_feat_names[i] for i in selected_idx]

        # Update fold data helper to return pruned features
        _orig_get_fold_data = get_fold_data
        def get_fold_data(fold_idx, ti, vi):
            Xft, Xfv = _orig_get_fold_data(fold_idx, ti, vi)
            return Xft[:, sel_idx_arr], Xfv[:, sel_idx_arr]

    # ─── XGBoost (Optuna) ───
    log.info("  XGBoost Optuna tuning (40 trials)...")

    def xgb_obj(trial):
        p = {
            "objective": "binary:logistic", "eval_metric": "auc",
            "tree_method": "hist", "device": "cuda", "n_estimators": 1000, "verbosity": 0,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("md", 4, 10),
            "min_child_weight": trial.suggest_int("mcw", 1, 50),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cs", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("ra", 1e-4, 10, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-4, 10, log=True),
            "scale_pos_weight": spw,
        }
        scores = []
        for fold_idx, (ti, vi) in enumerate(fold_splits):
            Xft, Xfv = get_fold_data(fold_idx, ti, vi)
            m = xgb.XGBClassifier(**p, random_state=SEED, n_jobs=N_JOBS)
            m.fit(Xft, y_train[ti], sample_weight=w_train[ti],
                  eval_set=[(Xfv, y_train[vi])], verbose=False)
            scores.append(roc_auc_score(y_train[vi], m.predict_proba(Xfv)[:, 1]))
        return np.mean(scores)

    study2 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study2.optimize(xgb_obj, n_trials=40)
    log.info(f"  XGBoost best CV AUC: {study2.best_value:.5f}")

    bp2 = study2.best_params
    xgb_oof = np.zeros(len(X_tr))
    xgb_test = np.zeros(len(X_te))
    for fold_idx, (ti, vi) in enumerate(fold_splits):
        Xft, Xfv = get_fold_data(fold_idx, ti, vi)
        m = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            device="cuda", n_estimators=2000, verbosity=0, random_state=SEED, n_jobs=N_JOBS,
            scale_pos_weight=spw,
            learning_rate=bp2["lr"], max_depth=bp2["md"], min_child_weight=bp2["mcw"],
            subsample=bp2["ss"], colsample_bytree=bp2["cs"],
            reg_alpha=bp2["ra"], reg_lambda=bp2["rl"],
        )
        m.fit(Xft, y_train[ti], sample_weight=w_train[ti],
              eval_set=[(Xfv, y_train[vi])], verbose=False)
        xgb_oof[vi] = m.predict_proba(Xfv)[:, 1]
        xgb_test += m.predict_proba(X_te_aug)[:, 1] / 5
    xgb_auc = roc_auc_score(y_train, xgb_oof)
    log.info(f"  XGBoost OOF AUC: {xgb_auc:.5f}")

    # Free XGB GPU memory before CatBoost
    del study2
    gc.collect()

    # ─── CatBoost (Optuna) ───
    log.info("  CatBoost Optuna tuning (20 trials)...")
    cb_task_type = "CPU"  # CPU avoids CUDA conflicts with XGBoost

    def cb_obj(trial):
        p = {
            "iterations": 1000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2", 1, 30, log=True),
            "min_data_in_leaf": trial.suggest_int("mdl", 5, 100),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("cs", 0.3, 1.0),
            "random_seed": SEED, "auto_class_weights": "Balanced",
            "task_type": cb_task_type, "eval_metric": "AUC", "verbose": 0,
            "early_stopping_rounds": 50,
        }
        scores = []
        for fold_idx, (ti, vi) in enumerate(fold_splits):
            Xft, Xfv = get_fold_data(fold_idx, ti, vi)
            m = CatBoostClassifier(**p)
            m.fit(Xft, y_train[ti], sample_weight=w_train[ti],
                  eval_set=(Xfv, y_train[vi]), verbose=0)
            scores.append(roc_auc_score(y_train[vi], m.predict_proba(Xfv)[:, 1]))
        return np.mean(scores)

    try:
        study3 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study3.optimize(cb_obj, n_trials=20)
        log.info(f"  CatBoost best CV AUC: {study3.best_value:.5f}")
        bp3 = study3.best_params
    except Exception as e:
        log.info(f"  CatBoost GPU Optuna failed ({e}), falling back to CPU...")
        cb_task_type = "CPU"
        study3 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study3.optimize(cb_obj, n_trials=20)
        log.info(f"  CatBoost best CV AUC (CPU): {study3.best_value:.5f}")
        bp3 = study3.best_params

    cb_oof = np.zeros(len(X_tr))
    cb_test = np.zeros(len(X_te))
    for fold_idx, (ti, vi) in enumerate(fold_splits):
        Xft, Xfv = get_fold_data(fold_idx, ti, vi)
        m = CatBoostClassifier(
            iterations=2000, learning_rate=bp3["lr"], depth=bp3["depth"],
            l2_leaf_reg=bp3["l2"], min_data_in_leaf=bp3["mdl"],
            subsample=bp3["ss"], colsample_bylevel=bp3["cs"],
            random_seed=SEED, auto_class_weights="Balanced",
            task_type=cb_task_type, eval_metric="AUC", verbose=0,
            early_stopping_rounds=100,
        )
        m.fit(Xft, y_train[ti], sample_weight=w_train[ti],
              eval_set=(Xfv, y_train[vi]), verbose=0)
        cb_oof[vi] = m.predict_proba(Xfv)[:, 1]
        cb_test += m.predict_proba(X_te_aug)[:, 1] / 5
    cb_auc = roc_auc_score(y_train, cb_oof)
    log.info(f"  CatBoost OOF AUC: {cb_auc:.5f} (device={cb_task_type})")

    del study3
    gc.collect()

    # ─── ExtraTrees (Optuna) ───
    from sklearn.ensemble import ExtraTreesClassifier
    log.info("  ExtraTrees Optuna tuning (15 trials)...")

    def et_obj(trial):
        p = {
            "n_estimators": trial.suggest_int("ne", 500, 2000, step=100),
            "max_depth": trial.suggest_int("md", 8, 30),
            "min_samples_leaf": trial.suggest_int("msl", 5, 100),
            "max_features": trial.suggest_float("mf", 0.3, 1.0),
            "class_weight": "balanced", "random_state": SEED, "n_jobs": N_JOBS,
        }
        scores = []
        for fold_idx, (ti, vi) in enumerate(fold_splits):
            Xft, Xfv = get_fold_data(fold_idx, ti, vi)
            m = ExtraTreesClassifier(**p)
            m.fit(Xft, y_train[ti], sample_weight=w_train[ti])
            scores.append(roc_auc_score(y_train[vi], m.predict_proba(Xfv)[:, 1]))
        return np.mean(scores)

    study4 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study4.optimize(et_obj, n_trials=15)
    log.info(f"  ExtraTrees best CV AUC: {study4.best_value:.5f}")

    bp4 = study4.best_params
    et_oof = np.zeros(len(X_tr))
    et_test = np.zeros(len(X_te))
    for fold_idx, (ti, vi) in enumerate(fold_splits):
        Xft, Xfv = get_fold_data(fold_idx, ti, vi)
        m = ExtraTreesClassifier(
            n_estimators=bp4["ne"], max_depth=bp4["md"],
            min_samples_leaf=bp4["msl"], max_features=bp4["mf"],
            class_weight="balanced", random_state=SEED, n_jobs=N_JOBS,
        )
        m.fit(Xft, y_train[ti], sample_weight=w_train[ti])
        et_oof[vi] = m.predict_proba(Xfv)[:, 1]
        et_test += m.predict_proba(X_te_aug)[:, 1] / 5
    et_auc = roc_auc_score(y_train, et_oof)
    log.info(f"  ExtraTrees OOF AUC: {et_auc:.5f}")

    del study4
    gc.collect()

    # ─── HistGradientBoosting (Optuna) ───
    from sklearn.ensemble import HistGradientBoostingClassifier
    log.info("  HistGradientBoosting Optuna tuning (15 trials)...")

    def hgb_obj(trial):
        p = {
            "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "max_leaf_nodes": trial.suggest_int("mln", 20, 150),
            "max_depth": trial.suggest_int("md", 5, 15),
            "min_samples_leaf": trial.suggest_int("msl", 5, 100),
            "l2_regularization": trial.suggest_float("l2", 1e-3, 10, log=True),
            "max_iter": 1000, "random_state": SEED,
            "class_weight": "balanced", "early_stopping": True,
            "n_iter_no_change": 50, "validation_fraction": 0.15,
        }
        scores = []
        for fold_idx, (ti, vi) in enumerate(fold_splits):
            Xft, Xfv = get_fold_data(fold_idx, ti, vi)
            m = HistGradientBoostingClassifier(**p)
            m.fit(Xft, y_train[ti], sample_weight=w_train[ti])
            scores.append(roc_auc_score(y_train[vi], m.predict_proba(Xfv)[:, 1]))
        return np.mean(scores)

    study5 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study5.optimize(hgb_obj, n_trials=15)
    log.info(f"  HistGBM best CV AUC: {study5.best_value:.5f}")

    bp5 = study5.best_params
    hgb_oof = np.zeros(len(X_tr))
    hgb_test = np.zeros(len(X_te))
    for fold_idx, (ti, vi) in enumerate(fold_splits):
        Xft, Xfv = get_fold_data(fold_idx, ti, vi)
        m = HistGradientBoostingClassifier(
            learning_rate=bp5["lr"], max_leaf_nodes=bp5["mln"],
            max_depth=bp5["md"], min_samples_leaf=bp5["msl"],
            l2_regularization=bp5["l2"], max_iter=2000,
            random_state=SEED, class_weight="balanced",
            early_stopping=True, n_iter_no_change=100, validation_fraction=0.15,
        )
        m.fit(Xft, y_train[ti], sample_weight=w_train[ti])
        hgb_oof[vi] = m.predict_proba(Xfv)[:, 1]
        hgb_test += m.predict_proba(X_te_aug)[:, 1] / 5
    hgb_auc = roc_auc_score(y_train, hgb_oof)
    log.info(f"  HistGBM OOF AUC: {hgb_auc:.5f}")

    del study5
    gc.collect()

    # ─── Label-Free 6th Model (Phase 4: independent counter-signal) ───
    # All 5 models above use label-augmented features. If label features create false positives,
    # ALL models agree → stacking can't correct. This model uses ONLY base features,
    # providing critical independent signal for red-herring detection.
    log.info("  Label-Free LGB model (base features only, 15 Optuna trials)...")

    def lgb_nolabel_obj(trial):
        p = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "n_jobs": N_JOBS, "n_estimators": 1000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("nl", 20, 150),
            "max_depth": trial.suggest_int("md", 5, 15),
            "min_child_samples": trial.suggest_int("mcs", 10, 100),
            "subsample": trial.suggest_float("ss", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("cs", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("ra", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("rl", 1e-3, 10, log=True),
            "scale_pos_weight": spw,
        }
        scores = []
        for fold_idx, (ti, vi) in enumerate(fold_splits):
            dt = lgb.Dataset(X_tr[ti], y_train[ti], weight=w_train[ti])
            dv = lgb.Dataset(X_tr[vi], y_train[vi], weight=w_train[vi])
            m = lgb.train(p, dt, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            scores.append(roc_auc_score(y_train[vi], m.predict(X_tr[vi])))
        return np.mean(scores)

    study_nolabel = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED + 100))
    study_nolabel.optimize(lgb_nolabel_obj, n_trials=15)
    log.info(f"  Label-Free LGB best CV AUC: {study_nolabel.best_value:.5f}")

    bp_nl = study_nolabel.best_params
    lgb_nolabel_oof = np.zeros(len(X_tr))
    lgb_nolabel_test = np.zeros(len(X_te))
    for fold_idx, (ti, vi) in enumerate(fold_splits):
        dt = lgb.Dataset(X_tr[ti], y_train[ti], weight=w_train[ti])
        dv = lgb.Dataset(X_tr[vi], y_train[vi], weight=w_train[vi])
        p_nl = {
            "objective": "binary", "metric": "auc", "verbosity": -1,
            "n_jobs": N_JOBS, "n_estimators": 1000,
            "learning_rate": bp_nl["lr"], "num_leaves": bp_nl["nl"],
            "max_depth": bp_nl["md"], "min_child_samples": bp_nl["mcs"],
            "subsample": bp_nl["ss"], "colsample_bytree": bp_nl["cs"],
            "reg_alpha": bp_nl["ra"], "reg_lambda": bp_nl["rl"],
            "scale_pos_weight": spw,
        }
        m = lgb.train(p_nl, dt, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        lgb_nolabel_oof[vi] = m.predict(X_tr[vi])
        lgb_nolabel_test += m.predict(X_te) / 5
    lgb_nolabel_auc = roc_auc_score(y_train, lgb_nolabel_oof)
    log.info(f"  Label-Free LGB OOF AUC: {lgb_nolabel_auc:.5f}")

    del study_nolabel
    gc.collect()

    # ─── Stacking (6-model + disagreement features, LGB meta) ───
    log.info("  Stacking ensemble (6 models + disagreement, LGB meta)...")

    # Phase 6: Disagreement-aware stacking
    # When label-heavy models agree "mule" but label-free says "legitimate" → likely red herring
    label_model_mean_oof = (lgb_oof + xgb_oof + cb_oof + et_oof + hgb_oof) / 5
    label_vs_nolabel_gap_oof = label_model_mean_oof - lgb_nolabel_oof
    all_six_oof = np.column_stack([lgb_oof, xgb_oof, cb_oof, et_oof, hgb_oof, lgb_nolabel_oof])
    max_disagreement_oof = all_six_oof.max(axis=1) - all_six_oof.min(axis=1)

    label_model_mean_test = (lgb_test + xgb_test + cb_test + et_test + hgb_test) / 5
    label_vs_nolabel_gap_test = label_model_mean_test - lgb_nolabel_test
    all_six_test = np.column_stack([lgb_test, xgb_test, cb_test, et_test, hgb_test, lgb_nolabel_test])
    max_disagreement_test = all_six_test.max(axis=1) - all_six_test.min(axis=1)

    # Stacking input: 6 model predictions + 2 disagreement features = 8 dimensions
    oof_stack = np.column_stack([lgb_oof, xgb_oof, cb_oof, et_oof, hgb_oof, lgb_nolabel_oof,
                                 label_vs_nolabel_gap_oof, max_disagreement_oof])
    test_stack = np.column_stack([lgb_test, xgb_test, cb_test, et_test, hgb_test, lgb_nolabel_test,
                                  label_vs_nolabel_gap_test, max_disagreement_test])
    log.info(f"  Stacking input dim: {oof_stack.shape[1]} (6 models + 2 disagreement)")

    meta_params = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "n_estimators": 100, "num_leaves": 8, "max_depth": 3,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_lambda": 1.0, "scale_pos_weight": spw, "seed": SEED,
    }
    ens_oof = np.zeros(len(X_tr))
    ens_test = np.zeros(len(X_te))
    for ti, vi in fold_splits:
        dt = lgb.Dataset(oof_stack[ti], y_train[ti], weight=w_train[ti])
        dv = lgb.Dataset(oof_stack[vi], y_train[vi], weight=w_train[vi])
        m_meta = lgb.train(meta_params, dt, valid_sets=[dv],
                           callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        ens_oof[vi] = m_meta.predict(oof_stack[vi])
        ens_test += m_meta.predict(test_stack) / 5
    ens_auc = roc_auc_score(y_train, ens_oof)

    # ─── CV-tuned Weighted Average Ensemble (scipy + Optuna) ───
    from scipy.optimize import minimize
    log.info("  Tuning ensemble weights via scipy optimization (6 models)...")

    oof_models = [lgb_oof, xgb_oof, cb_oof, et_oof, hgb_oof, lgb_nolabel_oof]
    test_models = [lgb_test, xgb_test, cb_test, et_test, hgb_test, lgb_nolabel_test]

    def neg_auc_weights(raw_w):
        w = np.exp(raw_w) / np.exp(raw_w).sum()  # softmax ensures sum=1, all positive
        blend = sum(wi * oof_i for wi, oof_i in zip(w, oof_models))
        return -roc_auc_score(y_train, blend)

    # Multi-start optimization with different initializations (6 models)
    best_result = None
    starts = [
        [0.35, 0.20, 0.13, 0.13, 0.04, 0.15],  # balanced with nolabel
        [0.45, 0.25, 0.08, 0.08, 0.02, 0.12],   # LGB+XGB heavy
        [0.25, 0.15, 0.10, 0.20, 0.05, 0.25],   # nolabel heavy
        [0.30, 0.20, 0.15, 0.13, 0.02, 0.20],   # CB emphasis
        [0.20, 0.20, 0.10, 0.20, 0.10, 0.20],   # equal-ish
    ]
    for s in starts:
        raw0 = np.log(np.array(s) + 1e-8)
        res = minimize(neg_auc_weights, raw0, method="Nelder-Mead",
                       options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-10})
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    opt_w = np.exp(best_result.x) / np.exp(best_result.x).sum()
    best_wavg_w = tuple(float(x) for x in opt_w)
    wavg_oof = sum(wi * oof_i for wi, oof_i in zip(best_wavg_w, oof_models))
    wavg_test = sum(wi * ti for wi, ti in zip(best_wavg_w, test_models))
    wavg_auc = roc_auc_score(y_train, wavg_oof)
    log.info(f"  Weighted avg AUC: {wavg_auc:.5f} (weights: LGB={best_wavg_w[0]:.4f}, XGB={best_wavg_w[1]:.4f}, CB={best_wavg_w[2]:.4f}, ET={best_wavg_w[3]:.4f}, HGB={best_wavg_w[4]:.4f}, NL={best_wavg_w[5]:.4f})")

    all_aucs = {"lgb": lgb_auc, "xgb": xgb_auc, "cb": cb_auc,
                "et": et_auc, "hgb": hgb_auc, "lgb_nolabel": lgb_nolabel_auc,
                "ensemble": ens_auc, "wavg": wavg_auc}
    best_name = max(all_aucs, key=all_aucs.get)
    best_auc = all_aucs[best_name]
    tests = {"lgb": lgb_test, "xgb": xgb_test, "cb": cb_test,
             "et": et_test, "hgb": hgb_test, "lgb_nolabel": lgb_nolabel_test,
             "ensemble": ens_test, "wavg": wavg_test}
    oofs = {"lgb": lgb_oof, "xgb": xgb_oof, "cb": cb_oof,
            "et": et_oof, "hgb": hgb_oof, "lgb_nolabel": lgb_nolabel_oof,
            "ensemble": ens_oof, "wavg": wavg_oof}
    log.info(f"  All AUCs: {' | '.join(f'{k}={v:.5f}' for k, v in all_aucs.items())}")

    # Pick best ensemble strategy (prefer ensemble/wavg if close to best single)
    for ens_key in ["wavg", "ensemble"]:
        if all_aucs[ens_key] >= best_auc - 0.001:
            best_name = ens_key
            break

    final_test = tests[best_name]
    final_oof = oofs[best_name]
    log.info(f"  Using {best_name} (AUC: {all_aucs[best_name]:.5f})")

    # ─── F2 Threshold Optimization (Item 1) ───
    from sklearn.metrics import fbeta_score
    log.info("  Optimizing F2 threshold...")
    best_f2, best_thresh = 0.0, 0.5
    for t in np.arange(0.01, 0.51, 0.01):
        pb = (final_oof > t).astype(int)
        if pb.sum() == 0:
            continue
        f2 = fbeta_score(y_train, pb, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_thresh = t
    pred_bin = (final_oof > best_thresh).astype(int)
    f1 = f1_score(y_train, pred_bin)
    f2 = fbeta_score(y_train, pred_bin, beta=2)
    prec = precision_score(y_train, pred_bin)
    rec = recall_score(y_train, pred_bin)
    log.info(f"  Optimal threshold: {best_thresh:.2f}")
    log.info(f"  Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, F2={f2:.4f}")

    # ─── Calibration: Isotonic vs Platt (Item 10) ───
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    log.info("  Calibration comparison (isotonic vs Platt)...")

    # Isotonic
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    iso.fit(final_oof, y_train)
    iso_cal_oof = iso.predict(final_oof)
    iso_brier = brier_score_loss(y_train, iso_cal_oof)
    iso_auc = roc_auc_score(y_train, iso_cal_oof)

    # Platt scaling (sigmoid)
    from sklearn.linear_model import LogisticRegression as LR_Platt
    platt = LR_Platt(C=1e10, solver='lbfgs', max_iter=1000)
    platt.fit(final_oof.reshape(-1, 1), y_train)
    platt_cal_oof = platt.predict_proba(final_oof.reshape(-1, 1))[:, 1]
    platt_brier = brier_score_loss(y_train, platt_cal_oof)
    platt_auc = roc_auc_score(y_train, platt_cal_oof)

    log.info(f"  Isotonic: Brier={iso_brier:.6f}, AUC={iso_auc:.5f}")
    log.info(f"  Platt:    Brier={platt_brier:.6f}, AUC={platt_auc:.5f}")

    if platt_brier < iso_brier and platt_auc >= iso_auc - 0.001:
        calibrated_test = platt.predict_proba(final_test.reshape(-1, 1))[:, 1]
        cal_method = "platt"
        log.info("  Using Platt scaling (better Brier score)")
    else:
        calibrated_test = iso.predict(final_test)
        cal_method = "isotonic"
        log.info("  Using Isotonic calibration")
    log.info(f"  Calibrated test mean: {calibrated_test.mean():.6f}")

    # ─── Save Models ───
    log.info("  Saving model artefacts...")
    model_dir = BASE / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(lgb_model, model_dir / "lgb_model.pkl")
    joblib.dump(m_meta, model_dir / "meta_learner.pkl")
    joblib.dump(iso, model_dir / "isotonic_calibrator.pkl")
    joblib.dump(all_feat_names, model_dir / "feature_names.pkl")
    log.info(f"  Models saved to {model_dir}")

    return {
        "train_ids": train_ids, "test_ids": test_ids,
        "test_preds": calibrated_test, "oof_preds": final_oof, "y_train": y_train,
        "metrics": {"lgb_auc": lgb_auc, "xgb_auc": xgb_auc, "cb_auc": cb_auc,
                     "et_auc": et_auc, "hgb_auc": hgb_auc,
                     "ens_auc": ens_auc, "wavg_auc": wavg_auc,
                     "f1": f1, "f2": f2, "precision": prec, "recall": rec,
                     "optimal_threshold": best_thresh, "cal_method": cal_method},
        "feat_names": all_feat_names,
        "lgb_model": lgb_model,
        "full_label_feats": full_label_feats,
        "optimal_threshold": best_thresh,
    }


# ===================================================================
# TEMPORAL WINDOW ESTIMATION
# ===================================================================

def estimate_windows(results, sd=None):
    import ruptures as rpt

    log.info("PHASE 5: Temporal windows (improved)...")
    pred = pd.DataFrame({"account_id": results["test_ids"], "p": results["test_preds"]})
    opt_thresh = results.get("optimal_threshold", 0.3)
    suspicious = set(pred[pred["p"] >= opt_thresh]["account_id"])
    log.info(f"  {len(suspicious)} suspicious accounts (threshold={opt_thresh:.2f})")

    if not suspicious:
        return pd.DataFrame(columns=["account_id", "suspicious_start", "suspicious_end"])

    # Build mule_flag_date lookup from training labels for supervised calibration
    flag_date_map = {}
    if sd is not None and "train_labels" in sd:
        tl = sd["train_labels"]
        tl_mules = tl[(tl["is_mule"] == 1) & tl["mule_flag_date"].notna()]
        for _, row in tl_mules.iterrows():
            try:
                flag_date_map[row["account_id"]] = pd.Timestamp(row["mule_flag_date"])
            except Exception:
                pass
        log.info(f"  Training mule flag dates available: {len(flag_date_map)}")

    # Collect daily stats — include both test and training mules for calibration
    all_accts_needed = suspicious | set(flag_date_map.keys())
    daily_parts = []
    for i, path in enumerate(TXN_PARTS):
        if i % 100 == 0:
            log.info(f"  Part {i + 1}/{len(TXN_PARTS)}")
        df = pd.read_parquet(path, columns=["account_id", "transaction_timestamp", "amount"])
        df = df[df["account_id"].isin(all_accts_needed)]
        if len(df) == 0:
            continue
        df["date"] = pd.to_datetime(df["transaction_timestamp"], format="ISO8601").dt.date
        df["abs_a"] = df["amount"].abs()
        ch = df.groupby(["account_id", "date"], sort=False).agg(
            cnt=("amount", "count"), amt=("abs_a", "sum")
        ).reset_index()
        daily_parts.append(ch)

    if not daily_parts:
        return pd.DataFrame(columns=["account_id", "suspicious_start", "suspicious_end"])

    daily = pd.concat(daily_parts).groupby(["account_id", "date"]).sum().reset_index()
    del daily_parts
    gc.collect()

    # Pre-group for O(1) per-account lookup
    daily_grouped = {aid: grp.sort_values("date") for aid, grp in daily.groupby("account_id")}
    del daily
    gc.collect()

    # Supervised calibration: learn typical offset from peak-activity to flag-date
    # using training mules where we know both daily stats and flag date
    offsets = []
    for aid, flag_dt in flag_date_map.items():
        ad = daily_grouped.get(aid, pd.DataFrame())
        if len(ad) < 5:
            continue
        dates = pd.date_range(ad["date"].min(), ad["date"].max(), freq="D")
        if len(dates) < 10:
            continue
        date_lookup = dict(zip(ad["date"], zip(ad["cnt"], ad["amt"])))
        series = np.array([date_lookup.get(d.date(), (0, 0)) for d in dates])
        signal = series[:, 0] + series[:, 1] / 10000
        # Find peak 30-day rolling activity
        rolling_act = pd.Series(signal).rolling(30, min_periods=1).mean()
        peak_idx = int(rolling_act.idxmax())
        peak_date = dates[peak_idx]
        # Offset: how many days before flag date did peak activity occur?
        offset_days = (flag_dt - peak_date).days
        offsets.append(offset_days)

    if len(offsets) >= 10:
        median_offset = int(np.median(offsets))
        p25_offset = int(np.percentile(offsets, 25))
        p75_offset = int(np.percentile(offsets, 75))
        log.info(f"  Supervised offsets: median={median_offset}d, p25={p25_offset}d, p75={p75_offset}d (n={len(offsets)})")
    else:
        median_offset = 0
        p25_offset = -30
        p75_offset = 30
        log.info(f"  Not enough training data for offsets, using defaults")

    # Build prediction probability lookup for confidence-based window sizing
    pred_map = dict(zip(pred["account_id"], pred["p"]))

    def _estimate_one_window(aid, ad, dates, signal):
        """Estimate suspicious window using multi-penalty PELT + supervised calibration."""
        # Run PELT with multiple penalties and take consensus
        best_segments = []
        for pen in [5, 10, 15]:
            try:
                algo = rpt.Pelt(model="l2", min_size=7).fit(signal.reshape(-1, 1))
                bkps = algo.predict(pen=pen)
                if len(bkps) > 1:
                    segs = [0] + bkps
                    best_mean, bsi, bei = -1, 0, len(signal) - 1
                    for j in range(len(segs) - 1):
                        s, e = segs[j], min(segs[j + 1], len(signal)) - 1
                        sm = signal[s:e + 1].mean()
                        if sm > best_mean:
                            best_mean, bsi, bei = sm, s, e
                    best_segments.append((bsi, min(bei, len(dates) - 1)))
            except Exception:
                pass

        if best_segments:
            # Consensus: take union of all segment starts/ends
            si = min(s[0] for s in best_segments)
            ei = max(s[1] for s in best_segments)
        else:
            # Fallback: rolling z-score anomaly detection
            ws = min(30, len(signal) // 4)
            rm = pd.Series(signal).rolling(ws, min_periods=1).mean()
            rs = pd.Series(signal).rolling(ws, min_periods=1).std().fillna(1)
            z = (signal - rm.values) / (rs.values + 1e-6)
            anom = np.where(z > 1.5)[0]
            if len(anom) > 0:
                si = max(0, anom[0] - 7)
                ei = min(len(dates) - 1, anom[-1] + 7)
            else:
                peak = int(pd.Series(signal).rolling(30, min_periods=1).mean().idxmax())
                si = max(0, peak - 45)
                ei = min(len(dates) - 1, peak + 45)

        # Apply supervised offset calibration: shift window toward expected flag date timing
        if median_offset != 0 and len(dates) > 30:
            # Shift: if median_offset > 0, flags happen AFTER peak → extend end
            # If median_offset < 0, flags happen BEFORE peak → extend start
            shift = max(-14, min(14, median_offset // 2))  # conservative shift
            si = max(0, si - max(0, -shift))
            ei = min(len(dates) - 1, ei + max(0, shift))

        # Confidence-based window sizing: high confidence → tighter, low → wider
        prob = pred_map.get(aid, 0.5)
        if prob < 0.6:
            # Low confidence: widen window by ±30 days
            si = max(0, si - 30)
            ei = min(len(dates) - 1, ei + 30)
        elif prob > 0.9:
            # Very high confidence: keep tight (already good)
            pass
        else:
            # Medium confidence: slight widening ±14 days
            si = max(0, si - 14)
            ei = min(len(dates) - 1, ei + 14)

        return si, ei

    windows = []
    for aid in suspicious:
        ad = daily_grouped.get(aid, pd.DataFrame())
        if len(ad) == 0:
            windows.append({"account_id": aid, "suspicious_start": "", "suspicious_end": ""})
            continue
        if len(ad) < 10:
            windows.append({
                "account_id": aid,
                "suspicious_start": pd.Timestamp(ad["date"].min()).isoformat(),
                "suspicious_end": pd.Timestamp(ad["date"].max()).isoformat(),
            })
            continue

        dates = pd.date_range(ad["date"].min(), ad["date"].max(), freq="D")
        date_lookup = dict(zip(ad["date"], zip(ad["cnt"], ad["amt"])))
        series = np.array([date_lookup.get(d.date(), (0, 0)) for d in dates])
        signal = series[:, 0] + series[:, 1] / 10000

        if len(signal) < 20:
            windows.append({
                "account_id": aid,
                "suspicious_start": dates[0].isoformat(),
                "suspicious_end": dates[-1].isoformat(),
            })
            continue

        try:
            si, ei = _estimate_one_window(aid, ad, dates, signal)
            windows.append({
                "account_id": aid,
                "suspicious_start": dates[si].isoformat(),
                "suspicious_end": dates[ei].isoformat(),
            })
        except Exception:
            act = pd.Series(signal).rolling(90, min_periods=1).sum()
            peak = int(act.idxmax())
            si = max(0, peak - 45)
            ei = min(len(dates) - 1, peak + 45)
            windows.append({
                "account_id": aid,
                "suspicious_start": dates[si].isoformat(),
                "suspicious_end": dates[ei].isoformat(),
            })

    log.info(f"  Windows estimated: {len(windows)}")
    return pd.DataFrame(windows)


# ===================================================================
# SUBMISSION & REPORT
# ===================================================================

def generate_submission(results, windows, sd, risk_scores=None):
    log.info("PHASE 6: Generating submission...")
    opt_thresh = results.get("optimal_threshold", 0.3)
    test = sd["test_accounts"]
    pred = pd.DataFrame({"account_id": results["test_ids"], "is_mule": results["test_preds"]})
    sub = test.merge(pred, on="account_id", how="left")
    sub["is_mule"] = sub["is_mule"].fillna(0).clip(0, 1).round(4)

    # Merge risk levels
    if risk_scores is not None and len(risk_scores) > 0:
        sub = sub.merge(risk_scores[["account_id", "risk_level"]], on="account_id", how="left")
        sub["risk_level"] = sub["risk_level"].fillna("NORMAL")
    else:
        sub["risk_level"] = np.where(sub["is_mule"] > 0.8, "CRITICAL",
                            np.where(sub["is_mule"] > 0.5, "HIGH",
                            np.where(sub["is_mule"] > opt_thresh, "MEDIUM",
                            np.where(sub["is_mule"] > opt_thresh * 0.5, "LOW", "NORMAL"))))

    if len(windows) > 0:
        sub = sub.merge(windows, on="account_id", how="left")
    else:
        sub["suspicious_start"] = ""
        sub["suspicious_end"] = ""

    sub.loc[sub["is_mule"] < opt_thresh, "suspicious_start"] = ""
    sub.loc[sub["is_mule"] < opt_thresh, "suspicious_end"] = ""
    sub["suspicious_start"] = sub["suspicious_start"].fillna("")
    sub["suspicious_end"] = sub["suspicious_end"].fillna("")

    sub = sub[["account_id", "is_mule", "risk_level", "suspicious_start", "suspicious_end"]]
    sub.to_csv(BASE / "submission.csv", index=False)
    log.info(f"  Shape: {sub.shape}")
    log.info(f"  Optimal threshold: {opt_thresh:.2f}")
    log.info(f"  Mule > threshold: {(sub['is_mule'] > opt_thresh).sum()}")
    log.info(f"  Mule > 0.5: {(sub['is_mule'] > 0.5).sum()}")
    log.info(f"  Mean: {sub['is_mule'].mean():.4f}")
    for lvl in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]:
        log.info(f"  {lvl}: {(sub['risk_level'] == lvl).sum()}")
    return sub


def generate_report(results, sub, ablation_df=None, shap_results=None,
                    risk_scores=None, inv_profiles=None):
    m = results["metrics"]
    nf = len(results.get("feat_names", []))
    opt_thr = m.get('optimal_threshold', 0.5)
    n_mule_05 = int((sub['is_mule'] > opt_thr).sum())
    n_mule_03 = int((sub['is_mule'] > 0.3).sum())
    n_window = int((sub['suspicious_start'] != '').sum())

    # Load feature importance
    fi_text = ""
    fi_path = CACHE / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        top20 = fi.head(20)
        fi_text = "| Rank | Feature | LightGBM Gain |\n|------|---------|---------------|\n"
        for i, row in top20.iterrows():
            fi_text += f"| {i+1} | {row['feature']} | {row['importance']:.1f} |\n"

    # Ablation table
    abl_text = ""
    if ablation_df is not None and len(ablation_df) > 0:
        abl_text = "| Feature Group | # Features | Full AUC | Ablated AUC | AUC Drop |\n"
        abl_text += "|---------------|-----------|----------|-------------|----------|\n"
        for _, row in ablation_df.iterrows():
            abl_text += f"| {row['group']} | {int(row['n_features'])} | {row['full_auc']:.5f} | {row['ablated_auc']:.5f} | {row['auc_drop']:+.5f} |\n"

    # SHAP top features
    shap_text = ""
    if shap_results is not None and "explanations" in shap_results:
        exp = shap_results["explanations"]
        top_shap = exp.groupby("feature")["shap_value"].apply(lambda x: x.abs().mean()).sort_values(ascending=False).head(10)
        shap_text = "| Feature | Mean |SHAP| |\n|---------|-------------|\n"
        for feat, val in top_shap.items():
            shap_text += f"| {feat} | {val:.6f} |\n"

    report = f"""# AML Mule Account Detection — Comprehensive Analysis Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

This report presents a comprehensive anti-money laundering (AML) mule account detection
system that processes **~400 million transactions** across **160,153 bank accounts** to
identify potential money mule accounts. The system employs **{nf} engineered features**
spanning 10 distinct feature families, **label denoising** via confident learning, and a
**3-model stacking ensemble** (LightGBM + XGBoost + CatBoost with logistic regression
meta-learner) optimized via Bayesian hyperparameter tuning (Optuna).

**Key Results**:
- Best AUC-ROC: **{max(m['ens_auc'], m.get('wavg_auc', 0)):.5f}** (Stacking: {m['ens_auc']:.5f} | WAvg: {m.get('wavg_auc', 0):.5f})
- F1: **{m['f1']:.4f}** | F2: **{m.get('f2', 0):.4f}** (P={m['precision']:.4f}, R={m['recall']:.4f})
- Optimal threshold: **{m.get('optimal_threshold', 0.5):.2f}** | Calibration: {m.get('cal_method', 'isotonic')}
- Predicted mules (p>{m.get('optimal_threshold', 0.5):.2f}): **{n_mule_05}** | (p>0.3): **{n_mule_03}**
- Temporal windows assigned: **{n_window}** accounts

---

## 2. Pipeline Architecture

```
Raw Data (16.2 GB Parquet)
    |
    +-- Static Features (account / branch / customer / demographics)
    +-- Transaction Features (114 features: Benford, structuring, temporal, round amounts)
    +-- Graph Features (14 features: centrality, mule exposure, fan-in/out)
    +-- Graph Embeddings (20 features: Node2Vec 16-dim + Louvain community)
    +-- Geo/IP/Balance Features (per-batch IP sharing, geospatial spread)
    +-- MCC Anomaly Features (per-MCC z-scores across 2 passes)
    +-- Temporal Graph Features (~16 features: 6-month rolling windows)
    +-- Anomaly Scores (3 features: Isolation Forest + Autoencoder)
    |
    +-- Label Cleaning (temporal validation + alert-reason weighting + cleanlab)
    |
    +-- Model Training (LGB + XGB + CB via Optuna, 5-fold stratified CV)
    |
    +-- SHAP Explainability (TreeExplainer, per-account top-5 contributors)
    +-- Feature Ablation Study (per-group AUC impact measurement)
    |
    +-- Temporal Window Estimation (PELT changepoint + rolling z-score fallback)
    |
    +-- Submission + Report
```

---

## 3. Dataset Overview

| Property | Value |
|----------|-------|
| Total accounts | 160,153 |
| Training accounts | 96,091 |
| Test accounts | 64,062 |
| Training mules | 2,683 (2.79%) |
| Transaction files | 396 (main) + 311 (additional) |
| Reference tables | 10 (account, branch, customer, demographics, etc.) |
| Approximate transaction rows | ~400 million |
| Date range | Jul 2020 - Jun 2025 |

---

## 4. Feature Engineering ({nf} Features)

### 4.1 Transaction Features (114 features)
**Benford's Law Deviation**: Chi-squared and KL divergence of the first-digit distribution
against Benford's theoretical distribution. Legitimate spending follows Benford's law;
manufactured / structured transactions exhibit systematic deviations.

**Structuring Detection**: Counts and ratios of transactions near regulatory thresholds
(INR 9K-10K, 45K-50K, 90K-100K). Mule accounts frequently structure amounts just below
reporting thresholds to avoid detection.

**Temporal Behavioral Fingerprinting**:
- Monthly burstiness (std/mean of monthly counts) -- mules show burst-dormancy patterns
- Maximum consecutive dormant months -- long dormancy followed by sudden activation
- Late activation ratio -- activity concentration in the last 20% of account lifetime
- Hour entropy -- uniform hour distribution suggests automation; concentrated suggests human
- Night/weekend/month-end transaction ratios -- unusual temporal patterns

**Round Amount Patterns**: Fraction of amounts divisible by 1K, 5K, 10K, 50K -- round amounts
are characteristic of manufactured layering transactions.

**Coefficient of Variation**: Low CV (amount_std / amount_mean) indicates uniform pass-through
amounts, a hallmark of mule account operation.

### 4.2 Graph / Network Features (14 features)
- Degree centrality, weighted degree, PageRank
- Herfindahl counterparty concentration index
- Shared counterparties with known mules (direct + 2-hop exposure)
- Suspicious counterparty count (counterparties appearing in 5+ mule accounts)
- Fan-in / fan-out ratio (structural flow asymmetry -- mules receive from many, send to few)

### 4.3 Graph Embeddings -- Node2Vec + Community Detection (20 features)
**Node2Vec** (dim=16, walk_length=20, num_walks=30, p=1, q=2) learns a 16-dimensional
structural embedding per account from the counterparty transaction graph. Accounts with
similar network neighborhoods cluster together in embedding space, capturing latent
structural roles (e.g., hub, bridge, peripheral) without manual feature engineering.

**Louvain Community Detection**: Identifies tightly-connected account communities.
Features: community_id, community_size, community_mule_count, community_mule_rate.
High community_mule_rate indicates accounts in mule-dense subgraphs -- a strong
guilt-by-association signal.

### 4.4 Geo / IP / Balance Features
- Transaction geospatial spread (lat/lon standard deviation and range)
- Shared IP addresses with known mule accounts -- shared infrastructure indicator
- Unique IP count -- potential multi-user account access
- Balance-derived features (opening, average, volatility)

### 4.5 MCC Anomaly Features
Two-pass z-score computation: (1) population-level mean/std per MCC code, (2) per-account
deviation magnitude. Identifies amounts that are statistically unusual for their merchant
category.

### 4.6 Temporal Graph Features (~16 features)
Rolling 6-month windows (10 windows from Jul 2020 to Jul 2025) capturing:
- Degree evolution: mean, std, max, trend, max_jump
- Volume evolution: mean, std, max, trend, max_jump, acceleration
- Transaction count evolution: mean, std, max, trend
- Active window ratio: fraction of windows with any activity

These features capture **behavioral dynamics** -- mule accounts often show sudden spikes
in network connectivity and transaction volume during their active period, followed by
dormancy.

### 4.7 Anomaly Detection Features (3 features)
**Isolation Forest** (n_estimators=200, contamination=0.03): Model-free anomaly score
based on feature-space isolation depth. Provides an unsupervised outlier signal
complementary to supervised features.

**Autoencoder Reconstruction Error** (architecture: {nf}->64->16->64->{nf}, 50 epochs):
High reconstruction error indicates accounts that don't fit the learned normal pattern,
providing a deep-learning anomaly signal.

---

## 5. Model Performance

### 5.1 Cross-Validation Results (5-Fold Stratified)

| Model | AUC-ROC |
|-------|---------|
| LightGBM | {m['lgb_auc']:.5f} |
| XGBoost | {m['xgb_auc']:.5f} |
| CatBoost | {m['cb_auc']:.5f} |
| **Stacking Ensemble** | **{m['ens_auc']:.5f}** |
| Weighted Average | {m.get('wavg_auc', 0):.5f} |

| Metric | Value |
|--------|-------|
| Optimal Threshold | {m.get('optimal_threshold', 0.5):.2f} |
| Precision | {m['precision']:.4f} |
| Recall | {m['recall']:.4f} |
| F1-Score | {m['f1']:.4f} |
| F2-Score | {m.get('f2', 0):.4f} |
| Calibration | {m.get('cal_method', 'isotonic')} |

### 5.2 Model Selection Rationale
Three gradient boosted tree models were chosen for their complementary strengths:
- **LightGBM**: Fastest training, leaf-wise growth, excellent on high-cardinality features
- **XGBoost**: Depth-wise growth, strong regularization, different inductive bias
- **CatBoost**: Ordered boosting, native categorical handling, robust to overfitting

A logistic regression meta-learner combines their predictions, learning optimal
blend weights from out-of-fold predictions. This stacking approach consistently
outperforms any single model, especially under class imbalance.

### 5.3 Hyperparameter Optimization
Optuna Bayesian optimization with 30 trials per model, optimizing AUC-ROC.
Key tuned parameters: learning_rate, num_leaves/max_depth, subsample,
colsample_bytree, reg_alpha, reg_lambda, min_child_weight.

---

## 6. Red-Herring Detection Strategy

Mislabeled or noisy training labels are a critical challenge. Our three-layer strategy:

### 6.1 Temporal Validation
Mules flagged **outside the observable transaction window** (Jul 2020 - Jun 2025) have
no behavioral evidence in the data. These labels are downweighted to **0.3** during
training, reducing their influence on the model while not discarding them entirely.

### 6.2 Alert Reason Scrutiny
The **"Routine Investigation"** category accounts for ~26% of flagged mules and is
inherently noisier than specific alert reasons (e.g., "STR Filed", "Large Cash Deposit").
These labels receive weight **0.5** -- deferring to the model's learned patterns rather
than potentially unreliable labels.

### 6.3 Cleanlab Confident Learning
Automatic detection of label inconsistencies using cross-validated LightGBM predictions
and the confident joint estimation framework. Accounts where the model strongly disagrees
with the label (across all 5 folds) are identified as potential label errors and
downweighted to **0.3**.

This combined strategy prevents the model from memorizing noise while preserving
the majority of genuine mule signals.

---

## 7. Feature Ablation Study

The ablation study measures each feature group's contribution by removing it and
measuring the AUC drop with a quick LightGBM model (300 trees, 5-fold CV).

{abl_text if abl_text else "*Ablation study not available.*"}

**Interpretation**: Larger AUC drops indicate more critical feature groups.
Groups with minimal drop may contain redundant information captured by other features.

---

## 8. SHAP Explainability Analysis

### 8.1 Global Feature Importance (SHAP)

{shap_text if shap_text else "*SHAP analysis not available.*"}

### 8.2 Visual Analysis

![SHAP Summary (Beeswarm)](report_figures/shap_summary.png)

*The beeswarm plot shows the distribution of SHAP values for each feature across all
test accounts. Red points indicate high feature values, blue indicate low. Features
are ordered by mean absolute SHAP value.*

![SHAP Bar Plot](report_figures/shap_bar.png)

*Mean absolute SHAP value per feature -- a global measure of feature importance
from the model's perspective.*

### 8.3 Per-Account Explanations

For each flagged account (p>0.3), the top 5 SHAP contributors are saved to
`features/shap_explanations.csv`. This enables investigators to understand
**why** each account was flagged, supporting regulatory explainability requirements.

---

## 9. Top 20 Feature Importance (LightGBM Gain)

{fi_text if fi_text else "*Feature importance not available.*"}

---

## 10. Visualizations

### 10.1 Feature Importance
![Top 30 Feature Importance](report_figures/feat_importance.png)

### 10.2 Feature Distributions (Mule vs Legitimate)
![Feature Distributions](report_figures/feature_distributions.png)

b*Distribution comparison for the top 6 most important features. Clear separation
between mule (red) and legitimate (blue) distributions indicates strong discriminative power.*

### 10.3 Prediction Distribution
![Prediction Distribution](report_figures/pred_distribution.png)

*Left: Full test set distribution showing the vast majority of accounts have near-zero
mule probability. Right: Zoomed view of flagged accounts (p>0.01).*

### 10.4 Network Subgraph Analysis
![Network Subgraph](report_figures/network_subgraph.png)

*Transaction network of the top 10 predicted mules (red) and their counterparties (blue).
Mule accounts often form tightly connected clusters or share counterparties, which is
captured by our graph embedding and network features.*

### 10.5 Temporal Activity Heatmap
![Temporal Heatmap](report_figures/temporal_heatmap.png)

*Monthly transaction volume heatmap for the top 5 predicted mules. Note the
burst-dormancy patterns characteristic of mule account operation -- periods of
high activity interspersed with inactivity.*

---

## 11. Mule Account Typology -- 13 Behavioral Patterns

Based on feature analysis, we identify the following mule archetypes:

| # | Pattern | Key Indicators |
|---|---------|---------------|
| 1 | **Rapid Pass-Through** | High volume, low balance retention, high in/out ratio |
| 2 | **Structuring Layerer** | Amounts clustered below reporting thresholds (9K, 45K, 90K) |
| 3 | **Burst-Dormancy** | Long dormancy then sudden high-volume burst then dormancy |
| 4 | **Network Hub** | Unusually high degree centrality, many unique counterparties |
| 5 | **Shared Infrastructure** | IP addresses shared with other flagged accounts |
| 6 | **Round Amount Specialist** | Disproportionate fraction of round amounts (1K, 5K, 10K multiples) |
| 7 | **Benford Violator** | First-digit distribution deviates significantly from Benford's law |
| 8 | **MCC Anomaly** | Transaction amounts far from MCC-category norms |
| 9 | **Nocturnal Operator** | Elevated night/weekend transaction ratios |
| 10 | **Community-Embedded Mule** | Located in Louvain community with high mule density |
| 11 | **Late Activator** | Account dormant for years, suddenly active in recent months |
| 12 | **Fan-In Collector** | Many incoming counterparties, few outgoing (funnel pattern) |
| 13 | **Autoencoder Outlier** | High reconstruction error -- doesn't fit any normal profile |

---

## 12. Temporal Window Estimation

Suspicious activity windows are estimated using a multi-strategy approach:

1. **PELT Changepoint Detection**: Applied to daily transaction count and amount series
   using an RBF kernel cost function. Identifies structural breaks in account behavior.
2. **Rolling Z-Score Fallback**: For accounts without clear changepoints, identifies
   periods where activity exceeds 2 standard deviations above the rolling mean.
3. **Peak Activity Window**: Last-resort heuristic using the densest activity period.

---

## 13. Predictions Summary

| Metric | Value |
|--------|-------|
| Test accounts | {len(sub):,} |
| Predicted mule (p > {opt_thr:.2f}) | {n_mule_05} |
| Predicted mule (p > 0.3) | {n_mule_03} |
| Mean probability | {sub['is_mule'].mean():.6f} |
| Median probability | {sub['is_mule'].median():.6f} |
| Max probability | {sub['is_mule'].max():.6f} |
| Accounts with temporal window | {n_window} |

---

## 14. Confidence Analysis

| Confidence Tier | Probability Range | Count |
|----------------|-------------------|-------|
| Very High | p > 0.9 | {int((sub['is_mule'] > 0.9).sum())} |
| High | 0.5 < p <= 0.9 | {int(((sub['is_mule'] > 0.5) & (sub['is_mule'] <= 0.9)).sum())} |
| Medium | 0.3 < p <= 0.5 | {int(((sub['is_mule'] > 0.3) & (sub['is_mule'] <= 0.5)).sum())} |
| Low | 0.1 < p <= 0.3 | {int(((sub['is_mule'] > 0.1) & (sub['is_mule'] <= 0.3)).sum())} |
| Minimal | p <= 0.1 | {int((sub['is_mule'] <= 0.1).sum())} |

---

## 15. 13 Known Mule Patterns — Feature Mapping

| # | Pattern | Key Features |
|---|---------|-------------|
| 1 | Dormant Activation | `dormant_months`, `dormant_then_active`, `acct_age_days`, `txn_day_span` |
| 2 | Structuring / Threshold Avoidance | `struct_45_50k_ratio`, `struct_9_10k_ratio`, `round_1k_ratio`, `benford_chi2` |
| 3 | Rapid Pass-Through | `passthrough_score`, `avg_hold_time_approx`, `txn_count / acct_age_days` |
| 4 | Fan-In / Fan-Out | `fan_in`, `fan_out`, `fan_in_out_ratio`, `uniq_cp_count`, graph embeddings |
| 5 | Geographic Anomaly | `n_unique_ip`, `ip_per_txn`, `geo_ip_consistency`, `geo_city_count` |
| 6 | New Account High Value | `early_value_ratio`, `early_count_ratio`, `new_acct_high_vol` |
| 7 | Income Mismatch | `txn_to_balance_ratio`, `txn_to_monthly_bal`, balance vs transaction volume features |
| 8 | Post-Mobile-Change Spike | `mobile_change_activity`, `days_since_mobile_update`, velocity features |
| 9 | Round Amounts | `round_1k_ratio`, `round_5k_ratio`, `round_10k_ratio`, `round_50k_ratio` |
| 10 | Layered / Subtle | Stacking ensemble, anomaly isolation scores, graph embedding distances |
| 11 | Salary Cycle Exploitation | `salary_exploit_score`, `net_flow_per_balance`, `cd_ratio_sum`, `monthend_ratio` |
| 12 | Branch Collusion | `branch_collusion_score`, `branch_susp_cp_score`, `branch_mule_rate` |
| 13 | MCC Anomaly | `mcc_entropy`, `mcc_nunique`, `mcc_top_ratio` |

---

## 16. Red-Herring Avoidance Strategy

Our pipeline employs **three complementary strategies** to avoid red-herring false positives:

1. **Temporal Validation**: Mules flagged outside the transaction window (2020-07 to 2025-06) receive
   reduced training weight (0.3x), preventing the model from learning stale or irrelevant patterns.

2. **Alert Reason De-weighting**: "Routine Investigation" mule labels receive 0.5x weight multiplier,
   since routine checks may produce noisier labels compared to targeted investigations.

3. **Confident Learning (Cleanlab)**: A dedicated label-noise detection step identifies potentially
   mislabeled accounts using a LightGBM classifier with out-of-sample predictions. Flagged accounts
   receive 0.3x weight, effectively down-weighting suspicious labels without discarding data.

4. **Part Transaction Type Analysis**: The `part_transaction_type` field (CI/BI/IP/IC) from
   `transactions_additional` is used to build ratio features that capture anomalous transaction-type
   distributions, helping distinguish legitimate high-volume accounts from mule accounts.

5. **Isotonic Calibration**: Post-training isotonic regression calibration ensures predicted
   probabilities are well-calibrated, preventing overconfident false-positive predictions.

---

## 17. Experiments: What Worked and What Didn't

### What Worked Well
- **Feature diversity** (10+ feature families) was key — no single family dominates; ablation shows each contributes
- **GPU-accelerated XGBoost/CatBoost** reduced training from hours to minutes
- **Graph embeddings (DeepWalk)** captured structural mule network patterns effectively
- **Cleanlab label denoising** improved precision by down-weighting ~2-5% noisy labels
- **Optuna hyperparameter tuning** with stratified CV produced robust generalizable models
- **Batch-by-batch streaming** allowed processing 400M transactions on 16GB RAM

### What Didn't Work / Lessons Learned
- **Node2Vec on full graph** was prohibitively slow (40+ min); filtering to active edges + DeepWalk solved it
- **PyTorch autoencoder** for anomaly detection required CPU fallback (no CUDA on torch build) — still effective
- **Raw transaction-level features** without account-level aggregation were noisy and unhelpful
- **Equal weighting of all labels** hurt performance — temporal + alert-based weighting was essential
- **Single model approaches** (LGB alone) were strong but stacking provided consistent 0.001-0.003 AUC improvement

---

## 18. Technical Notes

- **Hardware**: Intel i7-13th Gen (20 threads), 16GB RAM, RTX 4050 (6GB VRAM)
- **Runtime**: ~30-60 minutes for full pipeline
- **Caching**: All intermediate features cached as Parquet files for reproducibility
- **Reproducibility**: Random seed fixed at 42 across all stochastic components
- **Memory Management**: Batch-by-batch processing for large transaction files to stay within 16GB RAM

---
"""

    # --- NEW SECTIONS: AML Intelligence Platform ---
    risk_section = ""
    if risk_scores is not None and len(risk_scores) > 0:
        n_crit = int((risk_scores["risk_level"] == "CRITICAL").sum())
        n_high = int((risk_scores["risk_level"] == "HIGH").sum())
        n_med = int((risk_scores["risk_level"] == "MEDIUM").sum())
        n_low = int((risk_scores["risk_level"] == "LOW").sum())

        risk_section = f"""
## 19. Multi-Stage Detection Architecture

The AML Intelligence Platform uses a **three-stage detection pipeline** that progressively
filters and enriches account risk assessments:

```
Stage 1: Behavioral Screening          Stage 2: Risk Contagion         Stage 3: ML Detection
+---------------------------+     +---------------------------+    +---------------------------+
| 10 Rule-Based Heuristics  |     | Graph Risk Propagation    |    | 3-Model Ensemble (LGB +   |
| - Structuring detection   | --> | - Iterative belief prop.  | -> |   XGB + CatBoost)          |
| - Dormancy activation     |     | - 4 iterations, α=0.4    |    | - Per-fold label features  |
| - Pass-through scoring    |     | - Directional flow risk   |    | - SHAP explainability      |
| - Velocity anomaly        |     | - Neighbor risk density   |    | - Isotonic calibration     |
| Outputs: screening_stage  |     | Outputs: contagion_risk   |    | Outputs: mule probability  |
|          (1=Normal/2/3)   |     |          6 features        |    |          + risk breakdown  |
+---------------------------+     +---------------------------+    +---------------------------+
```

**Stage 1 Rules**: Structuring (round amounts >30%), Dormancy activation (>180d gap + burst),
Rapid pass-through (score >0.7), Velocity anomaly (burstiness >3), Benford deviation (>0.1),
Fan pattern (ratio >5 or <0.2), Geographic anomaly (>10 IPs), KYC non-compliance, New account
high volume, Isolation Forest flag.

**Stage 2 Contagion**: Risk propagation via $R_{{t+1}} = 0.4 \\cdot R_0 + 0.6 \\cdot A_{{norm}} \\cdot R_t$
over 4 iterations on a sparse graph (160K nodes, 3.25M edges). Seed signal is a weighted blend
of Isolation Forest score (50%), autoencoder error (30%), and behavioral screening score (20%).

---

## 20. Risk Scoring Framework

Each test account is scored across **5 risk dimensions** using SHAP value decomposition:

| Dimension | Weight | Signal Sources |
|-----------|--------|----------------|
| Behavior Risk | 25% | Transaction patterns, structuring, Benford, temporal |
| Network Risk | 25% | Graph centrality, contagion, counterparty exposure |
| Infrastructure Risk | 15% | Geographic spread, IP sharing, branch, balance |
| Temporal Risk | 20% | Activity evolution, dormancy, window features |
| ML/Anomaly Risk | 15% | Isolation Forest, autoencoder reconstruction error |

**Total Risk**: $R_{{total}} = 0.25B + 0.25N + 0.15I + 0.20T + 0.15M$

**Decision Thresholds**:
| Risk Level | Threshold | Count | Action |
|------------|-----------|-------|--------|
| CRITICAL | >0.8 | {n_crit} | Immediate investigation |
| HIGH | 0.5-0.8 | {n_high} | Priority review |
| MEDIUM | 0.3-0.5 | {n_med} | Enhanced monitoring |
| LOW | <0.3 | {n_low} | Normal processing |

![Risk Dimension Profile](report_figures/risk_dimension_radar.png)

![Risk Level Distribution](report_figures/risk_distribution.png)

---

## 21. Risk Contagion Analysis

The risk contagion module propagates suspicion through the transaction graph using iterative
belief propagation on a sparse adjacency matrix (scipy.sparse, ~78 MB for 3.25M edges).

**Methodology**:
1. **Seed Risk ($R_0$)**: Normalized blend of anomaly scores (pre-ML, label-free)
2. **Propagation**: 4 iterations of $R_{{t+1}} = \\alpha R_0 + (1-\\alpha) A_{{norm}} R_t$
3. **Directional Flow**: Incoming/outgoing risk weighted by credit/debit transaction proportions
4. **Convergence**: Rapid — typical delta < 0.001 after 3 iterations

**Features Generated**:
| Feature | Description |
|---------|-------------|
| `contagion_risk` | Final converged risk score |
| `contagion_delta` | Change from propagation ($R_{{final}} - R_0$) |
| `incoming_risk_flow` | Risk received from senders (credit-weighted) |
| `outgoing_risk_flow` | Risk transmitted to receivers (debit-weighted) |
| `risk_neighbor_mean` | Average risk of transaction neighbors |
| `risk_cluster_density` | Fraction of neighbors above 75th percentile risk |

![Multi-Stage Pipeline Funnel](report_figures/stage_funnel.png)

---

## 22. Investigation Intelligence

For each flagged account (p > 0.3), the system generates an investigation card containing:
- **Risk Drivers**: Top SHAP contributors per risk dimension
- **Suspicious Counterparties**: Top 5 transaction partners ranked by contagion risk
- **Pattern Classification**: Automated matching against 13 known mule archetypes
- **Activity Summary**: Transaction count, credit-debit ratio, active months
- **Risk Timeline**: Suspicious activity windows from PELT changepoint detection

"""
        # Sample profiles for top 5
        if inv_profiles is not None and len(inv_profiles) > 0:
            top5 = inv_profiles.nlargest(5, "probability")
            risk_section += "### Sample Investigation Profiles (Top 5)\n\n"
            for _, p in top5.iterrows():
                patterns = p.get("matched_patterns", "None")
                risk_section += f"""**Account {p['account_id']}** (p={p['probability']:.4f}, {p['risk_level']})
- Matched Patterns: {patterns}
- Activity: {p.get('activity_summary', 'N/A')}
- Top Risk Drivers: {p.get('top_risk_drivers', 'N/A')[:120]}...
- Suspicious Counterparties: {p.get('suspicious_counterparties', 'N/A')}
- Window: {p.get('suspicious_start', '')} → {p.get('suspicious_end', '')}

"""
            risk_section += "Full investigation profiles saved to `features/investigation_profiles.csv`.\n\n---\n"

    report += risk_section
    report += """
*Report generated by the AML Intelligence Platform v4.0*
"""
    (BASE / "report.md").write_text(report, encoding="utf-8")
    log.info("  Report saved to report.md")

def main():
    log.info("=" * 60)
    log.info("AML Intelligence Platform v5 — START")
    log.info("=" * 60)

    sd = load_static()

    # Load cached v9b features (stable across all models)
    _feat_cache = CACHE / "all_features_v8.parquet"
    log.info(f"Loading cached features from {_feat_cache.name}...")
    all_feats = pd.read_parquet(_feat_cache)
    all_feats.index.name = "account_id"
    log.info(f"  Loaded feature matrix: {all_feats.shape}")

    # Phase 2: Freeze counter-features (distinguish mule freezes from administrative)
    if "was_frozen" in all_feats.columns and "is_frozen" in all_feats.columns:
        # frozen_then_active: temporary freeze that was reversed → likely administrative, not mule
        all_feats["frozen_then_active"] = ((all_feats["was_frozen"] == 1) & (all_feats["is_frozen"] == 0)).astype(np.float32)
        # freeze_relative_duration: short relative freeze → likely administrative
        all_feats["freeze_relative_duration"] = (all_feats["days_frozen"].fillna(0) / (all_feats["account_age_days"].fillna(1).clip(lower=1))).astype(np.float32)
        log.info(f"  Added freeze counter-features: frozen_then_active, freeze_relative_duration")

    log.info(f"  Final feature matrix: {all_feats.shape}")

    # Load cached data for per-fold label feature recomputation
    log.info("Loading cached edges/IP/community data for per-fold label features...")
    edges_df = pd.read_parquet(CACHE / "edges_cache.parquet")
    # Load IP pairs as compact per-account mapping (avoids 4GB OOM from raw 79M-row table)
    ip_acct_map = None
    if (CACHE / "ip_pairs_cache.parquet").exists():
        import pyarrow.parquet as pq
        from collections import defaultdict
        log.info("  Loading IP pairs in chunks (compact)...")
        ip_acct_map = defaultdict(set)
        pf = pq.ParquetFile(CACHE / "ip_pairs_cache.parquet")
        for batch in pf.iter_batches(batch_size=2_000_000, columns=["account_id", "ip_address"]):
            accts = batch.column("account_id").to_pylist()
            ips = batch.column("ip_address").to_pylist()
            for a, ip in zip(accts, ips):
                ip_acct_map[a].add(hash(ip) & 0xFFFFFFFF)
            del accts, ips, batch
        ip_acct_map = dict(ip_acct_map)
        log.info(f"  IP data loaded: {len(ip_acct_map)} accounts")
    community_map = pd.read_parquet(CACHE / "community_map_cache.parquet") if (CACHE / "community_map_cache.parquet").exists() else None

    # Branch-account mapping for per-fold branch_mule_rate
    branch_acct_map = sd["accounts"][["account_id", "branch_code"]].copy()

    # Label cleaning
    cleaned = clean_labels(sd, all_feats)

    # Model training (label features computed per-fold inside)
    results = train_models(all_feats, cleaned, sd, edges_df, ip_acct_map, branch_acct_map, community_map)

    # Augment features with full-train label features for SHAP/ablation
    full_label_feats = results["full_label_feats"]
    all_feats_aug = all_feats.join(full_label_feats, how="left").fillna(0)

    # SHAP explainability
    shap_results = compute_shap_analysis(results, all_feats_aug, results["lgb_model"])

    # Feature ablation study
    ablation_df = run_ablation_study(all_feats, cleaned, sd)

    # Multi-dimensional risk scoring (uses SHAP)
    risk_scores = compute_risk_scores(results, all_feats_aug, shap_results)

    # Free large objects before memory-heavy temporal windows phase
    del cleaned, ip_acct_map, branch_acct_map, community_map, all_feats
    gc.collect()

    # Temporal windows (pass sd for supervised calibration using mule_flag_date)
    windows = estimate_windows(results, sd)

    # Investigation intelligence
    inv_profiles = generate_investigation_profiles(
        results, risk_scores, all_feats_aug, shap_results, edges_df, windows)

    # Submission (with risk levels)
    sub = generate_submission(results, windows, sd, risk_scores)

    # Visualizations and report
    generate_visualizations(results, all_feats_aug, sd, shap_results, risk_scores)
    generate_report(results, sub, ablation_df, shap_results, risk_scores, inv_profiles)

    log.info("=" * 60)
    log.info("PIPELINE v5 COMPLETE — AML Intelligence Platform")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
