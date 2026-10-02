"""Scores for probabilistic and positional predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6
CAL_BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def binary_scores(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = {"n": int(len(y)), "positives": int(y.sum()), "log_loss": log_loss(y, p), "brier": brier(y, p)}
    if 0 < y.sum() < len(y):
        out["auc"] = float(roc_auc_score(y, p))
        out["avg_precision"] = float(average_precision_score(y, p))
    else:
        out["auc"] = out["avg_precision"] = float("nan")
    return out


def calibration_table(y: np.ndarray, p: np.ndarray, bins=CAL_BINS) -> pd.DataFrame:
    """Predicted vs observed rate per probability bin."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.cut(df["p"], bins=list(bins), include_lowest=True)
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return g.reset_index()


def regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
    }


def position_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "n": int(len(y_true)),
        "exact": float(np.mean(err == 0)),
        "within_1": float(np.mean(err <= 1)),
        "mae": float(np.mean(err)),
    }
