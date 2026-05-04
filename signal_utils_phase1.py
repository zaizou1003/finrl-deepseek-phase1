# signal_utils_phase1.py

from __future__ import annotations

import numpy as np
import pandas as pd

from config_phase1 import Phase1Config


CFG = Phase1Config()


def fill_signal_columns(
    df: pd.DataFrame,
    sentiment_col: str = CFG.SENTIMENT_COL,
    risk_col: str = CFG.RISK_COL,
) -> pd.DataFrame:
    """Fill missing LLM columns with neutral values."""
    out = df.copy()
    if sentiment_col in out.columns:
        out[sentiment_col] = out[sentiment_col].fillna(CFG.FILL_SENTIMENT)
    if risk_col in out.columns:
        out[risk_col] = out[risk_col].fillna(CFG.FILL_RISK)
    return out


def normalize_signal(
    x: np.ndarray | pd.Series | float,
    neutral: float,
) -> np.ndarray:
    """
    Map original 1-5 scale to [-1, 1]:
    1 -> -1, 3 -> 0, 5 -> 1
    """
    arr = np.asarray(x, dtype=np.float32)
    return (arr - neutral) / 2.0


def sent_z(sentiment: np.ndarray | pd.Series | float) -> np.ndarray:
    return normalize_signal(sentiment, CFG.SENTIMENT_NEUTRAL)


def risk_z(risk: np.ndarray | pd.Series | float) -> np.ndarray:
    return normalize_signal(risk, CFG.RISK_NEUTRAL)


def compute_confidence(
    sent_z_value: np.ndarray | float,
    risk_z_value: np.ndarray | float,
    gamma: float = CFG.GAMMA,
) -> np.ndarray:
    """
    Confidence proxy:
    strong sentiment -> higher confidence
    high positive risk -> lower confidence
    """
    s = np.asarray(sent_z_value, dtype=np.float32)
    r = np.asarray(risk_z_value, dtype=np.float32)
    conf = np.abs(s) * (1.0 - gamma * np.maximum(r, 0.0))
    return np.clip(conf, 0.0, 1.0)


def normalize_turbulence(
    turbulence_value: float,
    turbulence_threshold: float,
) -> float:
    """
    Simple turbulence normalization.
    If turbulence is missing or threshold invalid, return 0.
    """
    if turbulence_threshold is None or turbulence_threshold <= 0:
        return 0.0
    if turbulence_value is None or np.isnan(turbulence_value):
        return 0.0
    return float(max(turbulence_value, 0.0) / turbulence_threshold)


def compute_regime_factor(
    turbulence_value: float,
    turbulence_threshold: float,
    lambda_reg: float = CFG.LAMBDA_REG,
) -> float:
    """
    Higher turbulence => lower trust in LLM signal.
    Returns a scalar in (0, 1].
    """
    turb_norm = normalize_turbulence(turbulence_value, turbulence_threshold)
    return float(1.0 / (1.0 + lambda_reg * turb_norm))


def compute_alpha_side(
    sent_z_value: np.ndarray | float,
    alpha_pos: float = CFG.ALPHA_POS,
    alpha_neg: float = CFG.ALPHA_NEG,
) -> np.ndarray:
    """
    Negative news gets stronger impact than positive news.
    """
    s = np.asarray(sent_z_value, dtype=np.float32)
    return np.where(s < 0.0, alpha_neg, alpha_pos).astype(np.float32)


def compute_alpha_eff(
    sent_z_value: np.ndarray | float,
    risk_z_value: np.ndarray | float,
    turbulence_value: float,
    turbulence_threshold: float,
    gamma: float = CFG.GAMMA,
    lambda_reg: float = CFG.LAMBDA_REG,
    alpha_pos: float = CFG.ALPHA_POS,
    alpha_neg: float = CFG.ALPHA_NEG,
) -> np.ndarray:
    conf = compute_confidence(sent_z_value, risk_z_value, gamma=gamma)
    regime = compute_regime_factor(
        turbulence_value=turbulence_value,
        turbulence_threshold=turbulence_threshold,
        lambda_reg=lambda_reg,
    )
    alpha_side = compute_alpha_side(
        sent_z_value,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
    )
    return alpha_side * conf * regime


def modulate_action(
    base_action: np.ndarray,
    sent_z_value: np.ndarray,
    risk_z_value: np.ndarray,
    turbulence_value: float,
    turbulence_threshold: float,
    gamma: float = CFG.GAMMA,
    lambda_reg: float = CFG.LAMBDA_REG,
    alpha_pos: float = CFG.ALPHA_POS,
    alpha_neg: float = CFG.ALPHA_NEG,
) -> np.ndarray:
    """
    Phase-1 action modulation:
        action_mod = action * (1 + alpha_eff * tanh(sent_z))
    """
    action = np.asarray(base_action, dtype=np.float32)
    s = np.asarray(sent_z_value, dtype=np.float32)
    r = np.asarray(risk_z_value, dtype=np.float32)

    alpha_eff = compute_alpha_eff(
        sent_z_value=s,
        risk_z_value=r,
        turbulence_value=turbulence_value,
        turbulence_threshold=turbulence_threshold,
        gamma=gamma,
        lambda_reg=lambda_reg,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
    )

    return action * (1.0 + alpha_eff * np.tanh(s))