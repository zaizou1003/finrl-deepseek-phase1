# config_phase1.py

from dataclasses import dataclass


@dataclass
class Phase1Config:
    # Neutral anchors on the original 1-5 scale
    SENTIMENT_NEUTRAL: float = 3.0
    RISK_NEUTRAL: float = 3.0

    # Missing-value handling
    FILL_SENTIMENT: float = 3.0
    FILL_RISK: float = 3.0

    # Nonlinear asymmetric modulation
    ALPHA_POS: float = 0.010
    ALPHA_NEG: float = 0.015

    # Confidence penalty from high risk
    GAMMA: float = 0.5

    # Regime sensitivity
    LAMBDA_REG: float = 1.0

    # Numerical safety
    EPS: float = 1e-8

    # Column names
    SENTIMENT_COL: str = "llm_sentiment"
    RISK_COL: str = "llm_risk"
    TURBULENCE_COL: str = "turbulence"

    # Output / bookkeeping
    MODEL_TAG: str = "cppo_deepseek_phase1"