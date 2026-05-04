# FinRL-DeepSeek Phase 1: LLM-Generated Sentiment and Risk Signals

This repository contains the code and outputs for a term paper project on risk-aware reinforcement learning trading with LLM-generated sentiment and risk signals.

The project extends the FinRL-DeepSeek setup by introducing a Phase 1 CPPO environment that uses:
- neutral missing-value handling for LLM sentiment and risk scores,
- sentiment/risk normalization from a 1–5 scale to [-1, 1],
- confidence weighting,
- turbulence-aware attenuation,
- asymmetric positive/negative sentiment weighting,
- nonlinear action modulation using tanh.

## Main files

- `config_phase1.py`: Phase 1 configuration constants.
- `signal_utils_phase1.py`: LLM signal preprocessing and action modulation utilities.
- `env_stocktrading_llm_risk_phase1.py`: Phase 1 stock trading environment.
- `train_cppo_llm_risk_phase1.py`: CPPO training script for the Phase 1 environment.
- `FinRL_DeepSeek_backtest_v3.ipynb`: Backtesting and visualization notebook.
- `metrics_all.csv`: Deterministic backtest metrics.
- `portfolio_values_all.csv`: Portfolio values from deterministic backtests.

## Figures

The main report figures are:

- `plot_portfolio_main_comparison.png`
- `plot_drawdown_main_comparison.png`
- `plot_metrics_main_comparison.png`
- `plot_rolling_sharpe_main_comparison.png`

## Notes

Large raw datasets, training logs, and generated result folders are excluded from this repository to keep it lightweight. The three Phase 1 checkpoints used in the report are included under `checkpoints_phase1/`.
The dataset used in this project is the FinRL-DeepSeek Nasdaq dataset from Hugging Face: `benstaf/nasdaq_2013_2023`.

## Author

Ahmed Aziz Ben Aissa  
Aivancity School for Technology, Business & Society  
Course: AI in Finance
