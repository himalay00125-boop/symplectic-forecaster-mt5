# Symplectic Forecaster MT5 (Version 2.0)

A high-frequency algorithmic trading system designed for MetaTrader 5, featuring a state-of-the-art Deep Learning and Reinforcement Learning microservice architecture.

## Overview

Symplectic Forecaster v2.0 transitions from a lightweight monolithic script into a fully decoupled, multi-process AI engine. It separates trade execution from heavy machine learning computations to ensure zero latency during live market operations.

The system relies on three core components:
1. **Execution Engine (MT5):** Handles real-time tick data, phase-space feature generation, and lightning-fast order execution.
2. **AI Engine (PyTorch & RL):** A background worker utilizing a PyTorch Long Short-Term Memory (LSTM) sequence model for true multi-horizon prediction, and a StableBaselines3 Proximal Policy Optimization (PPO) agent for counterfactual offline policy correction.
3. **Causal Optimizer (DoWhy & Optuna):** A background worker that continuously runs DoWhy causal discovery to prune non-causal features dynamically, alongside an Optuna hyperparameter study that optimizes the system in real-time.

## Key Features

* **Microservice Architecture:** Utilizes native Python multiprocessing to separate the execution thread from heavy AI computations, preventing blocking during live trading.
* **True Multi-Horizon Loss:** The PyTorch LSTM model optimizes across multiple forward horizons simultaneously to maximize holding stability.
* **Counterfactual Reinforcement Learning:** When a Stop Loss is hit, the system synthesizes the true adverse return and sends an Inverse Propensity Score penalty to the RL agent, correcting its policy offline.
* **Invariant Causal Prediction:** The causal optimizer tracks feature correlations across different market regimes (NORMAL vs ALERT) and mathematically penalizes features that exhibit high cross-regime variance.
* **Online Bayesian Optimization:** A continuous background Optuna study explores hyperparameter permutations, adapting the model to shifting market dynamics without interrupting execution.

## Installation

### Prerequisites
* Python 3.10 or higher
* MetaTrader 5 Terminal (installed and running)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/himalay00125-boop/symplectic-forecaster-mt5.git
   cd symplectic-forecaster-mt5
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Architecture Usage

Because Version 2.0 employs a multi-process architecture, the main script automatically spawns the required background workers upon initialization.

Run the main execution script:
```bash
python symplectic_forecaster.py
```

*Note: Ensure your hardware supports PyTorch (CUDA recommended for optimal performance).*

## Disclaimer

This software is for research and educational purposes only. Do not use this system to trade real capital without extensive forward-testing and risk management protocols. Algorithmic trading carries a high level of risk.
