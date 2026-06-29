# Symplectic Forecaster MT5 (Version 2.0)

A high-frequency algorithmic trading system designed exclusively for MetaTrader 5, utilizing a state-of-the-art Deep Learning and Reinforcement Learning microservice architecture.

## What is this?

Symplectic Forecaster is an advanced autonomous algorithmic trading bot. It bridges the gap between high-frequency quantitative finance and cutting-edge machine learning. Unlike traditional technical indicator-based bots, it views the financial markets as a complex dynamical system. It tracks the "phase space" (momentum and position) of price movements and uses deep neural networks to forecast the probabilistic trajectory of future prices. Version 2.0 represents a complete architectural overhaul, decoupling trade execution from heavy artificial intelligence computation to ensure zero-latency order routing.

## What is it based on?

This system synthesizes research from several domains of quantitative finance and topology:
* **Hamiltonian Mechanics & Symplectic Geometry (Mishra, 2026):** Models financial markets as physical systems conserving momentum, mapping price action into a 2D phase-space matrix.
* **Topological Data Analysis (TDA) (Shultz, 2023):** Tracks the shape and persistence of price clusters, computing Betti numbers to measure market fragmentation and consolidation.
* **Econophysics (Mantegna, 1999):** Views market microstructure as a stochastic process, capturing complex dynamics using convex hulls and structural regime alerts.

## Concepts and Tools Used

* **PyTorch (Deep Sequence Modeling):** Employs a custom Long Short-Term Memory (LSTM) network to model multi-horizon temporal dependencies, generating robust forecasts across multiple future bars.
* **StableBaselines3 (Reinforcement Learning):** Utilizes Proximal Policy Optimization (PPO) in a custom Gymnasium environment. It learns optimal trading policies and uses Inverse Propensity Scoring to learn from counterfactual scenarios (e.g., automatically adjusting weights when a Stop Loss is hit).
* **DoWhy (Causal Inference):** Integrates Microsoft's DoWhy framework for Invariant Causal Prediction. It dynamically prunes non-causal features that flip correlations during structural regime changes.
* **Optuna (Bayesian Optimization):** Runs continuous, non-blocking background hyperparameter studies to mathematically dial in the optimal Regularization (C) and Grace Periods.
* **MetaTrader 5 (MT5) Integration:** Directly interfaces with the MT5 terminal via the official `MetaTrader5` Python library for sub-millisecond execution.

## Features

* **Microservice Architecture:** Utilizes native Python multiprocessing to run the Execution Engine, AI Engine, and Causal Optimizer on separate threads. The MT5 execution thread never blocks.
* **True Multi-Horizon Loss:** The PyTorch model optimizes across three forward trajectories simultaneously to maximize trade holding stability.
* **Counterfactual Offline RL:** When a trade fails (hits a Stop Loss), the system mathematically penalizes its own blindspots and routes an Inverse Propensity Score back into the neural network to permanently correct the policy.
* **Live Portfolio Dashboard:** Features an embedded asynchronous web dashboard for monitoring live PnL, feature attribution, and predicted regime transition probabilities.
* **Adaptive Risk Management:** Automatically trails Stop Losses based on real-time Average True Range (ATR) volatility scaling.

## Options Available

The system supports a wide array of command-line configurations:

### Trading Modes
* `--auto-trade`: Enables live autonomous order execution. If omitted, the bot runs in signal-only mode.
* `--allow-live`: Safety switch required to execute trades on a Live (Real money) account rather than a Demo account.

### Risk Parameters
* `--risk-pct`: Percentage of account balance to risk per trade (Default: 1.0).
* `--reward-risk`: Target Reward-to-Risk ratio (Default: 1.5).
* `--max-daily-loss`: Hard cap on daily portfolio drawdown percentage before the bot shuts down (Default: 3.0).
* `--atr-sl`: Multiplier for initial Stop Loss based on ATR (Default: 2.0).

### Filters
* `--news-filter`: Suspends trading during high-impact macroeconomic news releases.
* `--time-filter`: Restricts trading to high-liquidity market hours (Avoids 00:00-06:00 UTC).

## Installation

### Prerequisites
* Windows OS (Required by MetaTrader 5).
* Python 3.10 or 3.11 (Python 3.12+ may not support the MT5 library).
* MetaTrader 5 Terminal installed and running.
* (Optional but Recommended) A CUDA-enabled NVIDIA GPU for PyTorch acceleration.

### Setup Steps
1. Clone the repository and checkout Version 2.0:
   ```bash
   git clone https://github.com/himalay00125-boop/symplectic-forecaster-mt5.git
   cd symplectic-forecaster-mt5
   git checkout version-2.0
   ```
2. Install the necessary heavyweight AI dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Open your MetaTrader 5 terminal and ensure you are logged into your broker.
2. In MT5, go to **Tools -> Options -> Expert Advisors** and enable "Allow algorithmic trading".
3. Open your command prompt (or PowerShell) in the project directory.

### Example Commands

**Run in Signal-Only Mode (Safest):**
```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe M15
```

**Run with Live Auto-Trading (Demo Account):**
```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe M15 --auto-trade --risk-pct 1.0 --reward-risk 2.0
```

**Run on Multiple Symbols:**
```bash
python symplectic_forecaster.py --symbols "EURUSD,GBPUSD,XAUUSD" --timeframe H1 --auto-trade
```

Once running, the system will initialize the PyTorch and Causal Optimizer background workers. It will require a "warm-up" period (by default, 80 historical bars) to build the initial phase-space matrix and causal graphs before outputting its first live trade.

## Disclaimer

This software integrates experimental machine learning techniques. It is strictly for educational and research purposes. Do not deploy this on live capital without extensive forward-testing. Algorithmic high-frequency trading carries substantial financial risk.
