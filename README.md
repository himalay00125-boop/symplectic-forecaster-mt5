# Symplectic Forecaster MT5 (Version 5.0)

A high-frequency algorithmic trading system designed exclusively for MetaTrader 5. Version 5.0 represents the cutting edge of quantitative finance, introducing Mamba State Space Models (SSM), newsfilter.io Live NLP Sentiment Analysis, NOTEARS Causal Graph Discovery, and a distributed Ray RLlib architecture.

---

## 1. What is this?

The Symplectic Forecaster is an autonomous algorithmic trading bot. It treats financial markets as complex, dynamic physical systems. By tracking the "phase space" (momentum and position) of price movements, it forecasts probabilistic trajectories. Version 5.0 upgrades the core engine to a Distributed Ray Framework, utilizing Mamba SSMs and real-time newsfilter.io NLP sentiment to optimize trading across clusters of machines.

---

## 2. Theoretical Foundations

*   **Hamiltonian Mechanics & Symplectic Geometry (Mishra, 2026):** Maps price action into a 2D phase-space matrix (Position vs. Velocity).
*   **Topological Data Analysis (TDA) (Shultz, 2023):** Computes Betti numbers and Vietoris-Rips complexes to measure market fragmentation.
*   **Kolmogorov-Arnold Representation Theorem (2024):** Proves that any multivariate continuous function can be represented as a superposition of continuous functions of one variable.
*   **DAGs with NO TEARS (Zheng et al., 2018):** Continuous optimization for structure learning to dynamically model causal relationships between assets.

---

## 3. Version 5.0 Upgrades (Concepts & Tools)

*   **Mamba State Space Models (SSM):** 
    We replaced standard PyTorch Transformers with a pure PyTorch MambaBlock. Mamba allows for linear-time sequence modeling via selective state spaces, outperforming Transformers on extremely long contexts and avoiding the quadratic memory bottleneck.
*   **Realtime NLP Sentiment Agent (newsfilter.io):** 
    Integrated a background daemon (`nlp_agent.py`) that utilizes `newsfilter.io` to fetch live market headlines via WebSockets/REST. It scores market sentiment using ProsusAI's FinBERT, feeding a continuous [-1.0, 1.0] alpha vector directly into the MARL Meta-Gate.
*   **NOTEARS Causal Graph Discovery:** 
    Instead of assuming static correlations, the system uses continuous Lagrangian optimization (`causal_discovery.py`) to build an adjacency matrix of the True Causal Graph between structural features, allowing the Spatio-Temporal Graph Convolutional Network (STGCN) to route information effectively.
*   **Distributed Architecture (Ray & RLlib):** 
    The raw `multiprocessing` PyTorch queues have been replaced with `ray`. The heavy neural network (STGCN + Transformer + KAN + SAC) now lives in an asynchronous Ray Actor (`AIEngineActor`), allowing the engine to scale infinitely across cloud clusters.

---

## 4. Core Features

*   **Asynchronous Microservice Architecture:** Execution, Ray AI Actors, NLP Sentiment, and Causal Optimization run completely decoupled.
*   **Multi-Agent Reinforcement Learning (MARL):** Two independent Soft Actor-Critic (SAC) agents optimized for trending and mean-reversion, gated by a PyTorch Meta-Network.
*   **Invariant Causal Feature Pruning:** Uses DoWhy and NOTEARS to sever non-causal features during structural regime changes.

---

## 5. Options Available

### Execution & Trading Modes
*   `--symbol` : The asset ticker to trade (e.g., `EURUSD`).
*   `--timeframe` : The chart timeframe to operate on (e.g., `M15`, `H1`).
*   `--auto-trade` : Enables live autonomous order execution.
*   `--allow-live` : Mandatory safety override required to execute trades on a Live account.
*   `--backtest` : Runs a fast offline simulation for X bars.
*   `--bars` : Number of historical bars to load (Default: `1000`).

### Risk Management Parameters
*   `--risk-pct` : Percentage of total account equity to risk per trade (Default: `1.0`).
*   `--reward-risk` : Target Reward-to-Risk ratio (Default: `1.5`).
*   `--max-daily-loss` : Hard cap on daily portfolio drawdown percentage (Default: `3.0`).
*   `--atr-sl` : Multiplier for the initial Stop Loss (Default: `2.0`).

---

## 6. How to Install It

1.  **Clone the Repository (Version 5.0 Branch):**
    ```bash
    git clone -b version-5.0 https://github.com/himalay00125-boop/symplectic-forecaster-mt5.git
    cd symplectic-forecaster-mt5
    ```
2.  **Install Dependencies:**
    ```bash
    pip install transformers yfinance ray[rllib] scipy
    pip install -r requirements.txt
    ```

---

## 7. How to Run It

### Signal-Only Mode (Safest for Testing)
```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe M15
```

### Live Auto-Trading (Demo Account)
```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe M15 --auto-trade --risk-pct 1.0 --reward-risk 2.0
```

## Disclaimer
This software integrates highly experimental machine learning and causal inference techniques (Transformers, Ray, KANs, STGCN, MARL). Strictly for educational and research purposes. Algorithmic high-frequency trading carries substantial financial risk.
