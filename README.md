# Symplectic Forecaster MT5 (Version 3.0)

A high-frequency algorithmic trading system designed exclusively for MetaTrader 5. Version 3.0 represents a massive architectural leap, introducing Kolmogorov-Arnold Networks (KANs), Spatio-Temporal Graph Convolutional Networks (STGCN), and a Multi-Agent Reinforcement Learning (MARL) framework based on Soft Actor-Critic (SAC).

---

## 1. What is this?

The Symplectic Forecaster is an autonomous algorithmic trading bot. It treats financial markets as complex, dynamic physical systems. By tracking the "phase space" (momentum and position) of price movements, it forecasts probabilistic trajectories. Version 3.0 upgrades the core prediction engine from standard LSTMs to a Graph-based neural network that dynamically routes capital using multiple competing AI agents.

---

## 2. Theoretical Foundations

*   **Hamiltonian Mechanics & Symplectic Geometry (Mishra, 2026):** Maps price action into a 2D phase-space matrix (Position vs. Velocity).
*   **Topological Data Analysis (TDA) (Shultz, 2023):** Computes Betti numbers and Vietoris-Rips complexes to measure market fragmentation.
*   **Kolmogorov-Arnold Representation Theorem (2024):** Proves that any multivariate continuous function can be represented as a superposition of continuous functions of one variable. This theory is the basis of our custom KAN implementation.

---

## 3. Version 3.0 Upgrades (Concepts & Tools)

*   **Kolmogorov-Arnold Networks (KANs):** 
    We replaced standard dense Neural Network layers with custom PyTorch KANs. Instead of static weights on nodes, KANs place learnable Fourier basis functions on the edges. This allows the network to discover complex symplectic invariants much faster and with higher extrapolation accuracy than traditional MLPs.
*   **Spatio-Temporal Graph Convolutional Networks (STGCN):** 
    The AI Engine now views data as a mathematical Graph (Nodes = Assets, Edges = Correlations). Even when running on a single asset, it processes the data through a spatial convolution layer, future-proofing the bot for massive cross-asset portfolio optimization.
*   **Soft Actor-Critic (SAC):** 
    Replaced PPO with Soft Actor-Critic. SAC is a maximum-entropy off-policy algorithm, which prevents the bot from collapsing into a single rigid trading style and encourages discovering highly unconventional but profitable strategies.
*   **Multi-Agent Reinforcement Learning (MARL):** 
    The system spawns two independent SAC agents: one optimized for trending markets and one for mean-reversion. A PyTorch Meta-Gating network acts as the Portfolio Manager, dynamically outputting an alpha weight ($\alpha$) to blend the two agents' actions in real-time based on the current market phase space.

---

## 4. Core Features

*   **Asynchronous Microservice Architecture:** Execution, AI, and Causal Optimization run on entirely separate CPU cores.
*   **True Multi-Horizon Loss:** The STGCN+LSTM combo optimizes across three forward trajectories simultaneously.
*   **Counterfactual Offline RL:** When a trade fails, it routes an Inverse Propensity Score back into the MARL agents to correct their policy offline.
*   **Invariant Causal Feature Pruning:** Uses DoWhy to sever non-causal features during structural regime changes.

---

## 5. Options Available

### Execution & Trading Modes
*   `--symbol` : The asset ticker to trade (e.g., `EURUSD`).
*   `--timeframe` : The chart timeframe to operate on (e.g., `M15`, `H1`).
*   `--auto-trade` : Enables live autonomous order execution.
*   `--allow-live` : Mandatory safety override required to execute trades on a Live account.

### Risk Management Parameters
*   `--risk-pct` : Percentage of total account equity to risk per trade (Default: `1.0`).
*   `--reward-risk` : Target Reward-to-Risk ratio (Default: `1.5`).
*   `--max-daily-loss` : Hard cap on daily portfolio drawdown percentage (Default: `3.0`).
*   `--atr-sl` : Multiplier for the initial Stop Loss (Default: `2.0`).

---

## 6. How to Install It

1.  **Clone the Repository (Version 3.0 Branch):**
    ```bash
    git clone -b version-3.0 https://github.com/himalay00125-boop/symplectic-forecaster-mt5.git
    cd symplectic-forecaster-mt5
    ```
2.  **Install Dependencies:**
    ```bash
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
This software integrates highly experimental machine learning and causal inference techniques (KANs, STGCN, MARL). Strictly for educational and research purposes. Algorithmic high-frequency trading carries substantial financial risk.
