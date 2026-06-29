# Symplectic Forecaster MT5 (Version 2.0)

A high-frequency algorithmic trading system designed exclusively for MetaTrader 5. Version 2.0 introduces a state-of-the-art Deep Learning and Reinforcement Learning microservice architecture, replacing traditional heuristics with dynamic, self-optimizing mathematical models.

---

## 1. What is this?

The Symplectic Forecaster is an autonomous algorithmic trading bot. It bridges the gap between high-frequency quantitative finance and cutting-edge machine learning. Unlike traditional technical indicator-based bots (which rely on lagging moving averages or RSI), this system views the financial markets as a complex, dynamic physical system. It tracks the "phase space" (momentum and position) of price movements and uses deep neural networks to forecast the probabilistic trajectory of future prices. 

Version 2.0 represents a complete architectural overhaul. It decouples trade execution from heavy artificial intelligence computation by utilizing a Python multiprocessing architecture. This ensures that the MetaTrader 5 execution thread experiences zero latency during live market operations, while PyTorch and Causal Inference engines crunch data in the background.

---

## 2. What is it based on?

This system synthesizes peer-reviewed research from several domains of quantitative finance, topology, and physics:

*   **Hamiltonian Mechanics & Symplectic Geometry (Mishra, 2026):** 
    Rather than looking at a standard time-series chart, the system models financial markets as physical systems conserving momentum. It maps price action into a 2D phase-space matrix (Position vs. Velocity), allowing the algorithm to detect when a market trend has exhausted its kinetic energy and is mathematically bound to mean-revert.
*   **Topological Data Analysis (TDA) (Shultz, 2023):** 
    The system tracks the topological shape and persistence of price clusters. By computing Betti numbers and utilizing Vietoris-Rips complexes, it measures market fragmentation and consolidation, mathematically identifying whether the current price action is noisy or structurally sound.
*   **Econophysics (Mantegna, 1999):** 
    Views market microstructure as a stochastic process. The forecaster captures complex, non-linear market dynamics using convex hulls and structural regime alerts to identify phase transitions (e.g., shifting from low-volatility accumulation to high-volatility distribution).

---

## 3. Concepts and Tools Used

Version 2.0 utilizes the most robust frameworks available in modern machine learning:

*   **PyTorch (Deep Sequence Modeling):** 
    Employs a custom Long Short-Term Memory (LSTM) recurrent neural network. Instead of predicting a single future data point, the LSTM models multi-horizon temporal dependencies, generating robust forecasts across multiple future bars simultaneously using Backpropagation Through Time (BPTT).
*   **StableBaselines3 & Gymnasium (Reinforcement Learning):** 
    Utilizes the Proximal Policy Optimization (PPO) algorithm operating within a custom `TradingEnv` Gymnasium environment. It learns optimal trading policies and uses Inverse Propensity Scoring to learn from counterfactual scenarios.
*   **DoWhy (Causal Inference):** 
    Integrates Microsoft's DoWhy framework for Invariant Causal Prediction. It utilizes Directed Acyclic Graphs (DAGs) and do-calculus to dynamically prune non-causal features that flip correlations during structural regime changes.
*   **Optuna (Bayesian Optimization):** 
    Runs continuous, non-blocking background hyperparameter studies using Tree-structured Parzen Estimators (TPE) to mathematically dial in the optimal neural network regularization rates and grace periods.
*   **MetaTrader 5 (MT5) Integration:** 
    Directly interfaces with the MT5 terminal via the official `MetaTrader5` C++ Python bridge for sub-millisecond data ingestion and order execution.

---

## 4. Core Features

*   **Asynchronous Microservice Architecture:** Utilizes native Python multiprocessing to run the Execution Engine, AI Engine, and Causal Optimizer on entirely separate CPU cores. The MT5 execution thread never drops a tick.
*   **True Multi-Horizon Loss:** The PyTorch LSTM optimizes across three forward trajectories simultaneously ($0.5 R_1 + 0.3 R_2 + 0.2 R_3$). This forces the model to maximize long-term holding stability rather than chasing immediate, noisy ticks.
*   **Counterfactual Offline RL:** When a trade fails (i.e., hits a Stop Loss), the system does not simply record a loss. It synthesizes the true adverse return, mathematically penalizes its own blindspots, and routes an Inverse Propensity Score back into the neural network to permanently correct the policy offline.
*   **Invariant Causal Feature Pruning:** The causal optimizer tracks how features correlate with price across different market regimes (NORMAL vs ALERT). It mathematically severs the weights of features that exhibit high cross-regime variance, leaving only the true causal drivers of price.
*   **Live Portfolio Dashboard:** Features an embedded asynchronous web dashboard for monitoring live PnL, feature attribution, and predicted regime transition probabilities.
*   **Adaptive Risk Management:** Automatically trails Stop Losses and computes Take Profits based on real-time Average True Range (ATR) volatility scaling.

---

## 5. Options Available

The system supports a highly granular array of command-line configurations to tailor the bot to your specific risk appetite.

### Execution & Trading Modes
*   `--symbol` : The asset ticker to trade (e.g., `EURUSD`, `XAUUSD`, `BTCUSD`).
*   `--timeframe` : The chart timeframe to operate on (e.g., `M1`, `M5`, `M15`, `H1`, `D1`).
*   `--auto-trade` : Enables live autonomous order execution. If omitted, the bot runs safely in **Signal-Only Mode** (printing predictions without placing trades).
*   `--allow-live` : A mandatory safety override required to execute trades on a Live (Real money) account. Without this flag, auto-trading is restricted to Demo accounts.

### Risk Management Parameters
*   `--risk-pct` : Percentage of total account equity to risk per trade (Default: `1.0`).
*   `--reward-risk` : Target Reward-to-Risk ratio (Default: `1.5`).
*   `--max-daily-loss` : Hard cap on daily portfolio drawdown percentage. If hit, the bot ceases trading for the day (Default: `3.0`).
*   `--atr-sl` : Multiplier for the initial Stop Loss, calculated based on the Average True Range (Default: `2.0`).
*   `--trailing-atr` : Multiplier used to trail the Stop Loss once a position is in profit (Default: `1.5`).
*   `--max-positions` : Maximum number of concurrent open positions (Default: `3`).

### Filters & Overrides
*   `--news-filter` : Suspends trading during high-impact macroeconomic news releases to protect against slippage.
*   `--time-filter` : Restricts trading to high-liquidity market hours (Avoids the 00:00-06:00 UTC low-liquidity rollover period).
*   `--confidence` : Minimum probabilistic confidence required from the AI engine before a trade is authorized (Default: `0.6`).

---

## 6. How to Install It

### Prerequisites
1.  **Operating System:** Windows 10 or 11 (Required by the MetaTrader 5 terminal).
2.  **Python:** Python 3.10 or 3.11. *(Note: Python 3.12+ may not be fully supported by the MT5 C++ bridge).*
3.  **MetaTrader 5:** The MT5 Terminal must be installed and logged into your broker.
4.  **Hardware (Optional):** A CUDA-enabled NVIDIA GPU is highly recommended to accelerate PyTorch training, though the system will fallback to CPU if necessary.

### Step-by-Step Installation
1.  **Clone the Repository (Version 2.0 Branch):**
    Open your command prompt or PowerShell and run:
    ```bash
    git clone -b version-2.0 https://github.com/himalay00125-boop/symplectic-forecaster-mt5.git
    cd symplectic-forecaster-mt5
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Install the heavyweight deep learning and causal inference libraries:
    ```bash
    pip install -r requirements.txt
    ```

---

## 7. How to Run It

### Preparing MetaTrader 5
1.  Launch your MetaTrader 5 terminal.
2.  Navigate to **Tools -> Options -> Expert Advisors**.
3.  Check the box for **"Allow algorithmic trading"** and click OK.

### Running the System
Open your command prompt, ensure your virtual environment is activated, and execute the python script with your desired arguments.

**Example 1: Signal-Only Mode (Safest for Testing)**
This will run the AI engines and print predictions to the console, but will *not* execute trades in MT5.
```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe M15
```

**Example 2: Live Auto-Trading (Demo Account)**
This will autonomously execute trades, risking 1% of the account per trade, with a 2.0 Reward-to-Risk ratio.
```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe M15 --auto-trade --risk-pct 1.0 --reward-risk 2.0
```

**Example 3: Multi-Symbol Execution**
Run the engine across a basket of assets simultaneously on the 1-Hour chart.
```bash
python symplectic_forecaster.py --symbols "EURUSD,GBPUSD,XAUUSD" --timeframe H1 --auto-trade
```

### What to Expect Upon Launch
1.  **Initialization:** The terminal will confirm connection to MT5.
2.  **Process Spawning:** You will see logs indicating that `[AI ENGINE]` and `[CAUSAL ENGINE]` have successfully booted in background processes.
3.  **Warm-up Period:** The bot requires historical data to build the initial phase-space matrix and causal graphs. It will process the last 80 bars silently.
4.  **Live Execution:** Once warmed up, the bot will begin printing live probabilistic forecasts, regime classifications, and executing trades when the confidence threshold is breached.

---

## Disclaimer

This software integrates experimental machine learning and causal inference techniques. It is strictly for educational and research purposes. Do not deploy this system on live capital without extensive forward-testing on a Demo account. Algorithmic high-frequency trading carries substantial financial risk, and past performance is not indicative of future results.
