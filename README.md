# Symplectic ML Forecaster — MetaTrader 5

<div align="center">

**A self-learning price forecaster with continuous trade feedback, built on symplectic phase-space geometry and topological data analysis, connected directly to MetaTrader 5.**

[![Python 3.11](https://img.shields.io/badge/Python-3.8--3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader_5-Live_Data-blue?logo=metatrader5)](https://www.metatrader5.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This project implements an **online machine learning forecaster** that ingests live market data from **MetaTrader 5** and produces **BUY / SELL / HOLD** trading signals. Unlike conventional technical-analysis tools, it models price action as a trajectory through a **symplectic phase space**, extracts topological features via **persistent homology**, and learns incrementally — bar by bar — so it adapts to regime changes in real time.

The model also **learns from every trade it takes**: when a position closes, the realized profit/loss is fed back into the ML model to reinforce winning patterns and penalize losing ones, creating a continuous improvement loop.

> **Signal-only by default** — add `--auto-trade` to enable automatic order execution with risk management. Live accounts require `--allow-live`.

---

## Key Features

| Feature | Description |
|---|---|
| **Continuous Trade Learning** | Model learns from every closed trade — feeds realized P&L back to update weights in real-time. |
| **Overfitting Protection** | L2 regularization + learning rate decay prevent the model from overreacting to individual trades. |
| **Portfolio Dashboard** | Professional portfolio analysis dashboard with equity curve, win rate, profit factor, drawdown, streaks, and full trade log. |
| **Symplectic Phase Space** | Maps price and volume to canonical coordinates `(q, p)` on a symplectic manifold. Convex hull area = first ECH capacity `C(t)`. |
| **Topological Data Analysis** | Persistent homology (H₀, H₁) on the rolling phase-space point cloud detects market cycles and structural breaks. |
| **Self-Learning Model** | Online ensemble (Passive-Aggressive Regressor + Hoeffding Adaptive Tree) updates after every bar — no batch retraining needed. |
| **Regime Detection** | Capacity spikes above the 95th percentile trigger an ALERT regime — the model avoids trading during phase-space bifurcations. |
| **Multi-Step Scenarios** | Generates bull / base / bear price paths with symplectic stability bounds (Lipschitz uncertainty bands). |
| **Any Symbol, Any Timeframe** | Works with any MT5 instrument — forex, indices, commodities, crypto — on any timeframe from M1 to MN1. |
| **Zero External Data Files** | All data comes directly from your MT5 broker. No CSVs, no API keys, no Yahoo Finance. |
| **Auto-Trade Mode** | Optional execution with % risk sizing, symplectic/ATR stop-loss, R:R take-profit, trailing stops, and daily loss limits. |

---

## How Continuous Learning Works

The model improves itself through two feedback loops:

### 1. Bar-by-Bar Learning (Market Structure)
After every new bar, the model compares its previous prediction against the actual price movement and updates its weights. This teaches it the general structure of the market.

### 2. Trade Outcome Learning (Direct Feedback)
When a trade closes (hit TP or SL), the model:
1. Retrieves the exact feature snapshot it used when opening the trade
2. Calculates the realized percentage return
3. Feeds this back into the model via `learn_one(features, realized_return)`
4. The model adjusts to encourage patterns that led to wins and avoid patterns that led to losses

### Overfitting Protection
To prevent the model from overreacting to a single bad trade:
- **L2 Regularization** — constrains weight magnitudes, ensuring smooth generalization
- **Learning Rate Decay** — the model adapts less aggressively over time (`invscaling` schedule)
- **Low C Parameter** — the Passive-Aggressive regressor uses `C=0.005` to limit per-sample influence

> **Memory requirements:** The continuous learning system uses only a few MB of RAM. Even 100,000+ trades would consume less than 500 MB of disk storage.

---

## Live Portfolio Dashboard

Every time the forecaster is running, a local web server starts up automatically in the background.

- **URL**: [http://localhost:8080](http://localhost:8080)
- **Features**:
  - **KPI Strip**: Balance, Total Return, Win Rate, Profit Factor, Max Drawdown, Total Trades
  - **Performance Stats**: Avg Win/Loss, Best/Worst Trade, Win/Loss Streaks, Expectancy
  - **Win/Loss Donut Chart**: Visual win rate breakdown
  - **Equity Curve**: Cumulative % return across all closed trades
  - **Price Action Chart**: Live price with real-time updates
  - **Direction Accuracy**: Rolling 30-bar forecast accuracy
  - **Trade Log Table**: Full history with Ticket, Action, Volume, Entry, SL, TP, and Return %
  - **Auto-refresh**: Updates every 2 seconds without page reload

---

## Mathematical Foundations

The forecaster rests on four pillars:

### 1. Financial Phase Space

Based on [Mishra (2026)](https://doi.org/10.xxxx) — models price dynamics as a Hamiltonian system:

```
Position:  q(t) = ln P(t)                   [log-price]
Momentum:  p(t) = V(t) · sign(ΔP(t))        [signed order-flow]
Capacity:  C(t) = Area(ConvexHull(Dₜ))       [symplectic area]
```

**Stability guarantee** (Lemma 3.1): `|C(t) − C(t′)| ≤ L·δ + π·δ²` bounds capacity variation under Hausdorff perturbation `δ`.

### 2. Persistent Homology (TDA)

Following [Shultz (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4378151):

- **H₀ features** — connected-component lifetimes (market fragmentation)
- **H₁ features** — loop persistence (cyclicity detection)

### 3. Hierarchical Market Structure

Per [Mantegna (1999)](https://link.springer.com/article/10.1007/s100510050929):

- Cross-asset correlation distance `d(i,j) = √(2(1 − ρᵢⱼ))`

### 4. Symplectic Capacities

Following [Cieliebak et al. (2005)](https://arxiv.org/abs/math/0506191):

- Gromov width `c₁(X_Ω) = Area(Ω)` for convex toric domains
- Capacity-preserving structure as a conservation law

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MetaTrader 5 Terminal                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ copy_rates / symbol_info_tick
                           ▼
              ┌─────────────────────────┐
              │     MT5Connection       │
              │  connect · ensure_symbol│
              └────────────┬────────────┘
                           │ Bar(OHLCV)
                           ▼
              ┌─────────────────────────┐
              │  SymplecticForecaster   │
              │                         │
              │  ┌───────────────────┐  │
              │  │ Phase Coordinates │  │  q = ln P,  p = V·sign(ΔP)
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │ Convex Hull       │  │  C(t) = Area,  L(t) = Perimeter
              │  │ + TDA (Ripser)    │  │  Betti-0, Betti-1, persistence
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │ Online ML Model   │  │  PA Regressor + Hoeffding Tree
              │  │ (self-learning)   │  │  Updates every bar + trade close
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │ Scenario Generator│  │  Bull / Base / Bear paths
              │  │ + Stability Bands │  │  Lipschitz uncertainty envelope
              │  └────────┬──────────┘  │
              └───────────┼─────────────┘
                          │ forecast dict
                          ▼
              ┌─────────────────────────┐
              │ AutoTradingEngine       │
              │  BUY  ▲  confidence > θ │──┐
              │  SELL ▼  confidence > θ │  │ on close: learn_one(features, pnl)
              │  HOLD ━  low conf/ALERT │──┘
              └─────────────────────────┘
```

---

## Installation

### Prerequisites

- **Windows** (MetaTrader 5 is Windows-only)
- **Python 3.8 – 3.13** (MT5 package does **not** support Python 3.14+)
- **MetaTrader 5 terminal** installed and logged in (demo or live account)

### Setup

```bash
# Clone the repository
git clone https://github.com/himalay00125-boop/symplectic-forecaster-mt5.git
cd symplectic-forecaster-mt5

# Install dependencies
pip install -r requirements.txt
```

### Optional (enhanced features)

```bash
# Ripser for exact TDA (persistent homology)
pip install ripser

# River for online ML ensemble (PA Regressor + Hoeffding Tree)
pip install river
```

> Without `ripser`, TDA features are approximated. Without `river`, the model falls back to sklearn's SGDRegressor with L2 regularization.

---

## Usage

### Quick Start

Make sure your **MetaTrader 5 terminal is running** and logged in, then:

```bash
# Interactive mode — prompts for symbol and timeframe
python symplectic_forecaster.py

# Or use the launcher (auto-selects correct Python version)
run.bat
```

### Command-Line Arguments

```bash
# Specify symbol and timeframe directly
python symplectic_forecaster.py --symbol EURUSD --timeframe H1

# Gold on daily chart with 2000 bars of history
python symplectic_forecaster.py --symbol XAUUSD --timeframe D1 --bars 2000

# Auto-trade on demo account (1% risk/trade, 3% max daily loss)
python symplectic_forecaster.py --symbol EURUSD --timeframe H1 --auto-trade

# Auto-trade with custom risk settings
python symplectic_forecaster.py --symbol XAUUSD --timeframe H4 --auto-trade \
    --risk-pct 0.5 --max-daily-loss 2.0 --reward-risk 2.5 --trailing-atr 1.2

# With explicit MT5 login
python symplectic_forecaster.py --symbol US500 --timeframe H4 \
    --account 12345678 --password mypass --server "BrokerDemo-Server"
```

### All Options

| Argument | Default | Description |
|---|---|---|
| `--symbol` | *(interactive)* | MT5 symbol (e.g., `EURUSD`, `XAUUSD`, `US500`, `BTCUSD`) |
| `--timeframe` | *(interactive)* | Timeframe: `M1` `M5` `M15` `M30` `H1` `H4` `H12` `D1` `W1` `MN1` |
| `--bars` | `1000` | Number of historical bars for initial training |
| `--confidence` | `0.4` | Minimum confidence for BUY/SELL signals (0.0 – 1.0) |
| `--optimize` | `false` | Grid-search confidence / risk / R:R on backtest data |
| `--opt-confidence` | `0.2,0.3,0.4,0.5,0.6` | Optimizer confidence grid |
| `--opt-risk` | `0.5,1.0,1.5` | Optimizer risk % grid |
| `--opt-reward-risk` | `1.5,2.0,2.5` | Optimizer R:R grid |
| `--window` | `60` | Rolling window size for symplectic phase space |
| `--poll` | *(auto)* | Poll interval in seconds (0 = auto-detect from timeframe) |
| `--auto-trade` | `false` | Enable automatic order execution |
| `--allow-live` | `false` | Allow trading on live (non-demo) accounts |
| `--risk-pct` | `1.0` | Risk per trade as % of equity |
| `--max-daily-loss` | `3.0` | Halt trading if daily loss exceeds this % |
| `--reward-risk` | `2.0` | Take-profit / stop-loss ratio |
| `--atr-sl` | `1.5` | ATR multiplier for stop-loss distance |
| `--trailing-atr` | `1.0` | ATR multiplier for trailing stop distance |
| `--max-positions` | `1` | Max concurrent positions per symbol |
| `--backtest` | `false` | Run out-of-sample backtest instead of live |
| `--train-bars` | `1000` | Training bars for backtest |
| `--test-bars` | `500` | Out-of-sample bars for backtest |
| `--initial-balance` | `10000` | Starting balance for backtest |
| `--spread-pips` | `1.0` | Simulated spread in pips |
| `--freeze-model` | `false` | Don't update model during backtest test period |
| `--state-dir` | `states` | Directory for model state files |
| `--load-state` | *(none)* | Load model state from pickle file |
| `--no-save-state` | `false` | Disable auto-save on exit |
| `--account` | *(none)* | MT5 account number (optional if terminal is logged in) |
| `--password` | *(none)* | MT5 password (optional) |
| `--server` | *(none)* | MT5 server name (optional) |
| `--mt5-path` | *(auto)* | Path to `terminal64.exe` (optional, auto-detected) |

---

## Signal Logic

The trading engine generates signals based on three factors:

| Signal | Condition |
|---|---|
| **▲ BUY** | `direction = +1` AND `confidence > threshold` AND `regime ≠ ALERT` |
| **▼ SELL** | `direction = -1` AND `confidence > threshold` AND `regime ≠ ALERT` |
| **━ HOLD** | `confidence < threshold` OR `regime = ALERT` |

---

## Auto-Trade Mode

Enable with `--auto-trade`. The bot executes signals through MT5 with built-in risk controls:

| Control | Behavior |
|---|---|
| **Position sizing** | Lot size computed from `% equity at risk` and stop distance |
| **Stop-loss** | Symplectic stability bands when available; falls back to ATR |
| **Take-profit** | Set at `reward_risk` × stop distance (default 1:2 R:R) |
| **Trailing stop** | ATR-based trail updated every poll cycle on open positions |
| **Daily loss limit** | Trading halts if equity drops by `max_daily_loss` % in a session |
| **Trade learning** | On every close, realized P&L is fed back into the model |
| **Live safety** | Demo accounts trade freely; live accounts require `--allow-live` |

### Recommended Demo Workflow

1. Open MT5 and log into a **demo** account
2. Run: `python symplectic_forecaster.py --symbol EURUSD --timeframe H1 --auto-trade`
3. Watch the console for `[TRADE]` messages and the dashboard at `http://localhost:8080`
4. Look for `[Learned Feedback]` in the console — confirms the model learned from a closed trade
5. Only add `--allow-live` after extensive demo testing

---

## Model State Persistence

The forecaster saves its learned weights and rolling buffers to disk so restarts don't wipe progress.

| Behavior | Description |
|---|---|
| **Auto-save** | On exit, state is saved to `states/SYMBOL_TIMEFRAME.pkl` |
| **Auto-load** | On startup, if a matching state file exists, training is skipped and the model resumes |
| **Manual load** | `--load-state states/EURUSD_H1.pkl` forces loading a specific file |
| **Disable** | `--no-save-state` prevents saving on exit |

---

## Backtest Mode

Run a walk-forward simulation on historical MT5 data without placing real orders.

```bash
# Train on 1000 bars, backtest on next 500 (default split)
python symplectic_forecaster.py --symbol EURUSD --timeframe H1 --backtest

# Custom train/test split with spread simulation
python symplectic_forecaster.py --symbol XAUUSD --timeframe H4 --backtest \
    --train-bars 2000 --test-bars 1000 --spread-pips 2.0 --initial-balance 50000

# Pure out-of-sample: freeze model weights during test period
python symplectic_forecaster.py --symbol EURUSD --timeframe H1 --backtest \
    --train-bars 1500 --test-bars 500 --freeze-model
```

### Walk-Forward Optimizer

Grid-search confidence, risk %, and R:R on the same train/test split:

```bash
python symplectic_forecaster.py --symbol EURUSD --timeframe H1 --optimize

# Custom search grid
python symplectic_forecaster.py --symbol EURUSD --timeframe H1 --optimize \
    --opt-confidence "0.25,0.35,0.45" --opt-risk "0.5,1.0" \
    --opt-reward-risk "1.5,2.0,3.0"
```

---

## Project Structure

```
symplectic-forecaster-mt5/
├── symplectic_forecaster.py   # Main engine + continuous learning + background server
├── dashboard.html             # Portfolio analysis dashboard (equity, trades, KPIs)
├── run.bat                    # Windows launcher (auto-selects Python 3.11)
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `MetaTrader5` | Yes | Live market data from MT5 terminal |
| `numpy` | Yes | Numerical computation |
| `pandas` | Yes | Data manipulation |
| `scipy` | Yes | Convex hull computation |
| `scikit-learn` | Yes | ML model (SGDRegressor with L2 regularization) |
| `matplotlib` | Yes | Backtest equity charts |
| `ripser` | Optional | Exact persistent homology (TDA) |
| `river` | Optional | Online ML ensemble (PA + Hoeffding Tree) |

---

## Disclaimer

> [!WARNING]
> **This software is for educational and research purposes only.**
>
> - This is **not** financial advice. Trading involves substantial risk of loss.
> - Past performance does not guarantee future results.
> - The mathematical models provide *structured analysis*, not certainty.
> - Always test on a **demo account** before considering real capital.
> - The authors assume no liability for financial losses.

---

## References

1. Mishra, H. (2026). *Symplectic Phase-Space Geometry for Financial Time Series*. Stability Lemma 1.
2. Shultz, G. (2023). *Topological Data Analysis for Financial Time Series*. SSRN 4378151.
3. Mantegna, R. N. (1999). *Hierarchical Structure in Financial Markets*. Eur. Phys. J. B, 11, 193–197.
4. Cieliebak, K. et al. (2005). *Symplectic Homology and the Eilenberg-Steenrod Axioms*. arXiv:math/0506191.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
