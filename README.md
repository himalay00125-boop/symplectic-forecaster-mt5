# Symplectic ML Forecaster — MetaTrader 5

<div align="center">

**A self-learning price forecaster built on symplectic phase-space geometry and topological data analysis, connected directly to MetaTrader 5.**

[![Python 3.11](https://img.shields.io/badge/Python-3.8--3.13-3776AB?logo=python&logoColor=white)](https://python.org)
[![MetaTrader 5](https://img.shields.io/badge/MetaTrader_5-Live_Data-blue?logo=metatrader5)](https://www.metatrader5.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This project implements an **online machine learning forecaster** that ingests live market data from **MetaTrader 5** and produces **BUY / SELL / HOLD** trading signals. Unlike conventional technical-analysis tools, it models price action as a trajectory through a **symplectic phase space**, extracts topological features via **persistent homology**, and learns incrementally — bar by bar — so it adapts to regime changes in real time.

> **Signal-only mode** — the system generates signals but does **not** place orders automatically. You stay in full control.

---

## Key Features

| Feature | Description |
|---|---|
| **Symplectic Phase Space** | Maps price and volume to canonical coordinates `(q, p)` on a symplectic manifold. Convex hull area = first ECH capacity `C(t)`. |
| **Topological Data Analysis** | Persistent homology (H₀, H₁) on the rolling phase-space point cloud detects market cycles and structural breaks. |
| **Self-Learning Model** | Online ensemble (Passive-Aggressive Regressor + Hoeffding Adaptive Tree) updates after every bar — no batch retraining needed. |
| **Regime Detection** | Capacity spikes above the 95th percentile trigger an ALERT regime — the model avoids trading during phase-space bifurcations. |
| **Multi-Step Scenarios** | Generates bull / base / bear price paths with symplectic stability bounds (Lipschitz uncertainty bands). |
| **Any Symbol, Any Timeframe** | Works with any MT5 instrument — forex, indices, commodities, crypto — on any timeframe from M1 to MN1. |
| **Zero External Data Files** | All data comes directly from your MT5 broker. No CSVs, no API keys, no Yahoo Finance. |

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
              │  │ (self-learning)   │  │  Updates every bar
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
              │    TradingEngine        │
              │  BUY  ▲  confidence > θ │
              │  SELL ▼  confidence > θ │
              │  HOLD ━  low conf/ALERT │
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

> Without `ripser`, TDA features are approximated. Without `river`, the model falls back to sklearn's PassiveAggressiveRegressor.

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

### Live Web Dashboard

Every time the forecaster is running, a local web server starts up automatically in the background.

- **URL**: [http://localhost:8080](http://localhost:8080)
- **Features**:
  - **Dynamic Updates**: Real-time polling updates the charts in-place every 2 seconds.
  - **Four Live Charts**:
    - **Price & Capacity**: Price trend with overlaying ECH capacity $C(t)$ and alert indicators.
    - **Phase Space**: Rolling $(q,p)$ phase coordinate point cloud and enclosing green convex hull polygon.
    - **Topological Persistence**: Betti-1 ($\beta_1$) loop generator counts and total persistence scale.
    - **Directional Accuracy**: Rolling 30-bar walking accuracy hit rate.
  - **Forecast Panel**: Dynamic Bull, Bear, Base predictions and stability limits.
  - **Theme Support**: Automatically adapts to your system Light/Dark mode.

### Command-Line Arguments

```bash
# Specify symbol and timeframe directly
python symplectic_forecaster.py --symbol EURUSD --timeframe H1

# Gold on daily chart with 2000 bars of history
python symplectic_forecaster.py --symbol XAUUSD --timeframe D1 --bars 2000

# Bitcoin on 5-minute chart with higher confidence threshold
python symplectic_forecaster.py --symbol BTCUSD --timeframe M5 --confidence 0.7

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
| `--confidence` | `0.6` | Minimum confidence for BUY/SELL signals (0.0 – 1.0) |
| `--window` | `60` | Rolling window size for symplectic phase space |
| `--poll` | *(auto)* | Poll interval in seconds (0 = auto-detect from timeframe) |
| `--account` | *(none)* | MT5 account number (optional if terminal is logged in) |
| `--password` | *(none)* | MT5 password (optional) |
| `--server` | *(none)* | MT5 server name (optional) |
| `--mt5-path` | *(auto)* | Path to `terminal64.exe` (optional, auto-detected) |

### Running with Docker

Since MetaTrader 5 is a Windows application, running the forecaster inside a Docker container (which uses Linux) requires a lightweight **bridge** between the Linux container and your Windows host.

#### Step 1: Start the MT5 bridge on your Windows host
In a Command Prompt or PowerShell terminal on your Windows host, run:
```cmd
# Install RPyC bridge package
pip install mt5linux rpyc

# Start the bridge server (default port 18812)
python -m mt5linux
```
*(Make sure your MetaTrader 5 terminal remains open and logged in on your Windows desktop).*

#### Step 2: Build and run the Docker container
Open a terminal in your project directory and run:
```bash
# Build the image and start the container
docker compose up --build -d
```

#### Step 3: View the dashboard
Once the container is running, access the dashboard at:
[http://localhost:8080](http://localhost:8080)

To stop the container, run:
```bash
docker compose down
```

### Using as a Library

```python
from symplectic_forecaster import SymplecticForecaster, MT5Connection, TradingEngine

# Connect to MT5
conn = MT5Connection()
conn.connect()

# Initialize
fc = SymplecticForecaster(window=60)
engine = TradingEngine(confidence_threshold=0.6)

# Train on historical data
fc.train_on_mt5("EURUSD", "H1", n_bars=1000, connection=conn)

# Get latest forecast
result = fc.forecast(horizon=5)
print(result)

# Start live monitoring with signal callbacks
fc.run_live_mt5("EURUSD", "H1", on_signal=engine.on_signal, connection=conn)
```

---

## Signal Logic

The trading engine generates signals based on three factors:

| Signal | Condition |
|---|---|
| **▲ BUY** | `direction = +1` AND `confidence > threshold` AND `regime ≠ ALERT` |
| **▼ SELL** | `direction = -1` AND `confidence > threshold` AND `regime ≠ ALERT` |
| **━ HOLD** | `confidence < threshold` OR `regime = ALERT` |

### Why HOLD during ALERT?

When the symplectic capacity `C(t)` spikes above the 95th percentile, it indicates a **phase-space bifurcation** — the market is undergoing a structural regime change. The stability lemma guarantees that capacity variations are bounded under normal conditions, but during bifurcations these bounds are violated. The model protects capital by refusing to trade during these unstable periods.

---

## Output Example

```
══════════════════════════════════════════════════════════════════
  SYMPLECTIC ML PRICE FORECASTER — MetaTrader 5
  Based on: Mishra (2026) · Shultz (2023) · Mantegna (1999)
  Mode: Signal-Only (no auto-execution)
══════════════════════════════════════════════════════════════════

[MT5] Connected to : MetaTrader 5
[MT5] Account      : 12345678 (Demo)
[MT5] Balance      : 100000.00 USD
[MT5] Fetching 1000 bars of EURUSD (H1) ...
[MT5] Received 1000 bars.
[INFO] Pipeline complete. 920 forecasts generated.

──────────────────────────────────────────────────────────────────
  ⏱  2026-06-05 14:00:00  │  EURUSD
  ▲ BUY   │  Price: 1.08542  │  Confidence: 72.3%
  Predicted Return: +0.0012%  │  Regime: NORMAL
  Bullish signal: predicted return +0.0012%, confidence 72.3%, regime stable.

  Scenarios (5-bar ahead):
    bull: 1.08612 → 1.08682 → 1.08752 → 1.08823 → 1.08893
    base: 1.08555 → 1.08568 → 1.08581 → 1.08594 → 1.08607
    bear: 1.08498 → 1.08455 → 1.08411 → 1.08367 → 1.08324
──────────────────────────────────────────────────────────────────
```

---

## Project Structure

```
symplectic-forecaster-mt5/
├── symplectic_forecaster.py   # Main engine + background server
├── dashboard.html             # Dynamic HTML/JS dashboard page
├── run.bat                    # Windows launcher (auto-selects Python 3.11)
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `MetaTrader5` | ✅ | Live market data from MT5 terminal |
| `numpy` | ✅ | Numerical computation |
| `pandas` | ✅ | Data manipulation |
| `scipy` | ✅ | Convex hull computation |
| `scikit-learn` | ✅ | Fallback ML model (PassiveAggressiveRegressor) |
| `ripser` | ⬜ Optional | Exact persistent homology (TDA) |
| `river` | ⬜ Optional | Online ML ensemble (PA + Hoeffding Tree) |

---

## Supported Timeframes

| Category | Timeframes |
|---|---|
| Minutes | `M1` `M2` `M3` `M4` `M5` `M6` `M10` `M12` `M15` `M20` `M30` |
| Hours | `H1` `H2` `H3` `H4` `H6` `H8` `H12` |
| Daily+ | `D1` `W1` `MN1` |

---

## How Self-Learning Works

The model **never peeks at the future**. For each new bar:

1. **Observe** — receive OHLCV from MT5
2. **Learn** — update weights using the *previous* bar's prediction error
3. **Extract** — compute symplectic (capacity, perimeter) + TDA (Betti numbers, persistence) features
4. **Predict** — forecast next-bar log-return with uncertainty
5. **Signal** — translate forecast into BUY / SELL / HOLD

This is pure **walk-forward online learning** — the model starts with zero knowledge and improves with every bar it processes.

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
