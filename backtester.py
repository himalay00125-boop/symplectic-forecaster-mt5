#!/usr/bin/env python3
"""
backtester.py — Standalone Symplectic Backtesting Engine
=========================================================
Downloads historical data from MT5, runs the full Symplectic pipeline
(including ICT features and multi-timeframe alignment), simulates trades,
and produces a detailed performance report with equity curve visualization.

Usage:
  python backtester.py --symbol XAUUSD --timeframe M15 --years 1
  python backtester.py --symbol EURUSD --timeframe H1 --years 2 --confidence 0.5
  python backtester.py --symbol US500 --timeframe M15 --train-bars 2000 --test-bars 1000
"""

import argparse
import datetime
import math
import sys
import time
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Windows workaround for Ray
os.environ['RAY_ENABLE_WINDOWS_ORPHAN_SAFE'] = '0'

# ---------------------------------------------------------------------------
# Import the core engine from the main module
# ---------------------------------------------------------------------------
try:
    from symplectic_forecaster import (
        Bar, SymplecticForecaster, TradingEngine,
        MT5Connection, RiskConfig, TIMEFRAME_MAP,
        HAS_MT5, CapacityRecord,
    )
except ImportError as e:
    print(f"[ERROR] Cannot import symplectic_forecaster: {e}")
    print("        Make sure symplectic_forecaster.py is in the same directory.")
    sys.exit(1)

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

try:
    import pandas as pd
except ImportError:
    pd = None


# ===========================================================================
# BACKTEST TRADE SIMULATOR
# ===========================================================================

@dataclass
class BTPosition:
    """An open simulated position."""
    direction: str
    entry_price: float
    volume: float
    sl: float
    tp: float
    entry_time: float
    entry_bar_idx: int
    entry_confidence: float
    trailing_sl: float = 0.0
    max_favorable: float = 0.0  # Max favorable excursion (points)
    max_adverse: float = 0.0    # Max adverse excursion (points)


@dataclass
class BTClosedTrade:
    """A closed simulated trade with full metrics."""
    direction: str
    entry_price: float
    exit_price: float
    volume: float
    sl: float
    tp: float
    pnl_dollars: float
    entry_time: float
    exit_time: float
    exit_reason: str
    entry_confidence: float
    holding_bars: int
    mfe: float  # Max Favorable Excursion (points)
    mae: float  # Max Adverse Excursion (points)


@dataclass
class BTResult:
    """Complete backtest results with all statistics."""
    # Trade list
    trades: List[BTClosedTrade] = field(default_factory=list)
    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    equity_timestamps: List[float] = field(default_factory=list)
    # Summary stats
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_bars: float = 0.0
    # Model stats
    total_signals: int = 0
    buys: int = 0
    sells: int = 0
    holds: int = 0
    skipped_risk: int = 0


class BacktestEngine:
    """
    Offline trade simulator that runs the SymplecticForecaster on
    historical MT5 data and tracks full portfolio metrics.
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_pct: float = 1.0,
        reward_risk: float = 2.0,
        atr_sl_mult: float = 1.5,
        max_atr_sl_mult: float = 3.5,
        confidence_threshold: float = 0.4,
        max_daily_loss_pct: float = 3.0,
        spread_points: float = 0.0,
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.equity_peak = initial_balance
        self.risk_pct = risk_pct
        self.reward_risk = reward_risk
        self.atr_sl_mult = atr_sl_mult
        self.max_atr_sl_mult = max_atr_sl_mult
        self.confidence_threshold = confidence_threshold
        self.max_daily_loss_pct = max_daily_loss_pct
        self.spread_points = spread_points

        self.position: Optional[BTPosition] = None
        self.closed_trades: List[BTClosedTrade] = []
        self.equity_curve: List[float] = [initial_balance]
        self.equity_timestamps: List[float] = [0.0]

        # Daily tracking
        self._day_start_balance = initial_balance
        self._current_day = None
        self._halted = False

        # Signal counters
        self._buys = 0
        self._sells = 0
        self._holds = 0
        self._skipped = 0

        # ATR buffer for SL calculation
        self._recent_bars: List[Bar] = []

    def _compute_atr(self, period: int = 14) -> float:
        """Compute ATR from recent bars."""
        if len(self._recent_bars) < 2:
            return 0.0
        trs = []
        for i in range(1, min(len(self._recent_bars), period + 1)):
            bar = self._recent_bars[-i]
            prev = self._recent_bars[-(i + 1)] if i + 1 <= len(self._recent_bars) else bar
            tr = max(
                bar.high - bar.low,
                abs(bar.high - prev.close),
                abs(bar.low - prev.close),
            )
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    def _check_daily_reset(self, bar: Bar):
        """Reset daily loss tracking on new day."""
        try:
            bar_day = datetime.datetime.utcfromtimestamp(bar.timestamp).date()
        except (OSError, ValueError):
            return
        if self._current_day != bar_day:
            self._current_day = bar_day
            self._day_start_balance = self.balance
            self._halted = False

    def _check_daily_limit(self) -> bool:
        """Check if daily loss limit is hit."""
        if self._day_start_balance <= 0:
            return True
        loss_pct = (self._day_start_balance - self.balance) / self._day_start_balance * 100
        if loss_pct >= self.max_daily_loss_pct:
            self._halted = True
            return True
        return False

    def process_bar(self, bar: Bar, forecast: Optional[Dict], bar_idx: int):
        """Process one bar: check exits, then check entries."""
        self._recent_bars.append(bar)
        if len(self._recent_bars) > 100:
            self._recent_bars = self._recent_bars[-100:]

        self._check_daily_reset(bar)

        # --- Check exit on open position ---
        if self.position is not None:
            self._check_exit(bar, bar_idx)

        # --- Check entry from forecast ---
        if forecast is not None and self.position is None and not self._halted:
            self._check_entry(bar, forecast, bar_idx)

        # Track equity
        unrealized = 0.0
        if self.position is not None:
            if self.position.direction == "BUY":
                unrealized = (bar.close - self.position.entry_price) * self.position.volume * 100
            else:
                unrealized = (self.position.entry_price - bar.close) * self.position.volume * 100
        self.equity_curve.append(self.balance + unrealized)
        self.equity_timestamps.append(bar.timestamp)

    def _check_exit(self, bar: Bar, bar_idx: int):
        """Check if SL or TP is hit."""
        pos = self.position
        if pos is None:
            return

        # Update MFE/MAE using this bar's high/low
        if pos.direction == "BUY":
            # Favorable: price went up (bar.high - entry_price)
            favorable = bar.high - pos.entry_price
            # Adverse: price went down (entry_price - bar.low)
            adverse = pos.entry_price - bar.low
        else:  # SELL
            # Favorable: price went down (entry_price - bar.low)
            favorable = pos.entry_price - bar.low
            # Adverse: price went up (bar.high - entry_price)
            adverse = bar.high - pos.entry_price

        if favorable > pos.max_favorable:
            pos.max_favorable = favorable
        if adverse > pos.max_adverse:
            pos.max_adverse = adverse

        exit_price = None
        exit_reason = None

        if pos.direction == "BUY":
            # Check SL first (worse outcome), then TP
            if bar.low <= pos.sl:
                exit_price = pos.sl
                exit_reason = "SL"
            elif bar.high >= pos.tp:
                exit_price = pos.tp
                exit_reason = "TP"
        else:  # SELL
            if bar.high >= pos.sl:
                exit_price = pos.sl
                exit_reason = "SL"
            elif bar.low <= pos.tp:
                exit_price = pos.tp
                exit_reason = "TP"

        if exit_price is not None:
            self._close_position(exit_price, exit_reason, bar, bar_idx)

    def _close_position(self, exit_price: float, reason: str, bar: Bar, bar_idx: int):
        """Close the current position and record PnL."""
        pos = self.position
        if pos is None:
            return

        if pos.direction == "BUY":
            pnl_points = exit_price - pos.entry_price
        else:
            pnl_points = pos.entry_price - exit_price

        # Approximate PnL in dollars (simplified: use tick_value heuristic)
        pnl_dollars = pnl_points * pos.volume * 100  # Rough approximation

        holding_bars = bar_idx - pos.entry_bar_idx

        self.closed_trades.append(BTClosedTrade(
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            volume=pos.volume,
            sl=pos.sl,
            tp=pos.tp,
            pnl_dollars=pnl_dollars,
            entry_time=pos.entry_time,
            exit_time=bar.timestamp,
            exit_reason=reason,
            entry_confidence=pos.entry_confidence,
            holding_bars=holding_bars,
            mfe=pos.max_favorable,
            mae=pos.max_adverse,
        ))

        self.balance += pnl_dollars
        self.equity_peak = max(self.equity_peak, self.balance)
        self.position = None

    def _check_entry(self, bar: Bar, forecast: Dict, bar_idx: int):
        """Evaluate forecast and open a position if conditions met."""
        confidence = forecast.get("confidence", 0.0)
        direction_val = forecast.get("direction", 0)

        if confidence < self.confidence_threshold:
            self._holds += 1
            return

        if direction_val > 0:
            direction = "BUY"
            self._buys += 1
        elif direction_val < 0:
            direction = "SELL"
            self._sells += 1
        else:
            self._holds += 1
            return

        # Compute SL/TP
        atr = self._compute_atr()
        if atr <= 0:
            self._skipped += 1
            return

        # Dynamic SL based on confidence
        sl_dist = atr * self.atr_sl_mult

        # Cap at max ATR multiplier
        max_dist = atr * self.max_atr_sl_mult
        if sl_dist > max_dist:
            sl_dist = max_dist

        # Dynamic R:R based on confidence
        rr = self.reward_risk
        if confidence > 0.8:
            rr *= 1.5
        elif confidence < 0.65:
            rr *= 0.8

        tp_dist = sl_dist * rr

        # Add spread
        entry_price = bar.close
        if direction == "BUY":
            entry_price += self.spread_points
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            entry_price -= self.spread_points
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        # Calculate lot size based on risk
        risk_amount = self.balance * (self.risk_pct / 100.0)
        loss_per_lot = sl_dist * 100  # Simplified
        if loss_per_lot <= 0:
            self._skipped += 1
            return

        lots = risk_amount / loss_per_lot
        lots = math.floor(lots * 100) / 100  # Round down to 0.01
        
        if lots < 0.01:
            lots = 0.01
            new_sl_dist = risk_amount / (lots * 100.0)
            min_dist = max(self.spread_points, 0.00001)
            new_sl_dist = max(new_sl_dist, min_dist)
            
            tp_dist = new_sl_dist * rr
            sl_dist = new_sl_dist
            
            if direction == "BUY":
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist

        lots = min(lots, 10.0)

        # Check daily loss limit
        if self._check_daily_limit():
            self._skipped += 1
            return

        self.position = BTPosition(
            direction=direction,
            entry_price=entry_price,
            volume=lots,
            sl=sl,
            tp=tp,
            entry_time=bar.timestamp,
            entry_bar_idx=bar_idx,
            entry_confidence=confidence,
        )

    def get_results(self) -> BTResult:
        """Compute final statistics."""
        result = BTResult()
        result.trades = self.closed_trades
        result.equity_curve = self.equity_curve
        result.equity_timestamps = self.equity_timestamps
        result.initial_balance = self.initial_balance
        result.final_balance = self.balance
        result.total_trades = len(self.closed_trades)
        result.buys = self._buys
        result.sells = self._sells
        result.holds = self._holds
        result.skipped_risk = self._skipped
        result.total_signals = self._buys + self._sells + self._holds

        if result.total_trades == 0:
            return result

        wins = [t for t in self.closed_trades if t.pnl_dollars > 0]
        losses = [t for t in self.closed_trades if t.pnl_dollars <= 0]

        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / result.total_trades * 100
        result.total_pnl = sum(t.pnl_dollars for t in self.closed_trades)
        result.avg_win = sum(t.pnl_dollars for t in wins) / len(wins) if wins else 0
        result.avg_loss = sum(t.pnl_dollars for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl_dollars for t in wins)
        gross_loss = abs(sum(t.pnl_dollars for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        result.avg_holding_bars = (
            sum(t.holding_bars for t in self.closed_trades) / result.total_trades
        )

        # Max drawdown
        peak = self.initial_balance
        max_dd_dollars = 0
        max_dd_pct = 0
        for eq in self.equity_curve:
            peak = max(peak, eq)
            dd = peak - eq
            dd_pct = dd / peak * 100 if peak > 0 else 0
            max_dd_dollars = max(max_dd_dollars, dd)
            max_dd_pct = max(max_dd_pct, dd_pct)
        result.max_drawdown_dollars = max_dd_dollars
        result.max_drawdown_pct = max_dd_pct

        # Sharpe ratio (annualized, assuming daily returns)
        returns = []
        for i in range(1, len(self.equity_curve)):
            if self.equity_curve[i - 1] > 0:
                returns.append(
                    (self.equity_curve[i] - self.equity_curve[i - 1])
                    / self.equity_curve[i - 1]
                )
        if returns and np.std(returns) > 0:
            result.sharpe_ratio = (np.mean(returns) / np.std(returns)) * math.sqrt(252)
        else:
            result.sharpe_ratio = 0.0

        return result


# ===========================================================================
# REPORT GENERATOR
# ===========================================================================

def print_report(result: BTResult, symbol: str, timeframe: str):
    """Print a beautifully formatted backtest report to console."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    pnl_color = GREEN if result.total_pnl >= 0 else RED

    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"  {CYAN}{BOLD}SYMPLECTIC BACKTEST REPORT{RESET}")
    print(f"  {DIM}{symbol} | {timeframe} | {result.total_trades} trades{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}\n")

    print(f"  {BOLD}Portfolio Summary{RESET}")
    print(f"  {'-' * 50}")
    print(f"  Initial Balance:     ${result.initial_balance:,.2f}")
    print(f"  Final Balance:       {pnl_color}${result.final_balance:,.2f}{RESET}")
    print(f"  Net P&L:             {pnl_color}${result.total_pnl:+,.2f}{RESET}")
    ret_pct = (result.final_balance - result.initial_balance) / result.initial_balance * 100
    print(f"  Return:              {pnl_color}{ret_pct:+.2f}%{RESET}")
    print()

    print(f"  {BOLD}Trade Statistics{RESET}")
    print(f"  {'-' * 50}")
    print(f"  Total Trades:        {result.total_trades}")
    print(f"  Winners:             {GREEN}{result.winning_trades}{RESET}")
    print(f"  Losers:              {RED}{result.losing_trades}{RESET}")
    print(f"  Win Rate:            {result.win_rate:.1f}%")
    print(f"  Avg Win:             {GREEN}${result.avg_win:+,.2f}{RESET}")
    print(f"  Avg Loss:            {RED}${result.avg_loss:+,.2f}{RESET}")
    pf_str = f"{result.profit_factor:.2f}" if result.profit_factor < 1e6 else "∞"
    print(f"  Profit Factor:       {pf_str}")
    print(f"  Avg Holding (bars):  {result.avg_holding_bars:.1f}")
    print()

    print(f"  {BOLD}Risk Metrics{RESET}")
    print(f"  {'-' * 50}")
    dd_color = RED if result.max_drawdown_pct > 10 else YELLOW
    print(f"  Max Drawdown:        {dd_color}{result.max_drawdown_pct:.2f}% "
          f"(${result.max_drawdown_dollars:,.2f}){RESET}")
    sr_color = GREEN if result.sharpe_ratio > 1 else YELLOW if result.sharpe_ratio > 0 else RED
    print(f"  Sharpe Ratio:        {sr_color}{result.sharpe_ratio:.2f}{RESET}")
    print()

    print(f"  {BOLD}Signal Breakdown{RESET}")
    print(f"  {'-' * 50}")
    print(f"  Total Signals:       {result.total_signals}")
    print(f"  BUY signals:         {GREEN}{result.buys}{RESET}")
    print(f"  SELL signals:        {RED}{result.sells}{RESET}")
    print(f"  HOLD signals:        {YELLOW}{result.holds}{RESET}")
    print(f"  Skipped (risk):      {DIM}{result.skipped_risk}{RESET}")
    print(f"\n{BOLD}{'=' * 66}{RESET}")


def export_trades_csv(result: BTResult, path: str):
    """Export all trades to a CSV file."""
    if pd is None:
        print("[WARN] pandas not installed — cannot export CSV.")
        return

    rows = []
    for t in result.trades:
        rows.append({
            "direction": t.direction,
            "entry_price": round(t.entry_price, 5),
            "exit_price": round(t.exit_price, 5),
            "volume": t.volume,
            "sl": round(t.sl, 5),
            "tp": round(t.tp, 5),
            "pnl": round(t.pnl_dollars, 2),
            "exit_reason": t.exit_reason,
            "confidence": round(t.entry_confidence, 4),
            "holding_bars": t.holding_bars,
            "entry_time": datetime.datetime.utcfromtimestamp(t.entry_time).strftime("%Y-%m-%d %H:%M"),
            "exit_time": datetime.datetime.utcfromtimestamp(t.exit_time).strftime("%Y-%m-%d %H:%M"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"  [EXPORT] Trades saved to {path}")


def plot_equity_curve(result: BTResult, symbol: str, timeframe: str, path: str):
    """Save equity curve as a PNG image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping equity chart.")
        return

    timestamps = [
        datetime.datetime.utcfromtimestamp(t) for t in result.equity_timestamps
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True)
    fig.suptitle(f"Symplectic Backtest — {symbol} {timeframe}", fontsize=14, fontweight="bold")

    # Equity curve
    ax1.plot(timestamps, result.equity_curve, color="#2196F3", linewidth=1.2, label="Equity")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.5,
                label="Initial Balance")
    ax1.fill_between(timestamps, result.initial_balance, result.equity_curve,
                     where=[e >= result.initial_balance for e in result.equity_curve],
                     color="#4CAF50", alpha=0.15)
    ax1.fill_between(timestamps, result.initial_balance, result.equity_curve,
                     where=[e < result.initial_balance for e in result.equity_curve],
                     color="#F44336", alpha=0.15)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown curve
    peak = result.initial_balance
    drawdown = []
    for eq in result.equity_curve:
        peak = max(peak, eq)
        dd_pct = (peak - eq) / peak * 100 if peak > 0 else 0
        drawdown.append(-dd_pct)
    ax2.fill_between(timestamps, 0, drawdown, color="#F44336", alpha=0.4)
    ax2.plot(timestamps, drawdown, color="#D32F2F", linewidth=0.8)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [CHART] Equity curve saved to {path}")


# ===========================================================================
# MAIN CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Symplectic Backtesting Engine — Offline Historical Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtester.py --symbol XAUUSD --timeframe M15 --years 1
  python backtester.py --symbol EURUSD --timeframe H1 --years 2 --confidence 0.5
  python backtester.py --symbol US500 --timeframe M15 --train-bars 2000 --test-bars 1000
        """,
    )
    parser.add_argument("--symbol", type=str, required=True,
                        help="MT5 symbol to backtest (e.g., XAUUSD, EURUSD)")
    parser.add_argument("--timeframe", type=str, default="M15",
                        help="Timeframe: M1, M5, M15, M30, H1, H4, D1 (default: M15)")
    parser.add_argument("--train-bars", type=int, default=1000,
                        help="Bars for initial model warm-up (default: 1000)")
    parser.add_argument("--test-bars", type=int, default=2000,
                        help="Bars for out-of-sample testing (default: 2000)")
    parser.add_argument("--years", type=float, default=None,
                        help="Alternative: specify years of test data (overrides --test-bars)")
    parser.add_argument("--initial-balance", type=float, default=10000.0,
                        help="Starting balance in USD (default: 10000)")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Confidence threshold for trade entry (default: 0.4)")
    parser.add_argument("--risk-pct", type=float, default=1.0,
                        help="Risk per trade as %% of equity (default: 1.0)")
    parser.add_argument("--reward-risk", type=float, default=2.0,
                        help="Base reward:risk ratio (default: 2.0)")
    parser.add_argument("--atr-sl", type=float, default=1.5,
                        help="ATR multiplier for stop-loss (default: 1.5)")
    parser.add_argument("--max-atr-sl", type=float, default=3.5,
                        help="Max ATR multiplier cap for SL (default: 3.5)")
    parser.add_argument("--spread", type=float, default=0.0,
                        help="Simulated spread in price points (default: 0)")
    parser.add_argument("--window", type=int, default=60,
                        help="Symplectic rolling window size (default: 60)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip equity curve chart generation")
    parser.add_argument("--export", type=str, default=None,
                        help="Export trades CSV path (default: auto-generated)")

    # MT5 connection args
    parser.add_argument("--account", type=int, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--server", type=str, default=None)
    parser.add_argument("--mt5-path", type=str, default=None)

    args = parser.parse_args()

    # == Banner ==
    CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"; DIM = "\033[2m"
    print(f"\n{BOLD}{'=' * 66}{RESET}")
    print(f"  {CYAN}{BOLD}SYMPLECTIC BACKTESTING ENGINE{RESET}")
    print(f"  {DIM}Offline historical simulation with ICT + MTF features{RESET}")
    print(f"{BOLD}{'=' * 66}{RESET}\n")

    if not HAS_MT5:
        print("[ERROR] MetaTrader5 not installed. pip install MetaTrader5")
        sys.exit(1)

    # ── Connect to MT5 ──
    conn = MT5Connection()
    try:
        conn.connect(account=args.account, password=args.password,
                     server=args.server, path=args.mt5_path)
    except ConnectionError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    tf_str = args.timeframe.upper()
    if tf_str not in TIMEFRAME_MAP:
        print(f"[ERROR] Unknown timeframe '{tf_str}'.")
        conn.disconnect()
        sys.exit(1)

    conn.ensure_symbol(args.symbol)

    # ── Calculate bar counts ──
    tf_mt5 = TIMEFRAME_MAP[tf_str]

    # If --years specified, calculate test bars from timeframe
    if args.years is not None:
        bars_per_day = {
            "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
            "H1": 24, "H4": 6, "D1": 1, "W1": 0.2,
        }
        bpd = bars_per_day.get(tf_str, 24)
        args.test_bars = int(args.years * 252 * bpd)
        print(f"  {DIM}Years={args.years} -> {args.test_bars} test bars for {tf_str}{RESET}")

    total_bars = args.train_bars + args.test_bars

    print(f"  Symbol:         {args.symbol}")
    print(f"  Timeframe:      {tf_str}")
    print(f"  Training bars:  {args.train_bars}")
    print(f"  Test bars:      {args.test_bars}")
    print(f"  Initial balance: ${args.initial_balance:,.2f}")
    print(f"  Confidence:     {args.confidence}")
    print(f"  Risk/trade:     {args.risk_pct}%")
    print(f"  R:R ratio:      1:{args.reward_risk}")
    print()

    # ── Fetch historical data ──
    print(f"  [MT5] Fetching {total_bars} bars of {args.symbol} ({tf_str})...")
    rates = mt5.copy_rates_from_pos(args.symbol, tf_mt5, 0, total_bars)

    if rates is None or len(rates) < 100:
        print(f"[ERROR] Insufficient data: got {len(rates) if rates is not None else 0} bars.")
        conn.disconnect()
        sys.exit(1)

    print(f"  [MT5] Received {len(rates)} bars.")

    # Convert to Bar objects
    all_bars: List[Bar] = []
    for r in rates:
        vol = float(r["real_volume"]) if r["real_volume"] > 0 else float(r["tick_volume"])
        all_bars.append(Bar(
            timestamp=float(r["time"]),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=vol,
        ))

    # Split into train/test
    split_idx = min(args.train_bars, len(all_bars) - 100)
    train_bars = all_bars[:split_idx]
    test_bars = all_bars[split_idx:]

    print(f"  [DATA] Training on {len(train_bars)} bars, testing on {len(test_bars)} bars.")
    first_dt = datetime.datetime.utcfromtimestamp(train_bars[0].timestamp).strftime("%Y-%m-%d")
    split_dt = datetime.datetime.utcfromtimestamp(test_bars[0].timestamp).strftime("%Y-%m-%d")
    last_dt = datetime.datetime.utcfromtimestamp(test_bars[-1].timestamp).strftime("%Y-%m-%d")
    print(f"  [DATA] Train: {first_dt} -> {split_dt}")
    print(f"  [DATA] Test:  {split_dt} -> {last_dt}")
    print()

    # ── Initialize forecaster ──
    fc = SymplecticForecaster(
        window=args.window,
        alert_pct=0.95,
        min_train_bars=80,
    )
    fc._mtf_symbol = args.symbol
    fc._mtf_tf_str = tf_str

    # ── Phase 1: Warm-up training ──
    print(f"  [TRAIN] Running symplectic pipeline on {len(train_bars)} training bars...")
    t0 = time.time()
    for bar in train_bars:
        fc.process_bar(bar)
    train_time = time.time() - t0
    print(f"  [TRAIN] Complete in {train_time:.1f}s. "
          f"Model updated {fc._model._n_updates} times.")
    print()

    # ── Phase 2: Out-of-sample simulation ──
    bt = BacktestEngine(
        initial_balance=args.initial_balance,
        risk_pct=args.risk_pct,
        reward_risk=args.reward_risk,
        atr_sl_mult=args.atr_sl,
        max_atr_sl_mult=args.max_atr_sl,
        confidence_threshold=args.confidence,
        max_daily_loss_pct=3.0,
        spread_points=args.spread,
    )

    print(f"  [TEST] Simulating trades on {len(test_bars)} out-of-sample bars...")
    t0 = time.time()
    for i, bar in enumerate(test_bars):
        forecast = fc.process_bar(bar)
        bt.process_bar(bar, forecast, i)

        # Progress indicator every 500 bars
        if (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{len(test_bars)} bars processed "
                  f"({len(bt.closed_trades)} trades so far)")

    # Close any remaining position at last bar
    if bt.position is not None:
        bt._close_position(test_bars[-1].close, "END", test_bars[-1], len(test_bars) - 1)

    test_time = time.time() - t0
    print(f"  [TEST] Complete in {test_time:.1f}s.")
    print()

    # ── Results ──
    result = bt.get_results()
    print_report(result, args.symbol, tf_str)

    # ── Export ──
    export_path = args.export or f"bt_{args.symbol}_{tf_str}_trades.csv"
    export_trades_csv(result, export_path)

    if not args.no_chart:
        chart_path = f"bt_{args.symbol}_{tf_str}_equity.png"
        plot_equity_curve(result, args.symbol, tf_str, chart_path)

    conn.disconnect()
    print(f"\n{BOLD}[DONE]{RESET} Backtest complete.")


if __name__ == "__main__":
    main()
