"""Technical indicator helpers used by the manual prediction form."""

from __future__ import annotations

import pandas as pd


DEFAULT_OPTIONAL_INDICATORS = {
    "rsi_14": 50.0,
    "volatility_20d_percent": 20.0,
    "max_drawdown_percent": -10.0,
    "open_gap_percent": 0.0,
    "volume_rel_20d_percent": 100.0,
}


def normalize_b3_ticker(stock_code: str | None) -> str | None:
    if not stock_code:
        return None

    ticker = str(stock_code).strip().upper()
    if not ticker:
        return None
    if "." not in ticker:
        ticker = f"{ticker}.SA"
    return ticker


def compute_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.dropna().iloc[-1])


def _flatten_yfinance_columns(history: pd.DataFrame) -> pd.DataFrame:
    if isinstance(history.columns, pd.MultiIndex):
        history = history.copy()
        history.columns = history.columns.get_level_values(0)
    return history


def calculate_optional_indicators(
    stock_code: str | None,
    open_price: float | None = None,
    period: str = "1y",
) -> dict:
    """Calculate the optional indicators from recent Yahoo Finance history.

    Returned values are in the same human-readable units used by the dashboard:
    percentages are returned as percentage points, not decimals.
    """
    ticker = normalize_b3_ticker(stock_code)
    if ticker is None:
        raise ValueError("Ticker nao informado para calcular indicadores.")

    import yfinance as yf

    history = yf.Ticker(ticker).history(period=period, auto_adjust=False)
    history = _flatten_yfinance_columns(history)
    if history.empty or len(history) < 21:
        raise ValueError(f"Historico insuficiente para {ticker}.")

    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    open_series = pd.to_numeric(history["Open"], errors="coerce").dropna()
    volume = pd.to_numeric(history["Volume"], errors="coerce").dropna()
    if len(close) < 21 or len(open_series) < 1 or len(volume) < 20:
        raise ValueError(f"Historico insuficiente para {ticker}.")

    returns = close.pct_change()
    latest_close = float(close.iloc[-1])
    latest_open = float(open_series.iloc[-1])
    previous_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close
    open_reference = float(open_price) if open_price is not None else latest_open

    running_max = close.cummax()
    max_drawdown = ((close - running_max) / (running_max + 1e-9)).min() * 100
    volume_rel = (float(volume.iloc[-1]) / (float(volume.tail(20).mean()) + 1e-9)) * 100
    open_gap = ((open_reference / (previous_close + 1e-9)) - 1) * 100

    return {
        "rsi_14": compute_rsi(close, 14),
        "volatility_20d_percent": float(returns.tail(20).std() * 100),
        "max_drawdown_percent": float(max_drawdown),
        "open_gap_percent": float(open_gap),
        "volume_rel_20d_percent": float(volume_rel),
        "ticker": ticker,
        "open_used_for_gap": open_reference,
        "last_close": latest_close,
        "last_date": str(close.index[-1].date()),
    }
