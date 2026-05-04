"""
Página de Recomendações para Amanhã
Com análise histórica e dados para decisão
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
import plotly.graph_objects as go
import plotly.express as px

from src.data_ingestion import DataIngestor


def _format_number(value, kind="number"):
    if value is None:
        return "n/d"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if kind == "percent":
        return f"{numeric * 100:.2f}%"
    if abs(numeric) >= 1e12:
        return f"{numeric / 1e12:.2f}T"
    if abs(numeric) >= 1e9:
        return f"{numeric / 1e9:.2f}B"
    if abs(numeric) >= 1e6:
        return f"{numeric / 1e6:.2f}M"
    return f"{numeric:.2f}"


def _format_date_br(value):
    if value is None or pd.isna(value):
        return "n/d"
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def _format_money(value):
    if value is None or pd.isna(value):
        return "n/d"
    return f"R$ {float(value):.2f}"


def _format_percent_points(value, signed=False):
    if value is None or pd.isna(value):
        return "n/d"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.2f}%"


def _format_sma(row, column, window):
    value = row.get(column)
    days = row.get(f"{column}_days", row.get("historical_days"))
    if value is None or pd.isna(value):
        if days is None or pd.isna(days):
            return f"n/d (<{window} pregões)"
        return f"n/d ({int(days)}/{window} pregões)"
    return f"{float(value):.2f}"


def _pick_row(frame, candidates):
    if frame is None or frame.empty:
        return None
    lower_index = {str(idx).lower(): idx for idx in frame.index}
    for candidate in candidates:
        matched = lower_index.get(candidate.lower())
        if matched is not None:
            series = frame.loc[matched].dropna()
            if not series.empty:
                return float(series.iloc[0])
    return None


def _cagr_from_series(series):
    if series is None:
        return None
    cleaned = pd.Series(series).dropna()
    cleaned = cleaned[cleaned > 0]
    if len(cleaned) < 2:
        return None
    start = float(cleaned.iloc[-1])
    end = float(cleaned.iloc[0])
    periods = max(len(cleaned) - 1, 1)
    return (end / start) ** (1 / periods) - 1


def _compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def _compute_buy_potential_score(frame):
    risk_score = frame['risk_score'] if 'risk_score' in frame.columns else pd.Series(100, index=frame.index)
    score = (
        frame['probability_up'].fillna(0) * 0.38
        + frame['confidence_score'].fillna(0) * 100 * 0.22
        + (100 - risk_score.fillna(100)) * 0.20
        + frame['trend_20days'].clip(lower=0).fillna(0).clip(0, 20) * 0.60
        + np.where(frame['rsi_14'].between(30, 60), 6, 0)
    )
    return score


def _build_horizon_candidates(frame):
    working = frame.copy()
    risk_score = working['risk_score'].fillna(100)
    probability_up = working['probability_up'].fillna(0)
    trend_20days = working['trend_20days'].fillna(0)
    rsi = working['rsi_14'].fillna(50)
    variation = working['variation_percent'].fillna(0)
    accuracy = working['accuracy_mean'].fillna(0)
    confidence = working['confidence_score'].fillna(0) * 100
    volatility = working['volatility'].fillna(0)
    drawdown = working['max_drawdown_percent'].abs().fillna(0)

    working['score_curto'] = (
        probability_up * 0.34
        + variation.clip(-3, 3) * 6.0
        + trend_20days.clip(-15, 15) * 0.75
        + np.where(rsi.between(30, 62), 8.0, 0.0)
        - risk_score * 0.22
        - np.where(working['recommendation'].eq('VENDA'), 18.0, 0.0)
    )
    working['score_medio'] = (
        probability_up * 0.25
        + confidence * 0.22
        + accuracy * 0.16
        + trend_20days.clip(-20, 20) * 0.95
        - risk_score * 0.25
        - volatility * 1.25
        + np.where(working['trend_regime'].isin(['Alta', 'Lateral']), 6.0, 0.0)
    )
    working['score_longo'] = (
        confidence * 0.26
        + accuracy * 0.24
        + probability_up * 0.14
        - risk_score * 0.34
        - drawdown.clip(0, 80) * 0.12
        - volatility * 0.75
        + np.where(working['trend_regime'].eq('Alta'), 8.0, 0.0)
        + np.where(rsi.between(35, 65), 4.0, 0.0)
    )

    cols = [
        'company', 'stock_code', 'recommendation', 'probability_up',
        'variation_percent', 'trend_20days', 'rsi_14', 'risk_score',
        'confidence', 'score_curto', 'score_medio', 'score_longo'
    ]
    labels = {
        'company': 'Empresa',
        'stock_code': 'Ticker',
        'recommendation': 'Sinal',
        'probability_up': 'Prob. Alta (%)',
        'variation_percent': 'Variação D+1 (%)',
        'trend_20days': 'Tendência 20d (%)',
        'rsi_14': 'RSI',
        'risk_score': 'Risco',
        'confidence': 'Confiança',
        'score_curto': 'Score Curto',
        'score_medio': 'Score Médio',
        'score_longo': 'Score Longo',
    }

    return {
        'Curto prazo': working.sort_values(['score_curto', 'probability_up'], ascending=[False, False])[cols].head(8).rename(columns=labels),
        'Médio prazo': working.sort_values(['score_medio', 'risk_score'], ascending=[False, True])[cols].head(8).rename(columns=labels),
        'Longo prazo': working.sort_values(['score_longo', 'risk_score'], ascending=[False, True])[cols].head(8).rename(columns=labels),
    }


def _compute_portfolio_score(frame):
    accuracy = frame['accuracy_mean'].fillna(0)
    directional_accuracy = frame['directional_accuracy_global'].fillna(accuracy)
    confidence = frame['confidence_score'].fillna(0) * 100
    probability_up = frame['probability_up'].fillna(0)
    probability_down = frame['probability_down'].fillna(100)
    trend_20days = frame['trend_20days'].fillna(0)
    risk_score = frame['risk_score'].fillna(100)
    drawdown = frame['max_drawdown_percent'].abs().fillna(0)
    rsi = frame['rsi_14'].fillna(50)

    score = (
        accuracy * 0.28
        + directional_accuracy * 0.10
        + confidence * 0.18
        + probability_up * 0.18
        + np.clip(trend_20days, -20, 20) * 1.40
        - risk_score * 0.25
        - probability_down * 0.12
        - np.clip(drawdown, 0, 100) * 0.04
        + np.where(rsi.between(30, 65), 5.0, 0.0)
        + np.where(trend_20days < 0, trend_20days * 0.90, 0.0)
        + np.where(probability_up > probability_down, 4.0, -6.0)
        + np.where(frame['recommendation'].eq('COMPRA'), 8.0, 0.0)
        + np.where(frame['recommendation'].eq('VENDA'), -18.0, 0.0)
        + np.where(frame['recommendation'].eq('ANALISAR'), -4.0, 0.0)
    )
    return pd.Series(score, index=frame.index)


def _portfolio_entry_note(row):
    if row['trend_20days'] < 0 and row['probability_down'] > row['probability_up']:
        return 'Tendência de queda ainda pesa; manter peso pequeno ou aguardar confirmação.'
    if row['trend_20days'] < 0 and row['rsi_14'] < 35 and row['probability_up'] >= row['probability_down']:
        return 'Sinal de possível recuperação; entrada fracionada faz mais sentido.'
    if row['trend_20days'] > 0 and row['probability_up'] > row['probability_down'] and row['risk_score'] < 50:
        return 'Tendência, probabilidade e risco estão alinhados; pode receber peso maior.'
    if row['recommendation'] == 'VENDA':
        return 'Evitar posição nova agora; prioridade é proteção de capital.'
    return 'Carteira com viés moderado; posição intermediária e acompanhamento próximo.'


def _build_portfolio_plan(frame, budget, basket_size, conservative_mode):
    if frame.empty:
        return frame.copy(), 'Sem dados suficientes para montar uma carteira.'

    working = frame.copy()
    working['portfolio_score'] = _compute_portfolio_score(working)

    if conservative_mode:
        protected_mask = (
            (working['probability_down'] > working['probability_up'])
            & (working['trend_20days'] < 0)
            & (working['risk_score'] >= 55)
        )
        filtered = working.loc[~protected_mask].copy()
        if filtered.empty:
            filtered = working.copy()
            opinion = 'O filtro conservador removeu tudo; usando o universo completo para não travar a carteira.'
        else:
            opinion = 'Modo conservador ativado: papéis com forte viés de queda foram penalizados ou removidos.'
    else:
        filtered = working.copy()
        opinion = 'Modo amplo ativado: a carteira considera todo o universo, mas ainda penaliza risco e queda.'

    plan = filtered.sort_values(
        ['portfolio_score', 'accuracy_mean', 'risk_score'],
        ascending=[False, False, True]
    ).head(basket_size).copy()

    if plan.empty:
        return plan, 'Sem candidatos após a ordenação final.'

    raw_weights = plan['portfolio_score'].clip(lower=0)
    if raw_weights.sum() <= 0:
        plan['target_weight'] = 100 / len(plan)
    else:
        plan['target_weight'] = (raw_weights / raw_weights.sum()) * 100

    plan['target_amount'] = plan['target_weight'] / 100 * float(budget)
    plan['entry_note'] = plan.apply(_portfolio_entry_note, axis=1)

    avg_score = plan['portfolio_score'].mean()
    avg_accuracy = plan['accuracy_mean'].mean()
    avg_risk = plan['risk_score'].mean()
    avg_trend = plan['trend_20days'].mean()
    down_count = int((plan['probability_down'] > plan['probability_up']).sum())

    if avg_risk >= 65 or down_count >= max(1, len(plan) // 2):
        opinion += ' A composição ainda está defensiva; eu reduziria o aporte total ou esperaria uma melhora no sinal.'
    elif avg_score >= 65 and avg_accuracy >= 55 and avg_trend > 0:
        opinion += ' A leitura geral é positiva: a carteira está bem posicionada para uma entrada gradual.'
    elif avg_score >= 45:
        opinion += ' A carteira está aceitável, mas ainda pede entrada parcelada e revisão frequente.'
    else:
        opinion += ' O sinal geral ainda é fraco; o melhor uso aqui é triagem, não agressividade.'

    return plan, opinion


def _swing_trade_universe():
    # Base estática inicial (curadoria manual)
    core = [
        'ABEV3', 'ALPA4', 'AMER3', 'ASAI3', 'BBDC4', 'CASH3', 'CEAB3', 'CIEL3',
        'COGN3', 'CSNA3', 'CURY3', 'CVCB3', 'CYRE3', 'GGBR4', 'GOAU4', 'HAPV3',
        'ITUB4', 'KLBN11', 'LREN3', 'MGLU3', 'MRVE3', 'OIBR3', 'PCAR3', 'PETR4',
        'PETZ3', 'POSI3', 'PRIO3', 'RADL3', 'RDOR3', 'SANB11', 'SUZB3', 'TEND3',
        'TOTS3', 'UGPA3', 'USIM5', 'VALE3', 'VAMO3', 'VIVA3', 'WEGE3', 'YDUQ3'
    ]

    # Adiciona automaticamente tickers presentes na pasta data/raw (evita sugerir papéis sem arquivos locais)
    try:
        raw_dir = Path('data/raw')
        raw_tickers = []
        if raw_dir.exists() and raw_dir.is_dir():
            import re
            TICKER_RE = re.compile(r'^[A-Z]{1,4}[0-9]{1,2}$')
            for p in raw_dir.glob('*.csv'):
                stem = p.stem.upper()
                if TICKER_RE.match(stem):
                    raw_tickers.append(stem)
    except Exception:
        raw_tickers = []

    universe = list(dict.fromkeys(core + raw_tickers))
    return universe


def _compute_swing_trade_score(frame):
    score = (
        frame['ytd_return_percent'].fillna(0) * 0.26
        + frame['ret_5d_percent'].fillna(0) * 0.16
        + frame['ret_20d_percent'].fillna(0) * 0.18
        + frame['trend_60d_percent'].fillna(0) * 0.12
        + frame['distance_sma20_percent'].fillna(0) * 0.10
        + frame['distance_sma50_percent'].fillna(0) * 0.08
        + np.where(frame['rsi_14'].between(30, 62), 6.0, 0.0)
        + np.where((frame['trend_20d_percent'] > 0) & (frame['trend_60d_percent'] > 0), 10.0, 0.0)
        + np.where((frame['trend_20d_percent'] < 0) & (frame['rsi_14'] < 35), 5.0, 0.0)
        + np.where((frame['trend_20d_percent'] < 0) & (frame['rsi_14'] > 60), -12.0, 0.0)
        - frame['volatility_20d_percent'].fillna(0) * 0.18
        - frame['ytd_drawdown_percent'].abs().fillna(0) * 0.06
    )
    return pd.Series(score, index=frame.index)


def _swing_trade_note(row):
    if row['trend_20d_percent'] > 0 and row['trend_60d_percent'] > 0 and row['rsi_14'] < 65:
        return 'Tendência alinhada para swing trade; entrada mais natural em pullback curto.'
    if row['trend_20d_percent'] < 0 and row['rsi_14'] < 35:
        return 'Possível recuperação curta; ideal fracionar a entrada e respeitar stop.'
    if row['trend_20d_percent'] < 0 and row['rsi_14'] > 60:
        return 'Sem sinal bom de compra agora; melhor esperar virada clara.'
    return 'Movimento intermediário; pode entrar pequeno e revisar a cada pregão.'


def _nearest_trading_date(dates, target_date):
    target = pd.Timestamp(target_date).normalize()
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    eligible = date_index[date_index <= target]
    if len(eligible) == 0:
        return date_index.min()
    return eligible.max()


def _swing_metrics_from_close(close: pd.Series):
    close = pd.Series(close).dropna().astype(float)
    if len(close) < 20:
        return None

    returns = close.pct_change().dropna()
    last_close = float(close.iloc[-1])
    first_close = float(close.iloc[0])
    sma20 = float(close.tail(min(20, len(close))).mean())
    sma50 = float(close.tail(min(50, len(close))).mean())
    high_20 = float(close.tail(min(20, len(close))).max())
    low_20 = float(close.tail(min(20, len(close))).min())
    rsi = float(_compute_rsi_series(close).iloc[-1]) if len(close) >= 14 else np.nan
    volatility_20 = float(returns.tail(min(20, len(returns))).std() * np.sqrt(252) * 100) if len(returns) >= 5 else np.nan

    ret_5d = ((last_close / float(close.iloc[-6])) - 1) * 100 if len(close) >= 6 else np.nan
    ret_20d = ((last_close / float(close.iloc[-21])) - 1) * 100 if len(close) >= 21 else np.nan
    ret_60d = ((last_close / float(close.iloc[-61])) - 1) * 100 if len(close) >= 61 else np.nan
    ytd_return = ((last_close / first_close) - 1) * 100 if first_close else np.nan
    trend_20d = ((last_close / sma20) - 1) * 100 if sma20 else np.nan
    trend_60d = ((last_close / sma50) - 1) * 100 if sma50 else np.nan
    distance_sma20 = ((last_close / sma20) - 1) * 100 if sma20 else np.nan
    distance_sma50 = ((last_close / sma50) - 1) * 100 if sma50 else np.nan
    ytd_drawdown = ((last_close / float(close.cummax().iloc[-1])) - 1) * 100 if len(close) else np.nan
    range_position = ((last_close - low_20) / ((high_20 - low_20) + 1e-9)) * 100

    return {
        'last_close': last_close,
        'first_close_ytd': first_close,
        'ytd_return_percent': ytd_return,
        'ret_5d_percent': ret_5d,
        'ret_20d_percent': ret_20d,
        'ret_60d_percent': ret_60d,
        'trend_20d_percent': trend_20d,
        'trend_60d_percent': trend_60d,
        'distance_sma20_percent': distance_sma20,
        'distance_sma50_percent': distance_sma50,
        'volatility_20d_percent': volatility_20,
        'rsi_14': rsi,
        'ytd_drawdown_percent': ytd_drawdown,
        'range_position_20d': range_position,
    }


@st.cache_data(show_spinner=False)
def _load_year_to_date_history():
    try:
        history = DataIngestor().load_and_merge()
    except Exception:
        return pd.DataFrame()

    if history is None or history.empty:
        return pd.DataFrame()

    required = {'datetime', 'stockCodeCompany', 'nameCompany', 'closeValueStock'}
    if not required.issubset(history.columns):
        return pd.DataFrame()

    history = history[['datetime', 'stockCodeCompany', 'nameCompany', 'closeValueStock']].copy()
    history['datetime'] = pd.to_datetime(history['datetime'], errors='coerce').dt.normalize()
    history['closeValueStock'] = pd.to_numeric(history['closeValueStock'], errors='coerce')
    history = history.dropna(subset=['datetime', 'stockCodeCompany', 'closeValueStock'])
    history = history[history['closeValueStock'] > 0]
    return history.sort_values(['stockCodeCompany', 'datetime']).reset_index(drop=True)


def _build_swing_frame_from_history(history_df, asof_date, lookback_days=45):
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    cutoff = pd.Timestamp(asof_date).normalize()
    rows = []
    for stock_code, group in history_df.groupby('stockCodeCompany', sort=False):
        series = group[group['datetime'] <= cutoff].sort_values('datetime')['closeValueStock'].dropna()
        if lookback_days:
            series = series.tail(int(lookback_days))
        if len(series) < 20:
            continue

        metrics = _swing_metrics_from_close(series)
        if metrics is None:
            continue

        rows.append({
            'company': group['nameCompany'].dropna().iloc[0] if 'nameCompany' in group.columns and not group['nameCompany'].dropna().empty else stock_code,
            'stock_code': stock_code,
            **metrics,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame['swing_score'] = _compute_swing_trade_score(frame)
    frame['setup'] = np.select(
        [
            (frame['trend_20d_percent'] > 0) & (frame['trend_60d_percent'] > 0) & (frame['rsi_14'] < 65),
            (frame['trend_20d_percent'] < 0) & (frame['rsi_14'] < 35),
            (frame['trend_20d_percent'] < 0) & (frame['rsi_14'] > 60),
        ],
        ['Tendência forte', 'Reversão curta', 'Evitar'],
        default='Observação'
    )
    frame['entry_note'] = frame.apply(_swing_trade_note, axis=1)
    return frame.sort_values(['swing_score', 'ytd_return_percent'], ascending=[False, False]).reset_index(drop=True)


def _simulate_swing_trade(history_df, entry_date, hold_days, basket_size, conservative_mode, budget, lookback_days=45):
    frame = _build_swing_frame_from_history(history_df, entry_date, lookback_days=lookback_days)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), 'Sem dados históricos suficientes para simular o período.'

    plan, opinion = _build_swing_trade_allocation(frame, budget, basket_size, conservative_mode)
    if plan.empty:
        return frame, plan, 'A carteira simulada ficou vazia após a seleção.'

    rows = []
    for _, row in plan.iterrows():
        full_company_history = history_df[history_df['stockCodeCompany'] == row['stock_code']].sort_values('datetime').reset_index(drop=True)
        eligible_history = full_company_history[full_company_history['datetime'] <= pd.Timestamp(entry_date).normalize()]
        if eligible_history.empty:
            continue

        entry_pos = int(eligible_history.index[-1])
        exit_pos = min(entry_pos + int(hold_days), len(full_company_history) - 1)

        if exit_pos <= entry_pos:
            continue

        entry_row = full_company_history.iloc[entry_pos]
        exit_row = full_company_history.iloc[exit_pos]
        entry_close = float(entry_row['closeValueStock'])
        exit_close = float(exit_row['closeValueStock'])
        if entry_close <= 0 or exit_close <= 0:
            continue
        return_percent = ((exit_close / entry_close) - 1) * 100 if entry_close else 0.0
        invested = float(row['target_amount'])
        profit = invested * return_percent / 100

        rows.append({
            'Empresa': row['company'],
            'Ticker': row['stock_code'],
            'Entrada': pd.Timestamp(entry_row['datetime']).date(),
            'Saída': pd.Timestamp(exit_row['datetime']).date(),
            'Preço Entrada': entry_close,
            'Preço Saída': exit_close,
            'Retorno (%)': return_percent,
            'Resultado': 'Lucro' if profit > 0 else 'Prejuízo' if profit < 0 else 'Neutro',
            'Capital (R$)': invested,
            'Lucro/Prejuízo (R$)': profit,
            'Setup': row['setup'],
            'Observação': row['entry_note'],
        })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return frame, detail, 'Não foi possível montar a simulação com as datas selecionadas.'

    total_invested = float(detail['Capital (R$)'].sum())
    total_profit = float(detail['Lucro/Prejuízo (R$)'].sum())
    portfolio_return = (total_profit / total_invested * 100) if total_invested else 0.0
    hit_rate = float((detail['Retorno (%)'] > 0).mean() * 100)

    summary = pd.DataFrame([
        {
            'Entrada simulada': pd.Timestamp(entry_date).date(),
            'Saída após (pregões)': int(hold_days),
            'Ações simuladas': len(detail),
            'Capital total (R$)': total_invested,
            'Lucro/Prejuízo (R$)': total_profit,
            'Retorno da carteira (%)': portfolio_return,
            'Resultado': 'Lucro' if total_profit > 0 else 'Prejuízo' if total_profit < 0 else 'Neutro',
            'Taxa de acerto (%)': hit_rate,
        }
    ])
    return summary, detail.sort_values('Lucro/Prejuízo (R$)', ascending=False).reset_index(drop=True), opinion


@st.cache_data(show_spinner=False)
def _load_swing_trade_candidates(asof_date: str):
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame()

    universe = _swing_trade_universe()
    end_date = pd.Timestamp(asof_date).normalize()
    start_date = pd.Timestamp(year=end_date.year, month=1, day=1)
    rows = []
    skipped = []

    for ticker in universe:
        symbol = f'{ticker}.SA'
        try:
            history = yf.Ticker(symbol).history(start=start_date, end=end_date + pd.Timedelta(days=1), auto_adjust=True)
        except Exception:
            continue

        if history is None or history.empty or 'Close' not in history.columns:
            skipped.append(ticker)
            continue

        close = history['Close'].dropna()
        if len(close) < 20:
            skipped.append(ticker)
            continue

        returns = close.pct_change().dropna()
        last_close = float(close.iloc[-1])
        first_close = float(close.iloc[0])
        sma20 = float(close.tail(min(20, len(close))).mean())
        sma50 = float(close.tail(min(50, len(close))).mean())
        high_20 = float(close.tail(min(20, len(close))).max())
        low_20 = float(close.tail(min(20, len(close))).min())
        high_ytd = float(close.max())
        low_ytd = float(close.min())
        rsi = float(_compute_rsi_series(close).iloc[-1]) if len(close) >= 14 else np.nan
        volatility_20 = float(returns.tail(min(20, len(returns))).std() * np.sqrt(252) * 100) if len(returns) >= 5 else np.nan

        ret_5d = ((last_close / float(close.iloc[-6])) - 1) * 100 if len(close) >= 6 else np.nan
        ret_20d = ((last_close / float(close.iloc[-21])) - 1) * 100 if len(close) >= 21 else np.nan
        ret_60d = ((last_close / float(close.iloc[-61])) - 1) * 100 if len(close) >= 61 else np.nan
        ytd_return = ((last_close / first_close) - 1) * 100
        trend_20d = ((last_close / sma20) - 1) * 100 if sma20 else np.nan
        trend_60d = ((last_close / sma50) - 1) * 100 if sma50 else np.nan
        distance_sma20 = ((last_close / sma20) - 1) * 100 if sma20 else np.nan
        distance_sma50 = ((last_close / sma50) - 1) * 100 if sma50 else np.nan
        ytd_drawdown = ((last_close / float(close.cummax().iloc[-1])) - 1) * 100 if len(close) else np.nan
        range_position = ((last_close - low_20) / ((high_20 - low_20) + 1e-9)) * 100

        rows.append({
            'company': ticker,
            'stock_code': ticker,
            'last_close': last_close,
            'first_close_ytd': first_close,
            'ytd_return_percent': ytd_return,
            'ret_5d_percent': ret_5d,
            'ret_20d_percent': ret_20d,
            'ret_60d_percent': ret_60d,
            'trend_20d_percent': trend_20d,
            'trend_60d_percent': trend_60d,
            'distance_sma20_percent': distance_sma20,
            'distance_sma50_percent': distance_sma50,
            'volatility_20d_percent': volatility_20,
            'rsi_14': rsi,
            'ytd_drawdown_percent': ytd_drawdown,
            'range_position_20d': range_position,
            'high_ytd': high_ytd,
            'low_ytd': low_ytd,
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, skipped

    frame['swing_score'] = _compute_swing_trade_score(frame)
    frame['setup'] = np.select(
        [
            (frame['trend_20d_percent'] > 0) & (frame['trend_60d_percent'] > 0) & (frame['rsi_14'] < 65),
            (frame['trend_20d_percent'] < 0) & (frame['rsi_14'] < 35),
            (frame['trend_20d_percent'] < 0) & (frame['rsi_14'] > 60),
        ],
        ['Tendência forte', 'Reversão curta', 'Evitar'],
        default='Observação'
    )
    frame['entry_note'] = frame.apply(_swing_trade_note, axis=1)
    return frame.sort_values(['swing_score', 'ytd_return_percent'], ascending=[False, False]).reset_index(drop=True), skipped


def _build_swing_trade_allocation(frame, budget, basket_size, conservative_mode):
    if frame.empty:
        return frame.copy(), 'Sem dados de swing trade para o universo selecionado.'

    working = frame.copy()
    if conservative_mode:
        protected_mask = (
            (working['trend_20d_percent'] < 0)
            & (working['rsi_14'] > 58)
            & (working['volatility_20d_percent'] > working['volatility_20d_percent'].median())
        )
        filtered = working.loc[~protected_mask].copy()
        if filtered.empty:
            filtered = working.copy()
            opinion = 'Filtro conservador foi agressivo demais; mantive o universo completo para não zerar a seleção.'
        else:
            opinion = 'Filtro conservador aplicado: cortei parte dos nomes que estavam caindo sem sinal claro de recuperação.'
    else:
        filtered = working.copy()
        opinion = 'Modo agressivo liberado: priorizei potencial de swing trade mesmo com mais volatilidade.'

    plan = filtered.sort_values(['swing_score', 'ytd_return_percent', 'rsi_14'], ascending=[False, False, True]).head(basket_size).copy()
    if plan.empty:
        return plan, 'Sem candidatos após a seleção final.'

    raw_weights = plan['swing_score'].clip(lower=0)
    if raw_weights.sum() <= 0:
        plan['target_weight'] = 100 / len(plan)
    else:
        plan['target_weight'] = (raw_weights / raw_weights.sum()) * 100
    plan['target_amount'] = plan['target_weight'] / 100 * float(budget)

    avg_score = plan['swing_score'].mean()
    avg_return = plan['ytd_return_percent'].mean()
    avg_rsi = plan['rsi_14'].mean()
    negative_count = int((plan['trend_20d_percent'] < 0).sum())

    if negative_count >= max(1, len(plan) // 2):
        opinion += ' Ainda há muitos nomes em queda; eu começaria pequeno e esperaria confirmação.'
    elif avg_score >= 35 and avg_return > 0 and avg_rsi < 65:
        opinion += ' O conjunto ficou bom para swing trade curto; dá para trabalhar com entrada parcelada.'
    elif avg_score >= 20:
        opinion += ' A carteira está razoável, mas exige stop curto e revisão frequente.'
    else:
        opinion += ' O sinal ainda está fraco para operar agressivo.'

    plan['entry_note'] = plan.apply(_swing_trade_note, axis=1)
    return plan, opinion


@st.cache_data(show_spinner=False)
def _load_fundamentals_bundle(ticker_symbol: str):
    try:
        import yfinance as yf
    except Exception:
        return {}

    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info or {}
        balance_sheet = tk.balance_sheet if hasattr(tk, 'balance_sheet') else pd.DataFrame()
        financials = tk.financials if hasattr(tk, 'financials') else pd.DataFrame()
        cashflow = tk.cashflow if hasattr(tk, 'cashflow') else pd.DataFrame()
    except Exception:
        return {}

    revenue_series = None
    earnings_series = None
    if financials is not None and not financials.empty:
        revenue_row = None
        earnings_row = None
        for row_name in financials.index:
            row_lower = str(row_name).lower()
            if revenue_row is None and 'revenue' in row_lower:
                revenue_row = row_name
            if earnings_row is None and ('net income' in row_lower or 'netincome' in row_lower):
                earnings_row = row_name
        if revenue_row is not None:
            revenue_series = financials.loc[revenue_row].dropna()
        if earnings_row is not None:
            earnings_series = financials.loc[earnings_row].dropna()

    def bs_value(candidates):
        return _pick_row(balance_sheet, candidates)

    def fin_value(candidates):
        return _pick_row(financials, candidates)

    revenue = fin_value(['Total Revenue', 'Revenue', 'Operating Revenue'])
    net_income = fin_value(['Net Income', 'Net Income Common Stockholders', 'Net Income Continuous Operations'])
    ebitda = fin_value(['EBITDA'])
    ebit = fin_value(['EBIT', 'Operating Income'])
    gross_profit = fin_value(['Gross Profit'])

    total_assets = bs_value(['Total Assets'])
    total_equity = bs_value(['Stockholders Equity', 'Stockholder Equity', 'Total Stockholder Equity'])
    current_assets = bs_value(['Current Assets'])
    current_liabilities = bs_value(['Current Liabilities'])
    long_debt = bs_value(['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'])
    current_debt = bs_value(['Short Long Term Debt', 'Current Debt'])
    cash = bs_value(['Cash And Cash Equivalents', 'Cash And Short Term Investments', 'Cash'])

    total_debt = None
    if long_debt is not None or current_debt is not None:
        total_debt = float(long_debt or 0) + float(current_debt or 0)

    net_debt = None
    if total_debt is not None and cash is not None:
        net_debt = total_debt - cash

    result = dict(info)
    result['__revenue_series'] = revenue_series
    result['__earnings_series'] = earnings_series
    result['__total_debt'] = total_debt
    result['__net_debt'] = net_debt
    result['__total_equity'] = total_equity
    result['__current_assets'] = current_assets
    result['__current_liabilities'] = current_liabilities
    result['__total_assets'] = total_assets
    result['__gross_profit'] = gross_profit
    result['__ebitda'] = ebitda
    result['__ebit'] = ebit
    result['__revenue'] = revenue
    result['__net_income'] = net_income
    result['__revenue_growth'] = _cagr_from_series(revenue_series)
    result['__earnings_growth'] = _cagr_from_series(earnings_series)
    result['__gross_margin'] = (gross_profit / revenue) if gross_profit and revenue else None
    result['__ebitda_margin'] = (ebitda / revenue) if ebitda and revenue else None
    result['__ebit_margin'] = (ebit / revenue) if ebit and revenue else None
    result['__net_margin'] = (net_income / revenue) if net_income and revenue else None
    result['__debt_to_equity'] = (total_debt / total_equity) if total_debt and total_equity else None
    result['__debt_to_assets'] = (total_debt / total_assets) if total_debt and total_assets else None
    result['__current_ratio'] = (current_assets / current_liabilities) if current_assets and current_liabilities else None
    result['__net_debt_to_ebitda'] = (net_debt / ebitda) if net_debt is not None and ebitda else None
    result['__pl_to_assets'] = (total_equity / total_assets) if total_equity and total_assets else None
    result['__passivos_to_assets'] = (1 - (total_equity / total_assets)) if total_equity and total_assets else None
    result['__asset_turnover'] = (revenue / total_assets) if revenue and total_assets else None
    result['__price_to_assets'] = (info.get('marketCap') / total_assets) if info.get('marketCap') and total_assets else None
    result['__price_to_working_capital'] = (info.get('marketCap') / (current_assets - current_liabilities)) if info.get('marketCap') and current_assets and current_liabilities and (current_assets - current_liabilities) > 0 else None
    return result


st.set_page_config(
    page_title="Recomendações Amanhã",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Recomendações para Amanhã")
st.write("Predições com análise histórica para ajudar sua decisão")

# Carrega predições
pred_file = Path('predictions_tomorrow.json')
if not pred_file.exists():
    st.error("❌ Arquivo predictions_tomorrow.json não encontrado. Execute predict_tomorrow.py primeiro.")
    st.stop()

with open(pred_file) as f:
    predictions = json.load(f)

if not predictions:
    st.error("❌ Nenhuma predição disponível.")
    st.stop()

df_pred = pd.DataFrame(predictions)

analysis_date = None
if not df_pred.empty and 'analysis_base_date' in df_pred.columns:
    analysis_date = str(df_pred['analysis_base_date'].dropna().iloc[0])
pred_date = None
if not df_pred.empty and 'date' in df_pred.columns:
    pred_date = str(df_pred['date'].dropna().iloc[0])

# Score simples de risco (0-100, maior = mais arriscado)
df_pred['risk_score'] = (
    (df_pred['vol_annualized'].clip(0, 120) / 120) * 40 +
    (df_pred['max_drawdown_percent'].abs().clip(0, 80) / 80) * 35 +
    ((df_pred['prediction_status'] != 'ok').astype(int)) * 25
).clip(0, 100)

df_pred['buy_potential_score'] = _compute_buy_potential_score(df_pred)

# Carrega histórico de backtest para comparação
backtest_file = Path('backtest_results.json')
if backtest_file.exists():
    with open(backtest_file) as f:
        backtest_data = json.load(f)
    df_backtest = pd.DataFrame(backtest_data)
    df_backtest['date'] = pd.to_datetime(df_backtest['date'])
else:
    df_backtest = pd.DataFrame()

# ========== DISCLAIMER ==========
st.error("""
🚨 **AVISO: Estas são PREDIÇÕES EDUCACIONAIS**
- Baseadas em padrões históricos, mas o futuro pode ser diferente
- Sempre consulte um profissional antes de investir
- Risco real de perda de dinheiro
""")

# ========== FILTROS ==========
st.sidebar.header("🔍 Filtros")

recommendation_filter = st.sidebar.multiselect(
    "Tipo de Recomendação",
    ["COMPRA", "VENDA", "NEUTRO", "ANALISAR"],
    default=["COMPRA", "VENDA", "ANALISAR"]
)

page_section = st.sidebar.radio(
    "Área da página",
    [
        "Ofertas por prazo",
        "Resumo rápido",
        "Radar swing trade",
        "Página completa",
    ],
    index=0,
    help="Carrega só o bloco escolhido para deixar a navegação mais leve."
)

st.sidebar.caption("A seção de fundamentals usa apenas dados reais via yfinance; quando algo não existir, aparece como n/d.")

df_filtered = df_pred[df_pred['recommendation'].isin(recommendation_filter)]

if df_filtered.empty:
    st.warning("Nenhuma acao para os filtros selecionados.")
    st.stop()

# ========== RESUMO DE BASE TEMPORAL ========== 
if pred_date:
    if analysis_date:
        st.info(
            f"📅 Esta análise projeta o pregão de {_format_date_br(pred_date)} usando dados históricos até {_format_date_br(analysis_date)}. "
            f"Ou seja, a leitura do mercado está sempre um passo atrás do D+1."
        )
    else:
        st.info(f"📅 Esta análise projeta o pregão de {_format_date_br(pred_date)} com base nos dados históricos mais recentes disponíveis.")

if page_section == "Ofertas por prazo":
    st.header("🟢 Candidatos de Compra por Prazo")
    st.caption(
        "Use isto como triagem, não como ordem automática. Curto prazo olha D+1 e momentum; "
        "médio prazo mistura tendência, risco e consistência; longo prazo prioriza menor risco e validação."
    )

    horizon_candidates = _build_horizon_candidates(df_filtered)
    score_columns = {
        "Curto prazo": "Score Curto",
        "Médio prazo": "Score Médio",
        "Longo prazo": "Score Longo",
    }
    horizon_notes = {
        "Curto prazo": "Para operações rápidas: favorece probabilidade de alta, variação esperada, RSI saudável e risco controlado.",
        "Médio prazo": "Para algumas semanas/meses: dá mais peso para confiança, acurácia, tendência recente e volatilidade menor.",
        "Longo prazo": "Para montar posição aos poucos: exige risco menor, consistência histórica e tendência mais estável. Fundamentos ainda devem ser conferidos fora do score.",
    }

    for title, table in horizon_candidates.items():
        st.subheader(title)
        st.caption(horizon_notes[title])
        score_col = score_columns[title]
        st.dataframe(
            table.style.format({
                'Prob. Alta (%)': '{:.1f}',
                'Variação D+1 (%)': '{:+.2f}',
                'Tendência 20d (%)': '{:+.2f}',
                'RSI': '{:.1f}',
                'Risco': '{:.1f}',
                score_col: '{:.1f}',
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "Minha regra prática aqui: curto prazo precisa de sinal técnico forte; médio prazo precisa de risco aceitável; "
        "longo prazo precisa de fundamentos e diversificação além do modelo."
    )
    st.stop()

# ========== RESUMO GERAL ==========
st.header("📊 Resumo das Recomendações")

col1, col2, col3, col4 = st.columns(4)

compra_count = len(df_filtered[df_filtered['recommendation'] == 'COMPRA'])
venda_count = len(df_filtered[df_filtered['recommendation'] == 'VENDA'])
neutro_count = len(df_filtered[df_filtered['recommendation'] == 'NEUTRO'])
analisar_count = len(df_filtered[df_filtered['recommendation'] == 'ANALISAR'])

with col1:
    st.metric("🟢 Recomendações de COMPRA", compra_count)

with col2:
    st.metric("🔴 Recomendações de VENDA", venda_count)

with col3:
    st.metric("🟡 NEUTRO", neutro_count)

with col4:
    st.metric("🟠 ANALISAR", analisar_count)

# ========== SUGESTÕES DE COMPRA ==========
st.subheader("🟢 Sugestões com Potencial de Compra")

suggestions = df_pred.copy()
# Preferir nomes com histórico mínimo (trend_20days/rsi_14 disponíveis).
desired = 6
has_20d = suggestions['trend_20days'].notna() & (suggestions['trend_20days'] > -99) & suggestions['rsi_14'].notna()
primary = suggestions[has_20d].sort_values('buy_potential_score', ascending=False).head(desired)
if len(primary) < desired:
    filler = suggestions[~has_20d].sort_values('buy_potential_score', ascending=False).head(desired - len(primary))
    # mark filler rows
    filler = filler.copy()
    filler['__dados_incompletos'] = True
    primary = pd.concat([primary, filler], ignore_index=True)
else:
    primary = primary.copy()

# Guardar flag de dados incompletos antes de renomear
dados_incompletos_mask = primary.get('__dados_incompletos', False)
if isinstance(dados_incompletos_mask, bool):
    dados_incompletos_mask = [False] * len(primary)

buy_candidates = primary[[
    'company', 'stock_code', 'recommendation', 'probability_up', 'risk_score', 'trend_20days',
    'rsi_14', 'variation_percent', 'buy_potential_score'
]].copy()

buy_candidates.columns = [
    'Empresa', 'Ticker', 'Sinal', 'Prob. Alta (%)', 'Risco', 'Tendência 20d (%)',
    'RSI', 'Variação (%)', 'Score Compra'
]

st.dataframe(
    buy_candidates.style.format({
        'Prob. Alta (%)': '{:.1f}',
        'Risco': '{:.1f}',
        'Tendência 20d (%)': '{:+.2f}',
        'RSI': '{:.1f}',
        'Variação (%)': '{:+.2f}',
        'Score Compra': '{:.1f}',
    }),
    use_container_width=True,
    hide_index=True,
)

# Mostrar aviso se houver dados incompletos
if any(dados_incompletos_mask):
    incompletos_idx = [i for i, x in enumerate(dados_incompletos_mask) if x]
    incompletos = buy_candidates.iloc[incompletos_idx]['Ticker'].tolist()
    st.warning(f"Alguns papéis exibidos têm dados YTD incompletos (20d): {', '.join(incompletos)}")

st.caption("O score combina probabilidade de alta, confiança, risco e tendência recente para priorizar oportunidades. Papéis com dados incompletos aparecem com aviso.")

if page_section == "Resumo rápido":
    st.stop()

if page_section not in ["Radar swing trade", "Página completa"]:
    st.stop()

# ========== RADAR SWING TRADE YTD ========== 
st.header("💼 Radar Swing Trade do Ano")
st.caption(
    "Esta seção puxa apenas dados do início do ano até hoje e monta uma carteira curta, "
    "pensada para swing trade e para aportes menores. O radar trabalha com 40 ações líquidas."
)

swing_col1, swing_col2, swing_col3 = st.columns(3)
with swing_col1:
    swing_budget = st.number_input(
        "Quanto você quer investir neste lote (R$)",
        min_value=100.0,
        value=3000.0,
        step=100.0,
        format="%.2f"
    )
with swing_col2:
    basket_size = st.selectbox("Quantidade de ações na carteira", [5, 10, 15, 20], index=3)
with swing_col3:
    conservative_mode = st.checkbox("Modo protetivo", value=True)

today_key = pd.Timestamp.today().strftime('%Y-%m-%d')
swing_candidates, skipped_tickers = _load_swing_trade_candidates(today_key)

if isinstance(swing_candidates, tuple):
    # backward compatibility guard
    swing_candidates = swing_candidates[0]

if swing_candidates.empty:
    st.warning("Não consegui montar o radar YTD agora. Verifique se o yfinance está disponível e tente novamente.")
    if skipped_tickers:
        st.info(f"Tickers sem dados ou insuficientes: {', '.join(skipped_tickers[:20])}")
else:
    selected_count = min(40, len(swing_candidates))
    swing_candidates = swing_candidates.head(selected_count).copy()
    if skipped_tickers:
        st.info(f"Alguns tickers foram ignorados por falta de dados: {', '.join(skipped_tickers[:20])}")

    swing_plan, swing_opinion = _build_swing_trade_allocation(
        swing_candidates,
        swing_budget,
        basket_size,
        conservative_mode
    )

    top_summary = swing_candidates.head(40)
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("Ações no radar", len(top_summary))
    with summary_col2:
        st.metric("Score médio", f"{top_summary['swing_score'].mean():.1f}")
    with summary_col3:
        st.metric("Retorno YTD médio", f"{top_summary['ytd_return_percent'].mean():+.2f}%")
    with summary_col4:
        st.metric("RSI médio", f"{top_summary['rsi_14'].mean():.1f}")

    if conservative_mode and (swing_candidates['trend_20d_percent'] < 0).mean() > 0.5:
        st.warning("Mais da metade do radar ainda está em queda de curto prazo; por isso o modo protetivo reduz peso nesses nomes.")

    st.info(swing_opinion)

    radar_view = top_summary[[
        'company', 'stock_code', 'setup', 'swing_score', 'ytd_return_percent',
        'ret_5d_percent', 'ret_20d_percent', 'trend_20d_percent', 'trend_60d_percent',
        'volatility_20d_percent', 'rsi_14', 'entry_note'
    ]].copy()
    radar_view = radar_view.rename(columns={
        'company': 'Empresa',
        'stock_code': 'Ticker',
        'setup': 'Setup',
        'swing_score': 'Score Swing',
        'ytd_return_percent': 'Retorno YTD (%)',
        'ret_5d_percent': 'Retorno 5d (%)',
        'ret_20d_percent': 'Retorno 20d (%)',
        'trend_20d_percent': 'Dist. SMA20 (%)',
        'trend_60d_percent': 'Dist. SMA50 (%)',
        'volatility_20d_percent': 'Vol. 20d (%)',
        'rsi_14': 'RSI',
        'entry_note': 'Observação'
    })

    st.dataframe(
        radar_view.style.format({
            'Score Swing': '{:.1f}',
            'Retorno YTD (%)': '{:+.2f}',
            'Retorno 5d (%)': '{:+.2f}',
            'Retorno 20d (%)': '{:+.2f}',
            'Dist. SMA20 (%)': '{:+.2f}',
            'Dist. SMA50 (%)': '{:+.2f}',
            'Vol. 20d (%)': '{:.2f}',
            'RSI': '{:.1f}',
        }),
        use_container_width=True,
        hide_index=True,
    )

    fig_radar = px.bar(
        top_summary,
        x='company',
        y='swing_score',
        color='ytd_return_percent',
        labels={'company': 'Ação', 'swing_score': 'Score Swing', 'ytd_return_percent': 'Retorno YTD (%)'},
        title='Ranking de swing trade com base no ano corrente'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    if not swing_plan.empty:
        st.subheader("🎯 Distribuição sugerida do capital")
        allocation_view = swing_plan[[
            'company', 'stock_code', 'setup', 'swing_score', 'ytd_return_percent',
            'rsi_14', 'trend_20d_percent', 'target_weight', 'target_amount', 'entry_note'
        ]].copy()
        allocation_view = allocation_view.rename(columns={
            'company': 'Empresa',
            'stock_code': 'Ticker',
            'setup': 'Setup',
            'swing_score': 'Score Swing',
            'ytd_return_percent': 'Retorno YTD (%)',
            'rsi_14': 'RSI',
            'trend_20d_percent': 'Dist. SMA20 (%)',
            'target_weight': 'Peso Sugerido (%)',
            'target_amount': 'Valor Sugerido (R$)',
            'entry_note': 'Observação'
        })

        st.dataframe(
            allocation_view.style.format({
                'Score Swing': '{:.1f}',
                'Retorno YTD (%)': '{:+.2f}',
                'RSI': '{:.1f}',
                'Dist. SMA20 (%)': '{:+.2f}',
                'Peso Sugerido (%)': '{:.2f}',
                'Valor Sugerido (R$)': 'R$ {:,.2f}',
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig_allocation = px.bar(
            swing_plan,
            x='company',
            y='target_amount',
            color='swing_score',
            labels={'company': 'Ação', 'target_amount': 'Valor sugerido (R$)', 'swing_score': 'Score Swing'},
            title='Distribuição do aporte entre os melhores nomes do radar'
        )
        st.plotly_chart(fig_allocation, use_container_width=True)

        st.caption(
            'O radar acima usa somente informações de janeiro até hoje. A carteira favorece tendência, '
            'força de curto prazo e protege contra compras cedo demais em ativos que ainda estão caindo.'
        )

if page_section == "Radar swing trade":
    st.stop()

# ========== COMPARAÇÃO AUTOMÁTICA ========== 
st.subheader("🧪 Comparação automática com dados antigos")
st.caption(
    "Aqui o radar é refeito em datas passadas usando o histórico do ano corrente. "
    "Isso mostra quanto teria rendido se a carteira tivesse sido montada naquela data e vendida depois de alguns pregões."
)

history_df = _load_year_to_date_history()
if history_df.empty:
    st.warning("Não consegui carregar o histórico consolidado para simular os cenários antigos.")
else:
    all_dates = pd.DatetimeIndex(sorted(history_df['datetime'].unique()))
    latest_date = all_dates.max()

    auto_col1, auto_col2, auto_col3 = st.columns(3)
    with auto_col1:
        selected_lookback = st.selectbox("Entrada simulada", [5, 10, 21], index=1, help="5 = 1 semana, 10 = 2 semanas, 21 = 1 mês útil")
    with auto_col2:
        selected_hold = st.selectbox("Venda após (pregões)", [5, 10, 21], index=0)
    with auto_col3:
        selected_budget = st.number_input(
            "Capital para a simulação (R$)",
            min_value=100.0,
            value=float(swing_budget),
            step=100.0,
            format="%.2f"
        )
    selected_metric_window = st.slider(
        "Janela usada para escolher as ações (pregões)",
        min_value=20,
        max_value=60,
        value=30,
        step=5,
        help="Evita que a análise diária comece lá em janeiro quando a ideia é comparar movimentos recentes."
    )

    simulated_entry = _nearest_trading_date(
        all_dates,
        latest_date - pd.offsets.BDay(int(selected_lookback) + int(selected_hold) + 1)
    )
    summary_df, detail_df, simulation_note = _simulate_swing_trade(
        history_df,
        simulated_entry,
        int(selected_hold),
        int(basket_size),
        conservative_mode,
        float(selected_budget),
        lookback_days=int(selected_metric_window),
    )

    if not summary_df.empty:
        st.info(simulation_note)
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Retorno da carteira", f"{summary_df['Retorno da carteira (%)'].iloc[0]:+.2f}%")
        with metric_cols[1]:
            st.metric("Lucro/Prejuízo", f"R$ {summary_df['Lucro/Prejuízo (R$)'].iloc[0]:,.2f}")
        with metric_cols[2]:
            st.metric("Taxa de acerto", f"{summary_df['Taxa de acerto (%)'].iloc[0]:.1f}%")
        with metric_cols[3]:
            st.metric("Ações simuladas", int(summary_df['Ações simuladas'].iloc[0]))

        summary_display = summary_df.copy()
        if 'Entrada simulada' in summary_display.columns:
            summary_display['Entrada simulada'] = summary_display['Entrada simulada'].map(_format_date_br)
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        st.subheader("📋 Resultado por ação na simulação")
        detail_display = detail_df.copy()
        for date_col in ['Entrada', 'Saída']:
            if date_col in detail_display.columns:
                detail_display[date_col] = detail_display[date_col].map(_format_date_br)
        st.dataframe(
            detail_display.style.format({
                'Preço Entrada': 'R$ {:.2f}',
                'Preço Saída': 'R$ {:.2f}',
                'Retorno (%)': '{:+.2f}',
                'Capital (R$)': 'R$ {:,.2f}',
                'Lucro/Prejuízo (R$)': 'R$ {:,.2f}',
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📅 Cenários automáticos de comparação")
    scenario_specs = [
        (5, 5, 'Entrada de 1 semana / saída em 1 semana'),
        (10, 5, 'Entrada de 2 semanas / saída em 1 semana'),
        (21, 5, 'Entrada de 1 mês / saída em 1 semana'),
        (21, 10, 'Entrada de 1 mês / saída em 2 semanas'),
    ]
    scenario_rows = []
    for lookback_days, hold_days, label in scenario_specs:
        entry_date = _nearest_trading_date(
            all_dates,
            latest_date - pd.offsets.BDay(int(lookback_days) + int(hold_days) + 1)
        )
        scenario_summary, _, _ = _simulate_swing_trade(
            history_df,
            entry_date,
            int(hold_days),
            int(basket_size),
            conservative_mode,
            float(selected_budget),
            lookback_days=int(selected_metric_window),
        )
        if scenario_summary.empty:
            continue
        scenario_rows.append({
            'Cenário': label,
            'Entrada': scenario_summary['Entrada simulada'].iloc[0],
            'Saída após (pregões)': int(scenario_summary['Saída após (pregões)'].iloc[0]),
            'Retorno da carteira (%)': scenario_summary['Retorno da carteira (%)'].iloc[0],
            'Lucro/Prejuízo (R$)': scenario_summary['Lucro/Prejuízo (R$)'].iloc[0],
            'Taxa de acerto (%)': scenario_summary['Taxa de acerto (%)'].iloc[0],
        })

    if scenario_rows:
        scenarios_df = pd.DataFrame(scenario_rows)
        if 'Entrada' in scenarios_df.columns:
            scenarios_df['Entrada'] = scenarios_df['Entrada'].map(_format_date_br)
        st.dataframe(
            scenarios_df.style.format({
                'Retorno da carteira (%)': '{:+.2f}',
                'Lucro/Prejuízo (R$)': 'R$ {:,.2f}',
                'Taxa de acerto (%)': '{:.1f}',
            }),
            use_container_width=True,
            hide_index=True,
        )

# ========== PAINEL DA AÇÃO SELECIONADA ==========
st.subheader("🎯 Painel da Ação Selecionada")
selected_action_options = ["Todas"] + sorted(df_filtered['company'].dropna().unique().tolist())
selected_action = st.selectbox("Escolha uma ação para detalhar", selected_action_options)

focused_df = df_filtered.copy()
if selected_action != "Todas":
    focused_df = focused_df[focused_df['company'] == selected_action]

if not focused_df.empty:
    if 'buy_potential_score' not in focused_df.columns:
        focused_df = focused_df.copy()
        focused_df['buy_potential_score'] = _compute_buy_potential_score(focused_df)
    focus_row = focused_df.sort_values(['risk_score', 'buy_potential_score'], ascending=[True, False]).iloc[0]
    decision_map = {
        'COMPRA': 'Pode considerar entrada gradual se a liquidez e o horizonte fizerem sentido.',
        'VENDA': 'Evite comprar agora; faz mais sentido reduzir exposição ou aguardar reversão.',
        'NEUTRO': 'Sem gatilho forte. Acompanhe e espere confirmação extra.',
        'ANALISAR': 'Sinal ainda inconclusivo. Vale monitorar mais um pregão ou cruzar com fundamentos.',
    }
    decision_text = decision_map.get(focus_row['recommendation'], 'Sem recomendação clara.')

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Empresa", f"{focus_row['company']} ({focus_row['stock_code']})")
    with p2:
        st.metric("Sinal", focus_row['recommendation'])
    with p3:
        st.metric("Preço Previsto", _format_money(focus_row['predicted_close']))
    with p4:
        st.metric("Var. Esperada", _format_percent_points(focus_row['variation_percent'], signed=True))

    st.write(f"**Data da previsão:** {_format_date_br(focus_row['date'])}  ")
    if pd.notna(focus_row.get('analysis_base_date')):
        st.write(f"**Base da análise:** {_format_date_br(focus_row.get('analysis_base_date'))}  ")
    st.write(f"**O que fazer:** {decision_text}")

    detail_cols = st.columns(3)
    with detail_cols[0]:
        st.metric("Prob. Alta", f"{focus_row['probability_up']:.1f}%")
        st.metric("Prob. Queda", f"{focus_row['probability_down']:.1f}%")
    with detail_cols[1]:
        st.metric("Risco", f"{focus_row['risk_score']:.1f}/100")
        st.metric("Confiança", f"{focus_row['confidence_score']*100:.0f}%")
    with detail_cols[2]:
        st.metric("RSI", f"{focus_row['rsi_14']:.1f}")
        st.metric("Tendência 20d", f"{focus_row['trend_20days']:+.2f}%")

    if pd.notna(focus_row.get('analysis_base_date')):
        st.caption("A leitura acima compara uma previsão de D+1 com o histórico disponível até a data-base mostrada.")
else:
    st.info("Nenhuma ação disponível para detalhar com os filtros atuais.")

# Ranking rapido para decisao
st.subheader("🏆 Ranking Rápido para Triagem")
rank_df = df_filtered[[
    'company', 'stock_code', 'recommendation', 'variation_percent', 'accuracy_mean',
    'rsi_14', 'trend_regime', 'risk_score'
]].copy()
if 'probability_up' in df_filtered.columns:
    rank_df['probability_up'] = df_filtered['probability_up']
    rank_df['probability_down'] = df_filtered['probability_down']
rank_df = rank_df.sort_values(['recommendation', 'risk_score', 'accuracy_mean'], ascending=[True, True, False])
st.dataframe(rank_df, use_container_width=True, hide_index=True)

# ========== GRÁFICO DE RECOMENDAÇÕES ==========
st.subheader("Distribuição de Recomendações")

rec_counts = df_filtered['recommendation'].value_counts()
fig_rec = px.bar(
    x=rec_counts.index,
    y=rec_counts.values,
    labels={'x': 'Recomendação', 'y': 'Quantidade'},
    title='Quantidade de Ações por Tipo de Recomendação',
    color=rec_counts.index,
    color_discrete_map={
        'COMPRA': '#00aa00',
        'VENDA': '#ff0000',
        'NEUTRO': '#ffaa00',
        'ANALISAR': '#0099ff'
    }
)
st.plotly_chart(fig_rec, use_container_width=True)

# ========== RECOMENDAÇÕES DETALHADAS ==========
st.header("💡 Análise Detalhada por Ação")

# Ordenar por variação
detail_options = ["Melhor candidata"] + sorted(df_filtered['company'].dropna().unique().tolist())
detail_choice = st.selectbox("Carregar detalhe de", detail_options, help="Cotação e fundamentals são pesados; carregue uma ação por vez.")
if detail_choice == "Melhor candidata":
    df_sorted = df_filtered.sort_values(['buy_potential_score', 'risk_score'], ascending=[False, True]).head(1)
else:
    df_sorted = df_filtered[df_filtered['company'] == detail_choice].sort_values('variation_percent', ascending=False).head(1)

for idx, row in df_sorted.iterrows():
    
    # Cor baseada em recomendação
    if row['recommendation'] == 'COMPRA':
        color = '🟢'
    elif row['recommendation'] == 'VENDA':
        color = '🔴'
    elif row['recommendation'] == 'NEUTRO':
        color = '🟡'
    else:
        color = '🟠'
    
    with st.expander(
        f"{color} **{row['company']}** - {row['recommendation']} | "
        f"Variação: {row['variation_percent']:+.2f}% | "
        f"Confiança: {row['confidence']}"
    ):
        
        # Dados principais em 4 colunas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Preço Atual", f"R$ {row['last_close']:.2f}")
        
        with col2:
            st.metric("Predição Amanhã", f"R$ {row['predicted_close']:.2f}")
        
        with col3:
            st.metric("Variação Predita", f"{row['variation_percent']:+.2f}%")
        
        with col4:
            st.metric("Confiança", row['confidence'])

        if 'probability_up' in row and pd.notna(row.get('probability_up')):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prob. Alta", f"{row['probability_up']:.1f}%")
            with col2:
                st.metric("Prob. Queda", f"{row['probability_down']:.1f}%")
            with col3:
                st.metric("Acurácia Direcional", f"{row.get('directional_accuracy_global', 0):.1f}%")

        if 'prediction_status' in row and row['prediction_status'] != 'ok':
            st.warning("Predição fora do intervalo esperado. Exibindo valor de fechamento anterior.")
        
        # Análise histórica
        st.subheader("📈 Análise Histórica")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Média Histórica", f"R$ {row['mean_historical']:.2f}")
        
        with col2:
            st.metric("Desvio Padrão", f"R$ {row['std_historical']:.2f}")
        
        with col3:
            st.metric("Volatilidade (20d)", f"{row['volatility']:.2f}%")
        
        with col4:
            st.metric("Tendência (20d)", f"{row['trend_20days']:+.2f}%")

        # Indicadores tecnicos
        st.subheader("📐 Indicadores Técnicos")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RSI (14)", f"{row['rsi_14']:.1f}")
        with col2:
            st.metric("ATR (14)", f"R$ {row['atr_14']:.2f}")
        with col3:
            st.metric("SMA20 / SMA50", f"{_format_sma(row, 'sma_20', 20)} / {_format_sma(row, 'sma_50', 50)}")
        with col4:
            st.metric("SMA200", _format_sma(row, 'sma_200', 200))
        historical_days = row.get('historical_days', 0)
        historical_days = 0 if pd.isna(historical_days) else int(historical_days)
        st.caption(
            "SMA só aparece quando há pregões suficientes para a janela. "
            f"Histórico local deste papel: {historical_days} pregões."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Volatilidade Anualizada", f"{row['vol_annualized']:.2f}%")
        with col2:
            st.metric("Max Drawdown", f"{row['max_drawdown_percent']:.2f}%")
        with col3:
            st.metric("Regime de Tendência", row['trend_regime'])

        # ========== COTAÇÃO EM TEMPO REAL ==========
        st.subheader("📡 Cotação em Tempo Real")
        try:
            import yfinance as yf
        except Exception:
            st.info("Instale 'yfinance' para ver cotações em tempo real (pip install yfinance)")
        else:
            ticker_symbol = row['stock_code']
            if not ticker_symbol.endswith('.SA'):
                ticker_symbol = ticker_symbol + '.SA'
            try:
                tk = yf.Ticker(ticker_symbol)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    intraday = tk.history(period='1d', interval='5m')
                    if intraday is None or intraday.empty:
                        intraday = tk.history(period='5d', interval='1d')

                if intraday is not None and not intraday.empty:
                    last_price = intraday['Close'].iloc[-1]
                    last_dt = intraday.index[-1]
                    col_a, col_b = st.columns([1,3])
                    with col_a:
                        st.metric("Último Preço", f"R$ {last_price:.2f}")
                        st.write(f"Atualizado: {pd.to_datetime(last_dt).strftime('%Y-%m-%d %H:%M')}")
                    with col_b:
                        time_column = intraday.index.name or intraday.reset_index().columns[0]
                        plot_frame = intraday.reset_index().rename(columns={time_column: 'Timestamp'})
                        fig_rt = px.line(
                            plot_frame,
                            x='Timestamp',
                            y='Close',
                            title=f"Cotação recente {row['stock_code']}",
                            labels={'Timestamp': 'Data/Hora', 'Close': 'Preço (R$)'}
                        )
                        st.plotly_chart(fig_rt, width='stretch')
                else:
                    st.warning("Sem cotação recente disponível para esta ação neste momento.")
            except Exception as e:
                st.warning(f"Cotação recente indisponível para {row['stock_code']}: {e}")

            info = _load_fundamentals_bundle(ticker_symbol)
            if info:
                valuation_groups = [
                    ("Indicadores de Valuation", [
                        ("D.Y", "dividendYield", "percent", "📈"),
                        ("P/L", "trailingPE", "number", "💹"),
                        ("PEG Ratio", "pegRatio", "number", "📊"),
                        ("P/VP", "priceToBook", "number", "📘"),
                        ("EV/EBITDA", "enterpriseToEbitda", "number", "🧮"),
                        ("EV/EBIT", "enterpriseToEbit", "number", "🧾"),
                        ("P/EBITDA", "enterpriseToEbitda", "number", "📉"),
                        ("P/EBIT", "enterpriseToEbit", "number", "📈"),
                        ("VPA", "bookValue", "number", "🏷️"),
                        ("P/Ativo", "__price_to_assets", "number", "🏦"),
                        ("LPA", "trailingEps", "number", "🪙"),
                        ("P/SR", "priceToSalesTrailing12Months", "number", "🔎"),
                        ("P/Cap. Giro", "__price_to_working_capital", "number", "♻️"),
                        ("P/Ativo Circ. Liq.", "__price_to_working_capital", "number", "🧭"),
                    ]),
                    ("Indicadores de Endividamento", [
                        ("Dív. líquida/PL", "__debt_to_equity", "number", "🧱"),
                        ("Dív. líquida/EBITDA", "__net_debt_to_ebitda", "number", "🪫"),
                        ("Dív. líquida/EBIT", "enterpriseToEbit", "number", "⚖️"),
                        ("PL/Ativos", "__pl_to_assets", "number", "🏢"),
                        ("Passivos/Ativos", "__passivos_to_assets", "number", "📚"),
                        ("Liq. corrente", "__current_ratio", "number", "💧"),
                    ]),
                    ("Indicadores de Eficiência", [
                        ("M. Bruta", "__gross_margin", "percent", "🧾"),
                        ("M. EBITDA", "__ebitda_margin", "percent", "⚙️"),
                        ("M. EBIT", "__ebit_margin", "percent", "🔧"),
                        ("M. Líquida", "__net_margin", "percent", "🫧"),
                    ]),
                    ("Indicadores de Rentabilidade", [
                        ("ROE", "returnOnEquity", "percent", "🌱"),
                        ("ROA", "returnOnAssets", "percent", "🪴"),
                        ("ROIC", "returnOnInvestedCapital", "percent", "🎯"),
                        ("Giro ativos", "__asset_turnover", "number", "🔁"),
                    ]),
                    ("Indicadores de Crescimento", [
                        ("CAGR Receitas 5 anos", "__revenue_growth", "percent", "📦"),
                        ("CAGR Lucros 5 anos", "__earnings_growth", "percent", "📉"),
                    ]),
                ]

                def render_metric_cards(title, items):
                    st.markdown(f"### {title}")
                    rows = [items[i:i + 4] for i in range(0, len(items), 4)]
                    for row_items in rows:
                        cols = st.columns(len(row_items))
                        for col, (label, key, kind, icon) in zip(cols, row_items):
                            value = info.get(key)
                            col.metric(f"{icon} {label}", _format_number(value, kind))

                st.subheader("📊 Indicadores Fundamentais")
                st.caption("Os indicadores abaixo combinam `yfinance.info` com dados dos demonstrativos financeiros quando disponíveis.")
                for group_title, group_items in valuation_groups:
                    render_metric_cards(group_title, group_items)

                st.markdown("### Notas de Cobertura")
                note_cols = st.columns(3)
                with note_cols[0]:
                    st.info("P/Ativo, P/Cap. Giro e Giro de ativos agora são derivados do market cap, balanço e receita quando existem demonstrativos.")
                with note_cols[1]:
                    st.info("Margens, dívida líquida e crescimento usam demonstrativos financeiros anuais do ativo, quando a base tem dados.")
                with note_cols[2]:
                    st.info("Para comparar com concorrentes, o passo seguinte é adicionar uma fonte externa de fundamentals da bolsa.")
            else:
                st.info("Informações fundamentais não disponíveis.")

        st.progress(min(max(float(row['risk_score']) / 100.0, 0.0), 1.0))
        st.caption(f"Risk Score: {row['risk_score']:.1f}/100 (maior = mais risco)")

        # Validacao D-1 -> D
        st.subheader("✅ Validação Histórica Recente (D-1 → D)")
        if pd.notna(row.get('last_backtest_accuracy')):
            st.info(
                f"Último teste em {row.get('last_backtest_date')}: "
                f"acurácia {row.get('last_backtest_accuracy'):.2f}% | "
                f"erro R$ {row.get('last_backtest_error'):.2f}"
            )
        else:
            st.warning("Sem validação recente para esta ação no backtest.")
        
        # Performance histórica do modelo
        st.subheader("🎯 Performance Histórica do Modelo")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Acurácia Média", f"{row['accuracy_mean']:.2f}%")
        
        with col2:
            st.metric("Acurácia Mínima", f"{row['accuracy_min']:.2f}%")
        
        with col3:
            st.metric("Acurácia Máxima", f"{row['accuracy_max']:.2f}%")
        
        with col4:
            st.metric("Score de Confiança", f"{row['confidence_score']*100:.0f}%")
        
        # Análise técnica
        st.subheader("🔬 Análise Técnica")
        
        analysis_text = f"""
        **Setor:** {row['sector']}  
        **Segmento:** {row['segment']}  
        
        **O que significam esses dados:**
        """
        
        if row['variation_percent'] > 2:
            analysis_text += "\n- 📈 **Alta variação positiva** predita - Forte movimento de alta esperado"
        elif row['variation_percent'] > 0.5:
            analysis_text += "\n- 📊 **Variação positiva moderada** - Leve tendência de alta"
        elif row['variation_percent'] < -2:
            analysis_text += "\n- 📉 **Alta variação negativa** - Forte movimento de queda esperado"
        elif row['variation_percent'] < -0.5:
            analysis_text += "\n- 📊 **Variação negativa moderada** - Leve tendência de queda"
        else:
            analysis_text += "\n- ➡️ **Estável** - Pouco movimento esperado"
        
        if row['volatility'] > 3.5:
            analysis_text += "\n- ⚠️ **Alta volatilidade** - Ação oscila bastante, maior risco"
        else:
            analysis_text += "\n- ✅ **Volatilidade normal** - Movimento previsível"

        if row['rsi_14'] < 30:
            analysis_text += "\n- 🟢 **RSI baixo (sobrevendida)** - pode indicar ponto de recuperação"
        elif row['rsi_14'] > 70:
            analysis_text += "\n- 🔴 **RSI alto (sobrecomprada)** - risco de correção"
        else:
            analysis_text += "\n- 🟡 **RSI neutro**"
        
        if row['trend_20days'] > 0:
            analysis_text += f"\n- 📈 **Tendência positiva** nos últimos 20 dias (+{row['trend_20days']:.2f}%)"
        else:
            analysis_text += f"\n- 📉 **Tendência negativa** nos últimos 20 dias ({row['trend_20days']:.2f}%)"
        
        if row['accuracy_mean'] > 99.5:
            analysis_text += "\n- 🎯 **Modelo muito confiável** para esta ação - histórico excelente"
        elif row['accuracy_mean'] > 99:
            analysis_text += "\n- ✅ **Modelo confiável** para esta ação"
        else:
            analysis_text += "\n- ⚠️ **Falta dados históricos** - validação menor"
        
        st.markdown(analysis_text)
        
        # Recomendação detalhada
        st.subheader("💡 Recomendação Detalhada")
        
        if row['recommendation'] == 'COMPRA':
            st.success(f"""
            ✅ **RECOMENDAÇÃO: COMPRA**
            
            Indicadores positivos:
            - Predição de variação positiva: +{row['variation_percent']:.2f}%
            - Acurácia histórica forte: {row['accuracy_mean']:.2f}%
            - Confiança: {row['confidence']}
            
            ⚠️ Lembre-se: Sempre invista com moderação e consulte profissionais!
            """)
        
        elif row['recommendation'] == 'VENDA':
            st.error(f"""
            🔴 **RECOMENDAÇÃO: VENDA**
            
            Indicadores negativos:
            - Predição de variação negativa: {row['variation_percent']:.2f}%
            - Acurácia histórica forte: {row['accuracy_mean']:.2f}%
            - Confiança: {row['confidence']}
            
            ⚠️ Se você possui essa ação, considere realocação!
            """)
        
        elif row['recommendation'] == 'NEUTRO':
            st.info(f"""
            🟡 **RECOMENDAÇÃO: NEUTRO**
            
            Sem movimento significativo esperado:
            - Variação próxima a zero: {row['variation_percent']:+.2f}%
            - Ação estável e previsível
            - Bom para conservadores
            """)
        
        else:
            st.warning(f"""
            🟠 **RECOMENDAÇÃO: ANALISAR**
            
            Necessário análise mais profunda:
            - Resultado inconclusivo
            - Considere análise fundamentalista
            - Consulte especialista
            """)

# ========== COMPARAÇÃO HISTÓRICA ==========
if len(df_backtest) > 0:
    st.header("📊 Comparação com Histórico de Backtests")
    
    st.subheader("Acurácia das Ações no Histórico")
    
    # Resumo por empresa
    company_accuracy = df_backtest.groupby('company')['accuracy'].agg([
        ('Media', 'mean'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Total_Testes', 'count')
    ]).reset_index()
    
    company_accuracy = company_accuracy.sort_values('Media', ascending=False)
    
    fig_accuracy = px.bar(
        company_accuracy,
        x='company',
        y='Media',
        labels={'company': 'Ação', 'Media': 'Acurácia Média (%)'},
        title='Acurácia Histórica do Modelo por Ação',
        color='Media',
        color_continuous_scale='RdYlGn'
    )
    
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Tabela de comparação
    st.dataframe(
        company_accuracy.rename(columns={
            'company': 'Ação',
            'Media': 'Acurácia Média (%)',
            'Min': 'Mínima (%)',
            'Max': 'Máxima (%)',
            'Total_Testes': 'Total de Testes'
        }),
        use_container_width=True,
        hide_index=True
    )

# ========== INSTRUÇÕES E AVISOS ==========
st.header("📚 Como Usar Esta Página")

st.info("""
### Passo a Passo:

1. **Veja as Recomendações**: Use os filtros para ver COMPRA, VENDA, etc.

2. **Leia a Análise Histórica**: Cada ação mostra seu histórico de 20 dias

3. **Verifique a Confiança**: Quanto maior a acurácia histórica, melhor

4. **Compare**: Veja se faz sentido com outras análises

5. **Decida com Segurança**: Invista apenas o que pode perder

### O que Significa Cada Campo:

- **Variação Predita**: Quanto o preço deve subir/descer amanhã
- **Volatilidade**: Quanto a ação oscila (maior = mais arriscado)
- **Tendência 20d**: Direção dos últimos 20 dias
- **Acurácia Histórica**: Sucesso do modelo em predições passadas
- **Confiança**: Score de quão certo o modelo está

### ⚠️ Avisos Importantes:

- ❌ Estas são **apenas predições educacionais**
- ❌ O passado não garante o futuro
- ❌ Sempre diversifique seus investimentos
- ❌ Nunca invista dinheiro que você não pode perder
- ❌ Consulte sempre um profissional

**Boa sorte! 🍀**
""")
