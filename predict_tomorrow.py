"""
Predicao para o proximo pregao sem vazamento de dados.

Esta versao treina um modelo para prever o retorno do proximo dia usando
somente informacoes conhecidas ate o fechamento do dia atual.
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import Config
from src.data_ingestion import DataIngestor

TARGET = Config.TARGET
CATEGORICAL_FEATURES = ["stockCodeCompany", "sectorCompany", "segmentCompany"]
NUMERIC_FEATURES = [
    "dayWeekTime",
    "monthTime",
    "next_dayWeekTime",
    "next_monthTime",
    "close",
    "log_close",
    "ret_1d",
    "ret_2d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_mean_5d",
    "ret_mean_20d",
    "vol_5d",
    "vol_20d",
    "rsi_14",
    "atr_pct_14",
    "range_pct",
    "open_gap_pct",
    "close_position",
    "sma_5_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "sma_20_50_ratio",
    "drawdown_20d",
    "volume_rel_20d",
    "valueCoin",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def next_business_day(date_value):
    next_day = date_value + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    return next_day


def repair_invalid_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_cols = ["openValueStock", "highValueStock", "lowValueStock", TARGET]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df[TARGET]
    invalid_open = df["openValueStock"].isna() | (df["openValueStock"] <= 0)
    invalid_high = df["highValueStock"].isna() | (df["highValueStock"] <= 0)
    invalid_low = df["lowValueStock"].isna() | (df["lowValueStock"] <= 0)

    df.loc[invalid_open, "openValueStock"] = close[invalid_open]
    df.loc[invalid_high, "highValueStock"] = df.loc[
        invalid_high, ["openValueStock", TARGET]
    ].max(axis=1)
    df.loc[invalid_low, "lowValueStock"] = df.loc[
        invalid_low, ["openValueStock", TARGET]
    ].min(axis=1)
    df["highValueStock"] = df[["highValueStock", "openValueStock", TARGET]].max(axis=1)
    df["lowValueStock"] = df[["lowValueStock", "openValueStock", TARGET]].min(axis=1)
    return df


def compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr_series(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def compute_max_drawdown(close_series: pd.Series) -> float:
    running_max = close_series.cummax()
    drawdown = (close_series - running_max) / (running_max + 1e-9)
    return float(drawdown.min() * 100)


def rolling_mean_or_none(close_series: pd.Series, window: int):
    if len(close_series) < window:
        return None
    return float(close_series.tail(window).mean())


def add_company_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("datetime").copy()
    close = group[TARGET].astype(float)
    high = group["highValueStock"].astype(float)
    low = group["lowValueStock"].astype(float)
    open_price = group["openValueStock"].astype(float)
    volume = group["quantityStock"].astype(float)
    returns = close.pct_change()

    group["close"] = close
    group["log_close"] = np.log(close.clip(lower=1e-9))
    group["ret_1d"] = returns
    group["ret_2d"] = close.pct_change(2)
    group["ret_3d"] = close.pct_change(3)
    group["ret_5d"] = close.pct_change(5)
    group["ret_10d"] = close.pct_change(10)
    group["ret_20d"] = close.pct_change(20)
    group["ret_mean_5d"] = returns.rolling(5, min_periods=5).mean()
    group["ret_mean_20d"] = returns.rolling(20, min_periods=20).mean()
    group["vol_5d"] = returns.rolling(5, min_periods=5).std()
    group["vol_20d"] = returns.rolling(20, min_periods=20).std()
    group["rsi_14"] = compute_rsi_series(close, 14)
    group["atr_pct_14"] = compute_atr_series(high, low, close, 14) / (close + 1e-9)
    group["range_pct"] = (high - low) / (close + 1e-9)
    group["open_gap_pct"] = (open_price / (close.shift(1) + 1e-9)) - 1
    group["close_position"] = (close - low) / ((high - low) + 1e-9)

    sma_5 = close.rolling(5, min_periods=5).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    sma_50 = close.rolling(50, min_periods=50).mean()
    group["sma_5_ratio"] = (close / (sma_5 + 1e-9)) - 1
    group["sma_20_ratio"] = (close / (sma_20 + 1e-9)) - 1
    group["sma_50_ratio"] = (close / (sma_50 + 1e-9)) - 1
    group["sma_20_50_ratio"] = (sma_20 / (sma_50 + 1e-9)) - 1
    group["drawdown_20d"] = (close / (close.rolling(20, min_periods=20).max() + 1e-9)) - 1
    group["volume_rel_20d"] = (volume / (volume.rolling(20, min_periods=20).mean() + 1e-9)) - 1

    next_date = group["datetime"].shift(-1)
    group["next_dayWeekTime"] = next_date.dt.weekday
    group["next_monthTime"] = next_date.dt.month
    group["target_return_next_day"] = (close.shift(-1) / (close + 1e-9)) - 1
    group["target_close_next_day"] = close.shift(-1)
    return group


def build_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["stockCodeCompany", "datetime"]).reset_index(drop=True)
    df = repair_invalid_ohlc(df)
    frames = []
    for _, group in df.groupby("stockCodeCompany", sort=False):
        frames.append(add_company_features(group))
    return pd.concat(frames, ignore_index=True)


def make_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=350,
        max_leaf_nodes=24,
        l2_regularization=0.05,
        random_state=Config.RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def make_direction_model() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=80,
        max_leaf_nodes=10,
        l2_regularization=0.2,
        random_state=Config.RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def directional_backtest(dataset: pd.DataFrame, test_days: int = 80):
    valid_dates = sorted(dataset["datetime"].dropna().unique())
    if len(valid_dates) <= test_days + 80:
        test_dates = valid_dates[max(0, int(len(valid_dates) * 0.75)) :]
    else:
        test_dates = valid_dates[-test_days:]

    if not test_dates:
        return pd.DataFrame(), {}

    first_test_date = test_dates[0]
    train = dataset[dataset["datetime"] < first_test_date].copy()
    test = dataset[dataset["datetime"].isin(test_dates)].copy()
    if len(train) < 500 or test.empty:
        return pd.DataFrame(), {}

    train["target_up"] = (train["target_return_next_day"] > 0).astype(int)
    test["target_up"] = (test["target_return_next_day"] > 0).astype(int)

    direction_model = make_direction_model()
    direction_model.fit(train[FEATURES], train["target_up"])
    prob_up = direction_model.predict_proba(test[FEATURES])[:, 1]
    pred_return = ((prob_up - 0.5) * 2.0) * test["vol_20d"].fillna(test["vol_20d"].median()).to_numpy()
    real_return = test["target_return_next_day"].to_numpy()

    rows = []
    for row, pred, real, prob in zip(test.to_dict("records"), pred_return, real_return, prob_up):
        pred = float(pred)
        real = float(real)
        rows.append(
            {
                "date": str(pd.Timestamp(row["datetime"]).date()),
                "target_date": str(pd.Timestamp(row["datetime"]).date()),
                "company": row["stockCodeCompany"],
                "real_return_percent": real * 100,
                "predicted_return_percent": pred * 100,
                "probability_up": float(prob),
                "direction_hit": bool(np.sign(pred) == np.sign(real)) if abs(pred) > 1e-9 else False,
                "absolute_error_percent": abs(real - pred) * 100,
            }
        )

    if not rows:
        return pd.DataFrame(), {}

    df_bt = pd.DataFrame(rows)
    summary = {
        "tests": int(len(df_bt)),
        "directional_accuracy": float(df_bt["direction_hit"].mean() * 100),
        "mae_percent": float(df_bt["absolute_error_percent"].mean()),
        "buy_precision_55": float(
            (df_bt.loc[df_bt["probability_up"] >= 0.55, "real_return_percent"] > 0).mean() * 100
        )
        if (df_bt["probability_up"] >= 0.55).any()
        else 0.0,
        "sell_precision_45": float(
            (df_bt.loc[df_bt["probability_up"] <= 0.45, "real_return_percent"] < 0).mean() * 100
        )
        if (df_bt["probability_up"] <= 0.45).any()
        else 0.0,
        "buy_signals_55": int((df_bt["probability_up"] >= 0.55).sum()),
        "sell_signals_45": int((df_bt["probability_up"] <= 0.45).sum()),
        "mean_predicted_return_percent": float(df_bt["predicted_return_percent"].mean()),
        "mean_real_return_percent": float(df_bt["real_return_percent"].mean()),
    }
    return df_bt, summary


def recommendation_from_signal(probability_up, buy_precision, sell_precision, rsi_14, trend_regime):
    # Cortes conservadores calibrados no holdout temporal recente.
    # A faixa intermediaria vira ANALISAR para evitar "sinal" em ruido.
    if probability_up >= 0.57 and buy_precision >= 55:
        if rsi_14 > 72:
            return "ANALISAR"
        if trend_regime == "Baixa" and probability_up < 0.60:
            return "ANALISAR"
        return "COMPRA"

    if probability_up <= 0.40 and sell_precision >= 55:
        if rsi_14 < 28 and trend_regime != "Baixa":
            return "ANALISAR"
        return "VENDA"

    if 0.485 <= probability_up <= 0.515:
        return "NEUTRO"

    return "ANALISAR"


def _old_recommendation_from_signal(pred_return, direction_accuracy, rsi_14, trend_regime, vol_20d):
    pred_pct = pred_return * 100
    vol_pct = 0.0 if pd.isna(vol_20d) else float(vol_20d * 100)
    min_edge = max(0.35, min(1.0, vol_pct * 0.35))

    if direction_accuracy < 52:
        if abs(pred_pct) < min_edge * 1.5:
            return "NEUTRO"
        return "ANALISAR"

    if pred_pct >= min_edge:
        if rsi_14 > 72:
            return "ANALISAR"
        if trend_regime == "Baixa" and pred_pct < min_edge * 1.7:
            return "ANALISAR"
        return "COMPRA"

    if pred_pct <= -min_edge:
        if rsi_14 < 28 and trend_regime != "Baixa":
            return "ANALISAR"
        return "VENDA"

    return "NEUTRO"


def confidence_from_accuracy(direction_accuracy, mae_percent):
    if direction_accuracy >= 58 and mae_percent <= 2.0:
        return "Alta", 0.80
    if direction_accuracy >= 53:
        return "Media", 0.65
    return "Baixa", 0.45


def predict_tomorrow():
    print("[INFO] Gerando predicoes sem vazamento para o proximo pregao...")
    print("=" * 70)

    df_raw = DataIngestor().load_and_merge()
    featured = build_feature_frame(df_raw)
    train_data = featured.dropna(subset=FEATURES + ["target_return_next_day"]).copy()
    train_data = train_data[train_data["datetime"] >= pd.Timestamp("2018-01-01")].copy()
    train_data["target_up"] = (train_data["target_return_next_day"] > 0).astype(int)

    latest_idx = featured.groupby("stockCodeCompany")["datetime"].idxmax()
    latest_rows = featured.loc[latest_idx].copy()

    today = pd.Timestamp(featured["datetime"].max()).date()
    tomorrow = next_business_day(today)
    latest_rows["next_dayWeekTime"] = tomorrow.weekday()
    latest_rows["next_monthTime"] = tomorrow.month
    latest_rows = latest_rows.dropna(subset=FEATURES).copy()

    print(f"[DATA] Ultima data de dados: {today}")
    print(f"[DATA] Proximo pregao previsto: {tomorrow}")
    print(f"[DATA] Analise feita com dados ate: {today}")
    print(f"[TREINO] Linhas supervisionadas: {len(train_data)}")

    backtest_df, backtest_summary = directional_backtest(train_data, test_days=60)
    if not backtest_df.empty:
        backtest_df.to_json(
            "backtest_directional_results.json",
            orient="records",
            indent=2,
            force_ascii=False,
        )
        Path("backtest_directional_summary.json").write_text(
            json.dumps(backtest_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            "[BACKTEST] Acuracia direcional: "
            f"{backtest_summary['directional_accuracy']:.2f}% | "
            f"MAE: {backtest_summary['mae_percent']:.2f}%"
        )
    else:
        backtest_summary = {
            "directional_accuracy": 0.0,
            "mae_percent": 0.0,
            "tests": 0,
        }
        print("[BACKTEST] Dados insuficientes para backtest direcional.")

    direction_model = make_direction_model()
    direction_model.fit(train_data[FEATURES], train_data["target_up"])
    prob_up_values = direction_model.predict_proba(latest_rows[FEATURES])[:, 1]

    predictions = []
    direction_accuracy = float(backtest_summary.get("directional_accuracy", 0.0))
    mae_percent = float(backtest_summary.get("mae_percent", 0.0))
    buy_precision = float(backtest_summary.get("buy_precision_55", 0.0))
    sell_precision = float(backtest_summary.get("sell_precision_45", 0.0))
    confidence, confidence_score = confidence_from_accuracy(direction_accuracy, mae_percent)

    print("\n[PREDICAO] Sinais para o proximo pregao")
    print("=" * 70)

    for row, probability_up in zip(latest_rows.to_dict("records"), prob_up_values):
        stock_code = row["stockCodeCompany"]
        company_name = row["nameCompany"]
        close_series = featured[featured["stockCodeCompany"] == stock_code][TARGET].astype(float)
        last_close = float(row[TARGET])
        vol_20d_value = float(row["vol_20d"]) if pd.notna(row["vol_20d"]) else 0.0
        expected_return = float(np.clip((probability_up - 0.5) * 2.0 * vol_20d_value, -0.04, 0.04))
        predicted_close = float(last_close * (1.0 + expected_return))
        variation = float(expected_return * 100)

        historical_days = int(close_series.dropna().shape[0])
        sma_5 = rolling_mean_or_none(close_series, 5)
        sma_20 = rolling_mean_or_none(close_series, 20)
        sma_50 = rolling_mean_or_none(close_series, 50)
        sma_200 = rolling_mean_or_none(close_series, 200)
        returns = close_series.pct_change().dropna()
        vol_annualized = float(returns.std() * np.sqrt(252) * 100) if len(returns) > 2 else 0.0
        max_drawdown = compute_max_drawdown(close_series)
        trend_20days = float(row["ret_20d"] * 100) if pd.notna(row["ret_20d"]) else 0.0
        volatility = float(row["vol_20d"] * 100) if pd.notna(row["vol_20d"]) else 0.0
        rsi_14 = float(row["rsi_14"]) if pd.notna(row["rsi_14"]) else 50.0
        atr_14 = float(row["atr_pct_14"] * last_close) if pd.notna(row["atr_pct_14"]) else 0.0

        if None not in (sma_20, sma_50, sma_200) and sma_20 > sma_50 > sma_200:
            trend_regime = "Alta"
        elif None not in (sma_20, sma_50, sma_200) and sma_20 < sma_50 < sma_200:
            trend_regime = "Baixa"
        elif None in (sma_20, sma_50, sma_200):
            trend_regime = "Indefinido"
        else:
            trend_regime = "Lateral"

        recommendation = recommendation_from_signal(
            float(probability_up),
            buy_precision,
            sell_precision,
            rsi_14,
            trend_regime,
        )

        stock_backtest = (
            backtest_df[backtest_df["company"] == stock_code]
            if not backtest_df.empty
            else pd.DataFrame()
        )
        if not stock_backtest.empty:
            accuracy_mean = float(stock_backtest["direction_hit"].mean() * 100)
            accuracy_min = 0.0
            accuracy_max = 100.0
            last_bt = stock_backtest.iloc[-1]
            last_backtest_date = str(last_bt["date"])
            last_backtest_accuracy = 100.0 if bool(last_bt["direction_hit"]) else 0.0
            last_backtest_error = float(last_bt["absolute_error_percent"])
        else:
            accuracy_mean = direction_accuracy
            accuracy_min = 0.0
            accuracy_max = 100.0
            last_backtest_date = None
            last_backtest_accuracy = None
            last_backtest_error = None

        predictions.append(
            {
                "date": tomorrow.strftime("%Y-%m-%d"),
                "analysis_base_date": str(today),
                "model_version": "next_day_return_v2_no_leakage",
                "company": company_name,
                "stock_code": stock_code,
                "sector": row["sectorCompany"],
                "segment": row["segmentCompany"],
                "last_close": last_close,
                "predicted_close": predicted_close,
                "variation_percent": variation,
                "predicted_return_percent": variation,
                "probability_up": float(probability_up * 100),
                "probability_down": float((1.0 - probability_up) * 100),
                "rsi_14": rsi_14,
                "atr_14": atr_14,
                "sma_5": sma_5,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "sma_200": sma_200,
                "historical_days": historical_days,
                "sma_5_days": min(historical_days, 5),
                "sma_20_days": min(historical_days, 20),
                "sma_50_days": min(historical_days, 50),
                "sma_200_days": min(historical_days, 200),
                "vol_annualized": vol_annualized,
                "max_drawdown_percent": max_drawdown,
                "trend_regime": trend_regime,
                "mean_historical": float(close_series.mean()),
                "std_historical": float(close_series.std()),
                "trend_20days": trend_20days,
                "volatility": volatility,
                "accuracy_mean": accuracy_mean,
                "accuracy_min": accuracy_min,
                "accuracy_max": accuracy_max,
                "directional_accuracy_global": direction_accuracy,
                "mae_percent_global": mae_percent,
                "buy_precision_55": buy_precision,
                "sell_precision_45": sell_precision,
                "last_backtest_date": last_backtest_date,
                "last_backtest_accuracy": last_backtest_accuracy,
                "last_backtest_error": last_backtest_error,
                "confidence": confidence,
                "confidence_score": float(confidence_score),
                "prediction_status": "ok",
                "raw_predicted_close": predicted_close,
                "fallback_return_percent": 0.0,
                "recommendation": recommendation,
                "model_r2": 0.0,
            }
        )

        print(
            f"{stock_code:6s} {company_name:18s} "
            f"p_up={probability_up:5.1%} ret={variation:+6.2f}% rec={recommendation:8s} "
            f"rsi={rsi_14:5.1f} trend20={trend_20days:+6.2f}%"
        )

    output_file = Path("predictions_tomorrow.json")
    output_file.write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    df_pred = pd.DataFrame(predictions)
    print(f"\n[OK] Predicoes salvas em: {output_file}")
    print(f"[TOTAL] {len(predictions)} acoes analisadas")
    print("\n[RESUMO]")
    for rec in ["COMPRA", "VENDA", "NEUTRO", "ANALISAR"]:
        count = int((df_pred["recommendation"] == rec).sum())
        if count > 0:
            print(f"  {rec}: {count} acoes")


if __name__ == "__main__":
    predict_tomorrow()
