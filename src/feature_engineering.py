"""
Pipeline de Feature Engineering
Combina indicadores técnicos, sentimento, dados macro e features temporais
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

try:
    from .technical_indicators import TechnicalIndicators
    from .sentiment_analyzer import SentimentAnalyzer
    from .macro_data import MacroDataCollector
except ImportError:
    from technical_indicators import TechnicalIndicators
    from sentiment_analyzer import SentimentAnalyzer
    from macro_data import MacroDataCollector


class FeatureEngineer:
    """Pipeline completo de feature engineering para swing trade."""
    
    def __init__(self):
        self.technical_calc = None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.macro_collector = MacroDataCollector()
        self.macro_data = None
    
    def create_features_for_stock(self, df: pd.DataFrame, ticker: str, 
                                   include_sentiment: bool = True,
                                   include_macro: bool = True) -> pd.DataFrame:
        """
        Cria todas as features para um DataFrame de ações.
        
        Args:
            df: DataFrame com colunas OHLCV
            ticker: Ticker da ação
            include_sentiment: Incluir features de sentimento
            include_macro: Incluir features macroeconômicas
        
        Returns:
            DataFrame com todas as features
        """
        # 1. Indicadores técnicos
        print(f"   Calculando indicadores técnicos para {ticker}...")
        self.technical_calc = TechnicalIndicators(df)
        df_features = self.technical_calc.calculate_all()
        
        # 2. Features de preço relativo
        df_features = self._add_relative_price_features(df_features)
        
        # 3. Features de momentum avançado
        df_features = self._add_advanced_momentum_features(df_features)
        
        # 4. Features de volatilidade avançada
        df_features = self._add_advanced_volatility_features(df_features)
        
        # 5. Features temporais
        df_features = self._add_temporal_features(df_features)
        
        # 6. Features de sentimento (opcional)
        if include_sentiment:
            print(f"   Coletando sentimento para {ticker}...")
            df_features = self._add_sentiment_features(df_features, ticker)
        
        # 7. Features macroeconômicas (opcional)
        if include_macro:
            print(f"   Coletando dados macro...")
            if self.macro_data is None:
                self.macro_data = self.macro_collector.get_all_macro_data(730)
            df_features = self._add_macro_features(df_features)
        
        # 8. Features de alvo (targets)
        df_features = self._add_target_features(df_features)
        
        return df_features
    
    def _add_relative_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de preço relativo."""
        close = df['Close']
        
        # Retornos em múltiplos períodos
        df['Return_1d'] = close.pct_change(1)
        df['Return_3d'] = close.pct_change(3)
        df['Return_5d'] = close.pct_change(5)
        df['Return_10d'] = close.pct_change(10)
        df['Return_20d'] = close.pct_change(20)
        
        # Retornos cumulativos
        df['CumReturn_5d'] = close.pct_change(5) + 1
        df['CumReturn_10d'] = close.pct_change(10) + 1
        df['CumReturn_20d'] = close.pct_change(20) + 1
        
        # Força relativa vs médias
        if 'SMA_20' in df.columns:
            df['Close_SMA20_Ratio'] = close / df['SMA_20']
        if 'SMA_50' in df.columns:
            df['Close_SMA50_Ratio'] = close / df['SMA_50']
        if 'SMA_200' in df.columns:
            df['Close_SMA200_Ratio'] = close / df['SMA_200']
        
        # Posição no range (últimos 20 dias)
        df['High_Low_Range_20d'] = df['High'].rolling(20).max() - df['Low'].rolling(20).min()
        df['Close_Position_20d'] = (close - df['Low'].rolling(20).min()) / df['High_Low_Range_20d']
        
        return df
    
    def _add_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de momentum avançadas."""
        close = df['Close']
        
        # Convergência/divergência
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Divergence'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Hist_Signal'] = np.where(df['MACD_Hist'] > 0, 1, -1)
        
        # RSI divergência
        if 'RSI_14' in df.columns:
            df['RSI_Overbought'] = (df['RSI_14'] > 70).astype(int)
            df['RSI_Oversold'] = (df['RSI_14'] < 30).astype(int)
            df['RSI_Mid'] = ((df['RSI_14'] >= 40) & (df['RSI_14'] <= 60)).astype(int)
        
        # Momentum de volume
        if 'Volume_Ratio' in df.columns:
            df['Volume_Spike'] = (df['Volume_Ratio'] > 2).astype(int)
            df['Volume_Drop'] = (df['Volume_Ratio'] < 0.5).astype(int)
        
        return df
    
    def _add_advanced_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volatilidade avançadas."""
        close = df['Close']
        
        # ATR normalizado pelo preço
        if 'ATR' in df.columns:
            df['ATR_Normalized'] = df['ATR'] / close
        
        # Volatilidade em diferentes janelas
        returns = close.pct_change()
        df['Volatility_5d'] = returns.rolling(5).std()
        df['Volatility_10d'] = returns.rolling(10).std()
        df['Volatility_20d'] = returns.rolling(20).std()
        
        # Regime de volatilidade
        vol_20d = df['Volatility_20d']
        df['Volatility_High'] = (vol_20d > vol_20d.quantile(0.75)).astype(int)
        df['Volatility_Low'] = (vol_20d < vol_20d.quantile(0.25)).astype(int)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features temporais."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Efeitos sazonais
        df['Is_Monday'] = (df.index.dayofweek == 0).astype(int)
        df['Is_Friday'] = (df.index.dayofweek == 4).astype(int)
        df['Is_Month_End'] = (df.index.day > 25).astype(int)
        df['Is_Month_Start'] = (df.index.day <= 5).astype(int)
        
        # Dias desde eventos (simplificado)
        df['Days_Since_High_20d'] = (df['Close'].rolling(20).max().shift(1) < df['Close']).astype(int)
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Adiciona features de sentimento."""
        try:
            sentiment = self.sentiment_analyzer.get_comprehensive_sentiment(ticker)
            
            # Google Trends
            trend_score = sentiment['google_trends'].get('trend_score', 0)
            df['Google_Trends_Score'] = trend_score / 100  # Normalizar 0-1
            
            # Sentimento de notícias
            news_sentiment = sentiment['news_sentiment']
            df['News_Positive_Pct'] = news_sentiment.get('positive_pct', 0) / 100
            df['News_Negative_Pct'] = news_sentiment.get('negative_pct', 0) / 100
            df['News_Sentiment_Score'] = sentiment['overall_score']
            
            # Sinal de sentimento
            df['Sentiment_Signal'] = np.where(
                sentiment['overall_score'] > 0.2, 1,
                np.where(sentiment['overall_score'] < -0.2, -1, 0)
            )
            
        except Exception as e:
            print(f"   ⚠️ Erro ao adicionar sentimento: {e}")
            df['Google_Trends_Score'] = 0
            df['News_Positive_Pct'] = 0
            df['News_Negative_Pct'] = 0
            df['News_Sentiment_Score'] = 0
            df['Sentiment_Signal'] = 0
        
        return df
    
    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features macroeconômicas."""
        if self.macro_data is None or self.macro_data.empty:
            return df
        
        # Fazer merge por data
        df_index = df.index.normalize()
        
        for col in ['usd_brl', 'ibov_return', 'sp500_return', 'dollar_change']:
            if col in self.macro_data.columns:
                # Forward fill para alinhar com dados diários
                macro_series = self.macro_data[col].reindex(df_index, method='ffill')
                df[f'Macro_{col}'] = macro_series
        
        return df
    
    def _add_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features de alvo (targets) para treinamento.
        Alvos: Retorno em 5, 10, 20 dias
        """
        close = df['Close']
        
        # Targets de retorno futuro
        df['Target_Return_5d'] = close.shift(-5) / close - 1
        df['Target_Return_10d'] = close.shift(-10) / close - 1
        df['Target_Return_20d'] = close.shift(-20) / close - 1
        
        # Target de direção (classificação binária)
        df['Target_Direction_5d'] = (df['Target_Return_5d'] > 0).astype(int)
        df['Target_Direction_10d'] = (df['Target_Return_10d'] > 0).astype(int)
        df['Target_Direction_20d'] = (df['Target_Return_20d'] > 0).astype(int)
        
        # Target de drawdown esperado
        df['Target_Drawdown_20d'] = (
            (close.rolling(20).min().shift(-20) - close) / close
        )
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Retorna lista de colunas de features (excluindo targets e colunas originais).
        """
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Target_Return_5d', 'Target_Return_10d', 'Target_Return_20d',
            'Target_Direction_5d', 'Target_Direction_10d', 'Target_Direction_20d',
            'Target_Drawdown_20d'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols


def create_training_dataset(data_dir: str, tickers: List[str], 
                           include_sentiment: bool = False,
                           include_macro: bool = False) -> pd.DataFrame:
    """
    Cria dataset de treino combinando múltiplas ações.
    
    Args:
        data_dir: Diretório com dados CSV
        tickers: Lista de tickers para incluir
        include_sentiment: Incluir features de sentimento
        include_macro: Incluir features macro
    
    Returns:
        DataFrame combinado com todas as features
    """
    from pathlib import Path
    
    engineer = FeatureEngineer()
    all_data = []
    
    for ticker in tickers:
        csv_path = Path(data_dir) / f"{ticker}.csv"
        
        if not csv_path.exists():
            print(f"⚠️ CSV não encontrado: {csv_path}")
            continue
        
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Garantir colunas OHLCV
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠️ Colunas faltando em {ticker}")
                continue
            
            # Criar features
            df_features = engineer.create_features_for_stock(
                df, ticker, include_sentiment, include_macro
            )
            
            # Adicionar coluna de ticker
            df_features['Ticker'] = ticker
            
            all_data.append(df_features)
            print(f"✅ Features criadas para {ticker}: {df_features.shape}")
            
        except Exception as e:
            print(f"❌ Erro ao processar {ticker}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Combinar todos os DataFrames
    combined = pd.concat(all_data, ignore_index=False)
    
    # Remover linhas com NaN nos targets
    target_cols = ['Target_Return_5d', 'Target_Return_10d', 'Target_Return_20d']
    combined = combined.dropna(subset=target_cols)
    
    print(f"\n✅ Dataset combinado: {combined.shape}")
    print(f"   Features: {len([c for c in combined.columns if 'Target' not in c])}")
    print(f"   Amostras: {len(combined)}")
    
    return combined


if __name__ == "__main__":
    # Teste com dados fictícios
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=500, freq='D')
    close_prices = np.cumsum(np.random.randn(500) * 0.02) + 100
    
    test_df = pd.DataFrame({
        'Open': close_prices + np.random.randn(500) * 0.5,
        'High': close_prices + np.random.rand(500) * 2,
        'Low': close_prices - np.random.rand(500) * 2,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    engineer = FeatureEngineer()
    features = engineer.create_features_for_stock(test_df, 'TEST4', 
                                                 include_sentiment=False,
                                                 include_macro=False)
    
    print(f"\n=== Teste Feature Engineering ===")
    print(f"Shape: {features.shape}")
    print(f"Colunas: {features.columns.tolist()}")
    print(f"\nÚltima linha (primeiras 10 features):")
    print(features.iloc[-1][:10])
