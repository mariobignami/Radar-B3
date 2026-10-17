"""
Módulo de Indicadores Técnicos Avançados
Usa TA-Lib para calcular 50+ indicadores técnicos para análise de swing trade
"""
import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional


class TechnicalIndicators:
    """Calcula indicadores técnicos avançados para swing trade."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa com DataFrame de preços OHLCV.
        
        Args:
            df: DataFrame com colunas ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.df = df.copy()
        self.close = df['Close'].astype(float).values
        self.high = df['High'].astype(float).values
        self.low = df['Low'].astype(float).values
        self.open = df['Open'].astype(float).values
        self.volume = df['Volume'].astype(float).values
    
    def calculate_all(self) -> pd.DataFrame:
        """Calcula todos os indicadores e retorna DataFrame enriquecido."""
        result = self.df.copy()
        
        # Tendência
        result = self._add_trend_indicators(result)
        
        # Momentum
        result = self._add_momentum_indicators(result)
        
        # Volatilidade
        result = self._add_volatility_indicators(result)
        
        # Volume
        result = self._add_volume_indicators(result)
        
        # Padrões candlestick
        result = self._add_candlestick_patterns(result)
        
        return result
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de tendência."""
        # SMAs
        df['SMA_5'] = talib.SMA(self.close, timeperiod=5)
        df['SMA_10'] = talib.SMA(self.close, timeperiod=10)
        df['SMA_20'] = talib.SMA(self.close, timeperiod=20)
        df['SMA_50'] = talib.SMA(self.close, timeperiod=50)
        df['SMA_200'] = talib.SMA(self.close, timeperiod=200)
        
        # EMAs
        df['EMA_12'] = talib.EMA(self.close, timeperiod=12)
        df['EMA_26'] = talib.EMA(self.close, timeperiod=26)
        
        # MACD
        macd, macdsignal, macdhist = talib.MACD(self.close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        df['MACD_Hist'] = macdhist
        
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(self.high, self.low, self.close, timeperiod=14)
        df['ADX_Pos'] = talib.PLUS_DI(self.high, self.low, self.close, timeperiod=14)
        df['ADX_Neg'] = talib.MINUS_DI(self.high, self.low, self.close, timeperiod=14)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(self.high, self.low, acceleration=0.02, maximum=0.2)
        
        # Preços relativos vs médias
        df['Close_vs_SMA20'] = (self.close - df['SMA_20']) / df['SMA_20'] * 100
        df['Close_vs_SMA50'] = (self.close - df['SMA_50']) / df['SMA_50'] * 100
        df['Close_vs_SMA200'] = (self.close - df['SMA_200']) / df['SMA_200'] * 100
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de momentum."""
        # RSI
        df['RSI_14'] = talib.RSI(self.close, timeperiod=14)
        df['RSI_21'] = talib.RSI(self.close, timeperiod=21)
        
        # Estocástico
        slowk, slowd = talib.STOCH(self.high, self.low, self.close, 
                                    fastk_period=14, slowk_period=3, slowd_period=3)
        df['Stoch_K'] = slowk
        df['Stoch_D'] = slowd
        
        # Williams %R
        df['Williams_R'] = talib.WILLR(self.high, self.low, self.close, timeperiod=14)
        
        # CCI (Commodity Channel Index)
        df['CCI'] = talib.CCI(self.high, self.low, self.close, timeperiod=20)
        
        # ROC (Rate of Change)
        df['ROC_5'] = talib.ROC(self.close, timeperiod=5)
        df['ROC_10'] = talib.ROC(self.close, timeperiod=10)
        df['ROC_20'] = talib.ROC(self.close, timeperiod=20)
        
        # Momentum
        df['Momentum'] = talib.MOM(self.close, timeperiod=10)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de volatilidade."""
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        df['BB_Width'] = (upper - lower) / middle * 100
        df['BB_Position'] = (self.close - lower) / (upper - lower) * 100
        
        # ATR (Average True Range)
        df['ATR'] = talib.ATR(self.high, self.low, self.close, timeperiod=14)
        df['ATR_Pct'] = df['ATR'] / self.close * 100
        
        # Keltner Channels
        keltner_middle = talib.EMA(self.close, timeperiod=20)
        keltner_upper = keltner_middle + 2 * talib.ATR(self.high, self.low, self.close, timeperiod=14)
        keltner_lower = keltner_middle - 2 * talib.ATR(self.high, self.low, self.close, timeperiod=14)
        df['KC_Upper'] = keltner_upper
        df['KC_Middle'] = keltner_middle
        df['KC_Lower'] = keltner_lower
        
        # Volatilidade histórica
        returns = pd.Series(self.close).pct_change()
        df['Volatility_20d'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores de volume."""
        # OBV (On Balance Volume)
        df['OBV'] = talib.OBV(self.close, self.volume)
        
        # Volume médio
        df['Volume_SMA_20'] = talib.SMA(self.volume, timeperiod=20)
        df['Volume_Ratio'] = self.volume / df['Volume_SMA_20']
        
        # VWAP (Volume Weighted Average Price)
        typical_price = (self.high + self.low + self.close) / 3
        vwap = (typical_price * self.volume).cumsum() / self.volume.cumsum()
        df['VWAP'] = vwap
        df['Close_vs_VWAP'] = (self.close - vwap) / vwap * 100
        
        # MFI (Money Flow Index)
        df['MFI'] = talib.MFI(self.high, self.low, self.close, self.volume, timeperiod=14)
        
        return df
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona padrões candlestick básicos."""
        # Padrões de reversão
        df['CDL_DOJI'] = talib.CDLDOJI(self.open, self.high, self.low, self.close)
        df['CDL_HAMMER'] = talib.CDLHAMMER(self.open, self.high, self.low, self.close)
        df['CDL_SHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(self.open, self.high, self.low, self.close)
        df['CDL_ENGULFING'] = talib.CDLENGULFING(self.open, self.high, self.low, self.close)
        df['CDL_MORNINGSTAR'] = talib.CDLMORNINGSTAR(self.open, self.high, self.low, self.close)
        df['CDL_EVENINGSTAR'] = talib.CDLEVENINGSTAR(self.open, self.high, self.low, self.close)
        
        return df
    
    def get_latest_signals(self) -> Dict:
        """
        Retorna sinais baseados nos indicadores mais recentes.
        Útil para geração de features para o modelo de ML.
        """
        if len(self.df) < 200:
            return {}

        latest = self.df.iloc[-1]
        signals = {}

        # Tendência
        signals['trend'] = self._classify_trend(latest)
        signals['trend_strength'] = latest.get('ADX', 0) if pd.notna(latest.get('ADX')) else 0

        # Momentum
        signals['rsi_signal'] = self._classify_rsi(latest.get('RSI_14'))
        signals['macd_signal'] = self._classify_macd(latest.get('MACD'), latest.get('MACD_Signal'))

        # Volatilidade
        signals['volatility_regime'] = self._classify_volatility(latest.get('Volatility_20d'))

        # Volume
        signals['volume_signal'] = self._classify_volume(latest.get('Volume_Ratio'))

        # Bollinger Bands
        signals['bb_position'] = latest.get('BB_Position', 50) if pd.notna(latest.get('BB_Position')) else 50

        return signals
    
    def _classify_trend(self, row: pd.Series) -> str:
        """Classifica tendência baseado em SMAs."""
        sma20 = row.get('SMA_20')
        sma50 = row.get('SMA_50')
        close = row.get('Close')
        
        if pd.isna(sma20) or pd.isna(sma50):
            return 'unknown'
        
        if close > sma20 > sma50:
            return 'strong_uptrend'
        elif close > sma20:
            return 'uptrend'
        elif close < sma20 < sma50:
            return 'strong_downtrend'
        elif close < sma20:
            return 'downtrend'
        else:
            return 'sideways'
    
    def _classify_rsi(self, rsi: float) -> str:
        """Classifica sinal do RSI."""
        if pd.isna(rsi):
            return 'unknown'
        if rsi > 70:
            return 'overbought'
        elif rsi > 60:
            return 'bullish'
        elif rsi < 30:
            return 'oversold'
        elif rsi < 40:
            return 'bearish'
        else:
            return 'neutral'
    
    def _classify_macd(self, macd: float, signal: float) -> str:
        """Classifica sinal do MACD."""
        if pd.isna(macd) or pd.isna(signal):
            return 'unknown'
        if macd > signal and macd > 0:
            return 'strong_bullish'
        elif macd > signal:
            return 'bullish'
        elif macd < signal and macd < 0:
            return 'strong_bearish'
        else:
            return 'bearish'
    
    def _classify_volatility(self, vol: float) -> str:
        """Classifica regime de volatilidade."""
        if pd.isna(vol):
            return 'unknown'
        if vol > 40:
            return 'high'
        elif vol > 25:
            return 'medium'
        else:
            return 'low'
    
    def _classify_volume(self, ratio: float) -> str:
        """Classifica sinal de volume."""
        if pd.isna(ratio):
            return 'unknown'
        if ratio > 2:
            return 'high'
        elif ratio > 1.5:
            return 'above_average'
        elif ratio < 0.5:
            return 'low'
        else:
            return 'normal'


def calculate_indicators_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função helper para calcular indicadores para um DataFrame de ações.
    
    Args:
        df: DataFrame com colunas ['Open', 'High', 'Low', 'Close', 'Volume']
    
    Returns:
        DataFrame com indicadores adicionados
    """
    calculator = TechnicalIndicators(df)
    return calculator.calculate_all()


if __name__ == "__main__":
    # Teste com dados fictícios
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
    close_prices = np.cumsum(np.random.randn(250) * 0.02) + 100
    
    test_df = pd.DataFrame({
        'Open': close_prices + np.random.randn(250) * 0.5,
        'High': close_prices + np.random.rand(250) * 2,
        'Low': close_prices - np.random.rand(250) * 2,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 250)
    }, index=dates)
    
    calculator = TechnicalIndicators(test_df)
    result = calculator.calculate_all()
    
    print("Indicadores calculados:")
    print(f"Colunas: {result.columns.tolist()}")
    print(f"\nÚltima linha:")
    print(result.iloc[-1][['Close', 'SMA_20', 'RSI_14', 'MACD', 'ATR', 'BB_Position']])
    print(f"\nSinais mais recentes:")
    print(calculator.get_latest_signals())
