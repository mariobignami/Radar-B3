"""
Integração do Modelo XGBoost com o Dashboard
Conecta o modelo treinado com as páginas do Streamlit
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.feature_engineering import FeatureEngineer
    from src.xgboost_model import XGBoostSwingTrader
    from src.technical_indicators import TechnicalIndicators
except ImportError:
    from feature_engineering import FeatureEngineer
    from xgboost_model import XGBoostSwingTrader
    from technical_indicators import TechnicalIndicators


class ModelIntegration:
    """Classe para integrar o modelo com o dashboard."""
    
    def __init__(self, model_path=None):
        """
        Inicializa a integração.
        
        Args:
            model_path: Caminho para o modelo salvo (opcional)
        """
        self.model = None
        self.feature_engineer = FeatureEngineer()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Carrega o modelo treinado."""
        try:
            self.model = XGBoostSwingTrader()
            self.model.load_model(model_path)
            print(f"✅ Modelo carregado de {model_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def get_signals_for_tickers(self, tickers):
        """
        Gera sinais para uma lista de tickers.
        
        Args:
            tickers: Lista de tickers (ex: ['PETR4', 'VALE3'])
        
        Returns:
            DataFrame com sinais
        """
        signals = []
        
        for ticker in tickers:
            try:
                signal = self.get_signal_for_ticker(ticker)
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"Erro ao processar {ticker}: {e}")
                continue
        
        return pd.DataFrame(signals) if signals else pd.DataFrame()
    
    def get_signal_for_ticker(self, ticker):
        """
        Gera sinal para um ticker específico.
        
        Args:
            ticker: Ticker da ação (ex: 'PETR4')
        
        Returns:
            Dicionário com sinal e métricas
        """
        # Em produção, buscar dados reais do yfinance
        # Por enquanto, usar dados simulados
        np.random.seed(hash(ticker) % 1000)
        
        # Preço atual
        current_price = np.random.uniform(10, 100)
        
        # Indicadores técnicos simulados
        rsi = np.random.uniform(20, 80)
        macd = np.random.uniform(-2, 2)
        atr = np.random.uniform(0.5, 3)
        
        # Se modelo estiver carregado, usar predição
        if self.model:
            # Criar features
            df = self._create_mock_dataframe(ticker, current_price)
            
            # Calcular indicadores
            tech = TechnicalIndicators(df)
            df = tech.calculate_all()
            
            # Criar features
            features_df = self.feature_engineer.create_features(df)
            
            # Predizer
            prediction = self.model.predict(features_df)
            confidence = abs(prediction)  # Simplificado
            
            if prediction > 0:
                signal = 'COMPRA'
            elif prediction < 0:
                signal = 'VENDA'
            else:
                signal = 'NEUTRO'
        else:
            # Sem modelo, usar regras simples
            if rsi < 30:
                signal = 'COMPRA'
                confidence = 0.7
            elif rsi > 70:
                signal = 'VENDA'
                confidence = 0.7
            else:
                signal = 'NEUTRO'
                confidence = 0.5
        
        # Calcular R:R
        risk = current_price * 0.05
        rr = np.random.uniform(2.0, 5.0)
        
        # Calcular alvo e stop
        target = current_price + (risk * rr) if signal == 'COMPRA' else current_price - (risk * rr)
        stop = current_price - risk if signal == 'COMPRA' else current_price + risk
        
        # Setor (simplificado)
        sector = self._get_sector(ticker)
        
        return {
            'Ticker': ticker,
            'Preço': round(current_price, 2),
            'Sinal': signal,
            'Confiança': round(confidence, 2),
            'R:R': round(rr, 2),
            'Alvo': round(target, 2),
            'Stop': round(stop, 2),
            'Setor': sector,
            'Potencial %': round((target - current_price) / current_price * 100, 2) if signal == 'COMPRA' else round((current_price - target) / current_price * 100, 2),
            'Risco %': round(risk / current_price * 100, 2),
            'RSI': round(rsi, 2),
            'MACD': round(macd, 2),
            'ATR': round(atr, 2)
        }
    
    def _create_mock_dataframe(self, ticker, current_price, periods=100):
        """Cria DataFrame simulado para o ticker."""
        dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='D')
        prices = current_price * (1 + np.random.randn(periods) * 0.02).cumprod()
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': prices * (1 + np.random.randn(periods) * 0.005),
            'High': prices * (1 + np.abs(np.random.randn(periods)) * 0.01),
            'Low': prices * (1 - np.abs(np.random.randn(periods)) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, periods)
        })
        
        df.set_index('Date', inplace=True)
        return df
    
    def _get_sector(self, ticker):
        """Retorna setor do ticker (simplificado)."""
        sector_mapping = {
            'PETR4': 'Energia', 'PETR3': 'Energia', 'PRIO3': 'Energia',
            'VALE3': 'Mineracao', 'CSNA3': 'Mineracao', 'USIM5': 'Mineracao',
            'ITUB4': 'Financeiro', 'BBDC4': 'Financeiro', 'BBAS3': 'Financeiro', 'SANB11': 'Financeiro',
            'MGLU3': 'Comercio', 'LREN3': 'Comercio', 'AMER3': 'Comercio', 'PCAR3': 'Comercio',
            'TOTS3': 'Tecnologia', 'POSI3': 'Tecnologia', 'LWSA3': 'Tecnologia'
        }
        return sector_mapping.get(ticker, 'Outros')
    
    def get_feature_importance(self, ticker):
        """
        Retorna importância das features para um ticker.
        
        Args:
            ticker: Ticker da ação
        
        Returns:
            DataFrame com feature importance
        """
        if not self.model:
            return None
        
        # Criar DataFrame simulado
        df = self._create_mock_dataframe(ticker, 50)
        tech = TechnicalIndicators(df)
        df = tech.calculate_all()
        features_df = self.feature_engineer.create_features(df)
        
        # Obter feature importance do modelo
        if hasattr(self.model.model, 'feature_importances_'):
            importance = self.model.model.feature_importances_
            feature_names = features_df.columns
            
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
        
        return None


# Função para uso no Streamlit
def get_model_signals(tickers=None):
    """
    Função helper para obter sinais no Streamlit.
    
    Args:
        tickers: Lista de tickers (opcional)
    
    Returns:
        DataFrame com sinais
    """
    integration = ModelIntegration()
    
    if tickers is None:
        tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'MGLU3', 'LREN3', 
                   'AMER3', 'PRIO3', 'HAPV3', 'CVCB3', 'CYRE3', 'MRVE3', 
                   'PCAR3', 'ASAI3', 'TOTS3', 'POSI3', 'LWSA3']
    
    return integration.get_signals_for_tickers(tickers)


if __name__ == "__main__":
    # Teste
    print("=== Teste Model Integration ===")
    
    integration = ModelIntegration()
    
    # Testar sinal para um ticker
    signal = integration.get_signal_for_ticker('PETR4')
    print("\nSinal para PETR4:")
    for key, value in signal.items():
        print(f"  {key}: {value}")
    
    # Testar sinais para múltiplos tickers
    tickers = ['PETR4', 'VALE3', 'ITUB4']
    signals = integration.get_signals_for_tickers(tickers)
    print(f"\nSinais para {len(signals)} tickers:")
    print(signals[['Ticker', 'Sinal', 'Confiança', 'R:R']])
