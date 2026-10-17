"""
Módulo de Dados Macroeconômicos
Coleta dados macro do Brasil (Bacen, IBGE) e internacionais
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional
import requests


class MacroDataCollector:
    """Coleta dados macroeconômicos para análise de mercado."""
    
    def __init__(self):
        self.bacen_api = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados/ultimos/{n}/formato/json"
    
    def get_selic_rate(self, days: int = 365) -> pd.DataFrame:
        """
        Coleta taxa Selic do Bacen.
        
        Args:
            days: Número de dias de histórico
        
        Returns:
            DataFrame com data e taxa Selic
        """
        try:
            # Código da Selic no Bacen: 432
            url = self.bacen_api.format(code='432', n=days)
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df['valor'] = df['valor'].astype(float) / 100  # Converter de % para decimal
            df.columns = ['date', 'selic_rate']
            
            return df.set_index('date')
        except Exception as e:
            print(f"❌ Erro ao buscar Selic: {e}")
            return pd.DataFrame()
    
    def get_dollar_rate(self, days: int = 365) -> pd.DataFrame:
        """
        Coleta taxa de câmbio USD/BRL do Bacen.
        
        Args:
            days: Número de dias de histórico
        
        Returns:
            DataFrame com data e taxa de câmbio
        """
        try:
            # Código do dólar comercial no Bacen: 1
            url = self.bacen_api.format(code='1', n=days)
            response = requests.get(url, timeout=10)
            data = response.json()
            
            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df['valor'] = df['valor'].astype(float)
            df.columns = ['date', 'usd_brl']
            
            return df.set_index('date')
        except Exception as e:
            print(f"❌ Erro ao buscar dólar: {e}")
            # Fallback para yfinance
            return self._get_dollar_yf(days)
    
    def _get_dollar_yf(self, days: int) -> pd.DataFrame:
        """Fallback para buscar dólar via yfinance."""
        try:
            ticker = yf.Ticker('USDBRL=X')
            df = ticker.history(period=f'{days}d')
            
            if df.empty:
                return pd.DataFrame()
            
            result = pd.DataFrame({
                'usd_brl': df['Close']
            })
            result.index.name = 'date'
            
            return result
        except Exception as e:
            print(f"❌ Erro ao buscar dólar via yfinance: {e}")
            return pd.DataFrame()
    
    def get_ipca(self, months: int = 12) -> pd.DataFrame:
        """
        Coleta IPCA (inflação) do Bacen.
        
        Args:
            months: Número de meses de histórico
        
        Returns:
            DataFrame com data e IPCA
        """
        try:
            # Código do IPCA no Bacen: 433
            url = self.bacen_api.format(code='433', n=months)
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df['valor'] = df['valor'].astype(float)
            df.columns = ['date', 'ipca']
            
            return df.set_index('date')
        except Exception as e:
            print(f"❌ Erro ao buscar IPCA: {e}")
            return pd.DataFrame()
    
    def get_ibovespa(self, days: int = 365) -> pd.DataFrame:
        """
        Coleta dados do Ibovespa via yfinance.
        
        Args:
            days: Número de dias de histórico
        
        Returns:
            DataFrame com dados do Ibovespa
        """
        try:
            ticker = yf.Ticker('^BVSP')
            df = ticker.history(period=f'{days}d')
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'date'
            
            return df
        except Exception as e:
            print(f"❌ Erro ao buscar Ibovespa: {e}")
            return pd.DataFrame()
    
    def get_sp500(self, days: int = 365) -> pd.DataFrame:
        """
        Coleta dados do S&P 500 via yfinance.
        
        Args:
            days: Número de dias de histórico
        
        Returns:
            DataFrame com dados do S&P 500
        """
        try:
            ticker = yf.Ticker('^GSPC')
            df = ticker.history(period=f'{days}d')
            
            if df.empty:
                return pd.DataFrame()
            
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'date'
            
            return df
        except Exception as e:
            print(f"❌ Erro ao buscar S&P 500: {e}")
            return pd.DataFrame()
    
    def get_commodities(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados de commodities via yfinance.
        
        Args:
            days: Número de dias de histórico
        
        Returns:
            Dicionário com DataFrames de commodities
        """
        commodities = {
            'petroleo': 'CL=F',
            'minerio': 'VALE3.SA',  # Usando VALE como proxy para minério
            'ouro': 'GC=F'
        }
        
        result = {}
        for name, ticker in commodities.items():
            try:
                df = yf.Ticker(ticker).history(period=f'{days}d')
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    df.index.name = 'date'
                    result[name] = df
            except Exception as e:
                print(f"⚠️ Erro ao buscar {name}: {e}")
        
        return result
    
    def get_all_macro_data(self, days: int = 365) -> pd.DataFrame:
        """
        Coleta todos os dados macroeconômicos e retorna DataFrame consolidado.
        
        Args:
            days: Número de dias de histórico
        
        Returns:
            DataFrame com todos os dados macro
        """
        # Coletar dados individuais
        selic = self.get_selic_rate(days)
        dollar = self.get_dollar_rate(days)
        ipca = self.get_ipca(12)  # IPCA é mensal
        ibov = self.get_ibovespa(days)
        sp500 = self.get_sp500(days)
        
        # Criar DataFrame consolidado
        all_data = {}
        
        if not selic.empty:
            all_data['selic_rate'] = selic['selic_rate']
        
        if not dollar.empty:
            all_data['usd_brl'] = dollar['usd_brl']
        
        if not ipca.empty:
            all_data['ipca'] = ipca['ipca']
        
        if not ibov.empty:
            all_data['ibov_close'] = ibov['close']
            all_data['ibov_return'] = ibov['close'].pct_change()
        
        if not sp500.empty:
            all_data['sp500_close'] = sp500['close']
            all_data['sp500_return'] = sp500['close'].pct_change()
        
        if not all_data:
            return pd.DataFrame()
        
        # Consolidar em um DataFrame
        result = pd.DataFrame(all_data)
        
        # Forward fill para dados mensais (IPCA, Selic)
        result = result.ffill()
        
        # Calcular features derivadas
        if 'usd_brl' in result.columns:
            result['dollar_change'] = result['usd_brl'].pct_change()
        
        if 'selic_rate' in result.columns and 'usd_brl' in result.columns:
            # Diferencial de juros (aproximação)
            result['interest_differential'] = result['selic_rate']  # Simplificado
        
        if 'ibov_close' in result.columns and 'usd_brl' in result.columns:
            # Ibovespa ajustado pelo dólar
            result['ibov_usd_adjusted'] = result['ibov_close'] / result['usd_brl']
        
        return result
    
    def get_macro_features_for_date(self, date: datetime) -> Dict:
        """
        Retorna features macroeconômicas para uma data específica.
        
        Args:
            date: Data desejada
        
        Returns:
            Dicionário com features macro
        """
        df = self.get_all_macro_data(365)
        
        if df.empty:
            return {}
        
        # Encontrar data mais próxima
        date_str = pd.Timestamp(date).normalize()
        if date_str in df.index:
            row = df.loc[date_str]
        else:
            # Usar última data disponível antes da data desejada
            available_dates = df.index[df.index <= date_str]
            if len(available_dates) == 0:
                row = df.iloc[-1]
            else:
                row = df.loc[available_dates[-1]]
        
        features = {
            'selic_rate': row.get('selic_rate', np.nan),
            'usd_brl': row.get('usd_brl', np.nan),
            'dollar_change': row.get('dollar_change', np.nan),
            'ipca': row.get('ipca', np.nan),
            'ibov_return': row.get('ibov_return', np.nan),
            'sp500_return': row.get('sp500_return', np.nan),
            'interest_differential': row.get('interest_differential', np.nan)
        }
        
        return features


def get_macro_summary() -> Dict:
    """
    Função helper para obter resumo macroeconômico atual.
    
    Returns:
        Dicionário com indicadores macro atuais
    """
    collector = MacroDataCollector()
    
    selic = collector.get_selic_rate(1)
    dollar = collector.get_dollar_rate(1)
    ipca = collector.get_ipca(1)
    ibov = collector.get_ibovespa(1)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'selic': selic['selic_rate'].iloc[-1] if not selic.empty else None,
        'dollar': dollar['usd_brl'].iloc[-1] if not dollar.empty else None,
        'ipca': ipca['ipca'].iloc[-1] if not ipca.empty else None,
        'ibov': ibov['close'].iloc[-1] if not ibov.empty else None
    }
    
    return summary


if __name__ == "__main__":
    # Teste
    collector = MacroDataCollector()
    
    print("=== Teste Coleta Macro ===")
    
    print("\nSelic (últimos 7 dias):")
    selic = collector.get_selic_rate(7)
    if not selic.empty:
        print(selic)
    
    print("\nDólar (últimos 7 dias):")
    dollar = collector.get_dollar_rate(7)
    if not dollar.empty:
        print(dollar)
    
    print("\nIPCA (últimos 12 meses):")
    ipca = collector.get_ipca(12)
    if not ipca.empty:
        print(ipca)
    
    print("\nIbovespa (últimos 7 dias):")
    ibov = collector.get_ibovespa(7)
    if not ibov.empty:
        print(ibov[['close']].tail())
    
    print("\n=== Dados Consolidados ===")
    all_data = collector.get_all_macro_data(30)
    if not all_data.empty:
        print(f"Shape: {all_data.shape}")
        print(f"Colunas: {all_data.columns.tolist()}")
        print(f"\nÚltima linha:")
        print(all_data.iloc[-1])
    
    print("\n=== Resumo Atual ===")
    summary = get_macro_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
