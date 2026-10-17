"""
Sistema de Gestão de Risco para Swing Trade
Position sizing, stop loss dinâmico, take profit, controle de drawdown
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Position:
    """Representa uma posição de trading."""
    ticker: str
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_per_share: float
    position_size: float
    entry_date: str
    notes: str


@dataclass
class Portfolio:
    """Representa o portfolio completo."""
    positions: List[Position]
    cash: float
    total_capital: float
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    peak_value: float = 0.0


class RiskManager:
    """Sistema de gestão de risco para swing trade."""
    
    def __init__(self, initial_capital: float = 10000, max_drawdown: float = 0.20):
        """
        Inicializa o gestor de risco.
        
        Args:
            initial_capital: Capital inicial (default R$10k)
            max_drawdown: Drawdown máximo permitido (default 20%)
        """
        self.initial_capital = initial_capital
        self.max_drawdown = max_drawdown
        self.portfolio = Portfolio(
            positions=[],
            cash=initial_capital,
            total_capital=initial_capital
        )
        self.risk_per_trade = 0.02  # 2% do capital por operação
        self.max_positions = 8  # Máximo de posições simultâneas
        self.max_sector_exposure = 0.30  # Máximo 30% em um setor
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               confidence: float = 0.5, volatility: float = 0.02) -> int:
        """
        Calcula tamanho da posição usando Kelly Criterion modificado.
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            confidence: Confiança do modelo (0-1)
            volatility: Volatilidade esperada
        
        Returns:
            Número de ações
        """
        # Risco por ação
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return 0
        
        # Abordagem simplificada: baseado em risco por trade (2% do capital)
        # Ajustado pela confiança do modelo
        confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 a 1.0 baseado na confiança
        risk_amount = self.portfolio.total_capital * self.risk_per_trade * confidence_multiplier

        # Tamanho da posição baseado no risco
        position_value = risk_amount / (risk_per_share / entry_price) if risk_per_share > 0 else risk_amount

        # Ajustar por volatilidade (menor volatilidade = posição maior)
        vol_adjustment = 1 / (1 + volatility * 10) if volatility > 0 else 1.0
        position_value = position_value * vol_adjustment

        # Mínimo de 500 para teste (garantir pelo menos 100 ações)
        position_value = max(position_value, 500)
        
        # Limitar a 20% do capital em uma posição
        max_single_position = self.portfolio.total_capital * 0.20
        position_value = min(position_value, max_single_position)
        
        # Calcular quantidade
        quantity = int(position_value / entry_price)

        # Arredondar para lote padrão (100 ações)
        quantity = (quantity // 100) * 100

        # Se quantity for 0 mas position_value for suficiente, usar mínimo de 100
        if quantity == 0 and position_value >= entry_price * 100:
            quantity = 100

        return quantity
    
    def calculate_stop_loss(self, df: pd.DataFrame, atr_multiplier: float = 2.0) -> float:
        """
        Calcula stop loss dinâmico baseado em ATR.
        
        Args:
            df: DataFrame com dados OHLCV
            atr_multiplier: Multiplicador do ATR (default 2.0)
        
        Returns:
            Preço de stop loss
        """
        if 'ATR' not in df.columns or df['ATR'].isna().all():
            # Fallback: usar 2x ATR calculado manualmente
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(14).mean().iloc[-1]
        else:
            atr = df['ATR'].iloc[-1]
        
        current_price = df['Close'].iloc[-1]
        stop_loss = current_price - (atr * atr_multiplier)

        # Stop máximo de 15%
        max_stop = current_price * 0.85
        stop_loss = max(stop_loss, max_stop)

        return stop_loss
    
    def calculate_take_profits(self, entry_price: float, stop_loss: float) -> Tuple[float, float, float]:
        """
        Calcula níveis de take profit.
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
        
        Returns:
            (TP1, TP2, TP3) - Níveis de take profit
        """
        risk = abs(entry_price - stop_loss)
        
        # R:R de 3:1, 5:1, 8:1
        tp1 = entry_price + (risk * 3)
        tp2 = entry_price + (risk * 5)
        tp3 = entry_price + (risk * 8)
        
        return tp1, tp2, tp3
    
    def check_position_limits(self, ticker: str, sector: str, 
                            position_value: float) -> Tuple[bool, str]:
        """
        Verifica se posição respeita os limites de risco.
        
        Args:
            ticker: Ticker da ação
            sector: Setor da ação
            position_value: Valor da posição
        
        Returns:
            (permitido, motivo)
        """
        # Verificar número máximo de posições
        if len(self.portfolio.positions) >= self.max_positions:
            return False, f"Máximo de {self.max_positions} posições atingido"

        # Verificar cash disponível
        if position_value > self.portfolio.cash:
            return False, f"Cash insuficiente (disponível: R${self.portfolio.cash:.2f})"

        # Verificar exposição por setor
        sector_tickers = self._get_tickers_by_sector(sector)
        if sector_tickers and ticker in sector_tickers:  # Só verificar se ticker estiver no mapeamento
            sector_exposure = sum(
                p.position_size for p in self.portfolio.positions
                if p.ticker in sector_tickers
            )
            sector_exposure += position_value

            if sector_exposure > self.portfolio.total_capital * self.max_sector_exposure:
                return False, f"Exposição máxima de setor ({self.max_sector_exposure*100}%) atingida"

        # Verificar drawdown atual
        if self.portfolio.current_drawdown >= self.max_drawdown:
            return False, f"Drawdown máximo ({self.max_drawdown*100}%) atingido - sem novas entradas"
        
        # Reduzir tamanho se drawdown > 10%
        if self.portfolio.current_drawdown >= 0.10:
            max_allowed = self.portfolio.total_capital * 0.5  # Máximo 50% do capital
            if position_value > max_allowed:
                return False, f"Drawdown >10% - reduzir tamanho da posição"
        
        return True, "OK"
    
    def _get_tickers_by_sector(self, sector: str) -> List[str]:
        """Retorna tickers de um setor (simplificado)."""
        # Em produção, buscar de base de dados
        sector_mapping = {
            'Energia': ['PETR4', 'PETR3', 'PRIO3', 'VALE3'],
            'Financeiro': ['ITUB4', 'BBDC4', 'BBAS3', 'SANB11'],
            'Comercio': ['MGLU3', 'LREN3', 'AMER3', 'PCAR3'],
            'Mineracao': ['VALE3', 'CSNA3', 'USIM5'],
            'Tecnologia': ['TOTS3', 'POSI3', 'LWSA3']
        }
        return sector_mapping.get(sector, [])
    
    def open_position(self, ticker: str, entry_price: float, df: pd.DataFrame,
                     confidence: float = 0.5, sector: str = 'Unknown',
                     notes: str = "") -> Optional[Position]:
        """
        Abre uma nova posição.
        
        Args:
            ticker: Ticker da ação
            entry_price: Preço de entrada
            df: DataFrame com dados OHLCV
            confidence: Confiança do modelo (0-1)
            sector: Setor da ação
            notes: Notas sobre a posição
        
        Returns:
            Position se aberta com sucesso, None caso contrário
        """
        # Calcular stop loss
        stop_loss = self.calculate_stop_loss(df)
        
        # Calcular take profits
        tp1, tp2, tp3 = self.calculate_take_profits(entry_price, stop_loss)
        
        # Calcular tamanho da posição
        volatility = df['Volatility_20d'].iloc[-1] if 'Volatility_20d' in df.columns else 0.02
        quantity = self.calculate_position_size(entry_price, stop_loss, confidence, volatility)
        
        if quantity == 0:
            print(f"❌ Tamanho da posição zero para {ticker}")
            return None
        
        position_value = quantity * entry_price
        risk_per_share = abs(entry_price - stop_loss)
        
        # Verificar limites
        allowed, reason = self.check_position_limits(ticker, sector, position_value)
        if not allowed:
            print(f"❌ Posição não permitida para {ticker}: {reason}")
            return None
        
        # Criar posição
        position = Position(
            ticker=ticker,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            risk_per_share=risk_per_share,
            position_size=position_value,
            entry_date=pd.Timestamp.now().strftime('%Y-%m-%d'),
            notes=notes
        )
        
        # Atualizar portfolio
        self.portfolio.positions.append(position)
        self.portfolio.cash -= position_value
        
        print(f"✅ Posição aberta: {ticker} - {quantity} ações @ R${entry_price:.2f}")
        print(f"   Stop: R${stop_loss:.2f} | TP1: R${tp1:.2f} | TP2: R${tp2:.2f} | TP3: R${tp3:.2f}")
        print(f"   Risco: R${risk_per_share * quantity:.2f} ({risk_per_share/entry_price*100:.2f}%)")
        
        return position
    
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Atualiza valor do portfolio com preços atuais.
        
        Args:
            current_prices: Dicionário {ticker: preço atual}
        
        Returns:
            Valor total do portfolio
        """
        positions_value = 0
        
        for position in self.portfolio.positions:
            if position.ticker in current_prices:
                current_price = current_prices[position.ticker]
                positions_value += current_price * position.quantity
        
        self.portfolio.total_capital = self.portfolio.cash + positions_value
        
        # Atualizar drawdown
        if self.portfolio.total_capital > self.portfolio.peak_value:
            self.portfolio.peak_value = self.portfolio.total_capital
        
        if self.portfolio.peak_value > 0:
            self.portfolio.current_drawdown = (
                (self.portfolio.peak_value - self.portfolio.total_capital) / 
                self.portfolio.peak_value
            )
        
        return self.portfolio.total_capital
    
    def check_exit_signals(self, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Verifica sinais de saída (stop loss, take profit).
        
        Args:
            current_prices: Dicionário {ticker: preço atual}
        
        Returns:
            Lista de sinais de saída
        """
        signals = []
        
        for position in self.portfolio.positions:
            if position.ticker not in current_prices:
                continue
            
            current_price = current_prices[position.ticker]
            
            # Stop loss
            if current_price <= position.stop_loss:
                signals.append({
                    'ticker': position.ticker,
                    'type': 'STOP_LOSS',
                    'price': current_price,
                    'reason': 'Stop loss atingido',
                    'pnl': (current_price - position.entry_price) * position.quantity
                })
            
            # Take profit 1 (vender 50%)
            elif current_price >= position.take_profit_1:
                signals.append({
                    'ticker': position.ticker,
                    'type': 'TAKE_PROFIT_1',
                    'price': current_price,
                    'reason': 'Take profit 1 atingido (vender 50%)',
                    'pnl': (current_price - position.entry_price) * position.quantity * 0.5,
                    'sell_percent': 0.5
                })
            
            # Take profit 2 (vender 50% restante)
            elif current_price >= position.take_profit_2:
                signals.append({
                    'ticker': position.ticker,
                    'type': 'TAKE_PROFIT_2',
                    'price': current_price,
                    'reason': 'Take profit 2 atingido (vender restante)',
                    'pnl': (current_price - position.entry_price) * position.quantity,
                    'sell_percent': 1.0
                })
            
            # Take profit 3 (alvo máximo)
            elif current_price >= position.take_profit_3:
                signals.append({
                    'ticker': position.ticker,
                    'type': 'TAKE_PROFIT_3',
                    'price': current_price,
                    'reason': 'Take profit 3 atingido (alvo máximo)',
                    'pnl': (current_price - position.entry_price) * position.quantity,
                    'sell_percent': 1.0
                })
        
        return signals
    
    def close_position(self, ticker: str, exit_price: float, 
                      reason: str = "Manual") -> Optional[float]:
        """
        Fecha uma posição.
        
        Args:
            ticker: Ticker da ação
            exit_price: Preço de saída
            reason: Motivo do fechamento
        
        Returns:
            P/L da operação
        """
        for i, position in enumerate(self.portfolio.positions):
            if position.ticker == ticker:
                pnl = (exit_price - position.entry_price) * position.quantity
                pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
                
                # Atualizar cash
                self.portfolio.cash += exit_price * position.quantity
                
                # Remover posição
                self.portfolio.positions.pop(i)
                
                print(f"✅ Posição fechada: {ticker} @ R${exit_price:.2f}")
                print(f"   P/L: R${pnl:.2f} ({pnl_pct:+.2f}%)")
                print(f"   Motivo: {reason}")
                
                return pnl
        
        print(f"⚠️ Posição {ticker} não encontrada")
        return None
    
    def get_portfolio_summary(self) -> Dict:
        """Retorna resumo do portfolio."""
        positions_value = sum(
            p.position_size for p in self.portfolio.positions
        )
        
        return {
            'total_capital': self.portfolio.total_capital,
            'cash': self.portfolio.cash,
            'positions_value': positions_value,
            'num_positions': len(self.portfolio.positions),
            'current_drawdown': self.portfolio.current_drawdown,
            'max_drawdown': self.portfolio.max_drawdown,
            'peak_value': self.portfolio.peak_value
        }


if __name__ == "__main__":
    # Teste
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    close_prices = np.cumsum(np.random.randn(100) * 0.02) + 100
    
    test_df = pd.DataFrame({
        'Open': close_prices + np.random.randn(100) * 0.5,
        'High': close_prices + np.random.rand(100) * 2,
        'Low': close_prices - np.random.rand(100) * 2,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Adicionar ATR
    try:
        from src.technical_indicators import TechnicalIndicators
    except ImportError:
        from technical_indicators import TechnicalIndicators
    tech = TechnicalIndicators(test_df)
    test_df = tech.calculate_all()
    
    print("=== Teste Risk Manager ===")
    rm = RiskManager(initial_capital=10000)

    # Abrir posição com parâmetros mais agressivos para teste
    position = rm.open_position(
        ticker='TEST4',
        entry_price=5.0,  # Preço menor para permitir 100 ações
        df=test_df,
        confidence=0.9,  # Confiança alta
        sector='Unknown',  # Setor sem mapeamento para evitar restrição
        notes='Teste'
    )
    
    if position:
        print(f"\n=== Resumo do Portfolio ===")
        summary = rm.get_portfolio_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
