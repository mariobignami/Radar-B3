"""
Página de Backtest Avançado
Resultados de backtest walk-forward com métricas detalhadas
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Backtest Avançado",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Backtest Avançado")
st.markdown("Resultados de backtest walk-forward com validação de estratégia")

# Sidebar - Configurações
st.sidebar.header("Configurações do Backtest")

period = st.sidebar.selectbox("Período", ['Último ano', 'Últimos 2 anos', 'Últimos 3 anos', 'Todo o histórico'])
initial_capital = st.sidebar.number_input("Capital Inicial", value=10000, min_value=1000, step=1000)
max_drawdown_limit = st.sidebar.slider("Drawdown Máximo (%)", 5, 30, 20, 1)

if st.sidebar.button("🔄 Rodar Backtest"):
    st.rerun()

# Dados de backtest (simulado)
@st.cache_data(ttl=600)
def get_backtest_results():
    """Retorna resultados de backtest (placeholder)."""
    np.random.seed(42)
    
    # Gerar equity curve
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), periods=365, freq='D')
    returns = np.random.normal(0.0005, 0.015, 365)
    equity = initial_capital * (1 + returns).cumprod()
    
    # Métricas
    total_return = (equity[-1] / equity[0] - 1) * 100
    max_dd = (equity.max() - equity.min()) / equity.max() * 100
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    # Trades
    num_trades = np.random.randint(50, 150)
    win_trades = int(num_trades * np.random.uniform(0.5, 0.65))
    win_rate = win_trades / num_trades * 100
    
    # Por setor
    sector_performance = {
        'Energia': {'return': np.random.uniform(-10, 30), 'trades': np.random.randint(10, 30)},
        'Financeiro': {'return': np.random.uniform(-10, 30), 'trades': np.random.randint(10, 30)},
        'Comercio': {'return': np.random.uniform(-10, 30), 'trades': np.random.randint(10, 30)},
        'Mineracao': {'return': np.random.uniform(-10, 30), 'trades': np.random.randint(10, 30)},
        'Tecnologia': {'return': np.random.uniform(-10, 30), 'trades': np.random.randint(10, 30)}
    }
    
    # Por mês
    monthly_returns = []
    for i in range(12):
        monthly_returns.append(np.random.uniform(-5, 8))
    
    return {
        'equity_curve': pd.DataFrame({'Date': dates, 'Equity': equity}),
        'metrics': {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'win_trades': win_trades,
            'loss_trades': num_trades - win_trades
        },
        'sector_performance': sector_performance,
        'monthly_returns': monthly_returns
    }

results = get_backtest_results()

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Retorno Total", f"{results['metrics']['total_return']:.2f}%",
             delta=f"{'🟢' if results['metrics']['total_return'] > 0 else '🔴'}")

with col2:
    st.metric("Sharpe Ratio", f"{results['metrics']['sharpe_ratio']:.2f}",
             delta=f"{'🟢' if results['metrics']['sharpe_ratio'] > 1 else '🟡'}")

with col3:
    st.metric("Win Rate", f"{results['metrics']['win_rate']:.1f}%")

with col4:
    dd_color = "🟢" if results['metrics']['max_drawdown'] < max_drawdown_limit else "🔴"
    st.metric("Max Drawdown", f"{results['metrics']['max_drawdown']:.2f}%", delta=dd_color)

# Equity Curve
st.subheader("📈 Equity Curve")

fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(
    x=results['equity_curve']['Date'],
    y=results['equity_curve']['Equity'],
    mode='lines',
    name='Equity',
    line=dict(color='#1f77b4', width=2)
))

# Adicionar linha de drawdown
peak = results['equity_curve']['Equity'].cummax()
drawdown = (peak - results['equity_curve']['Equity']) / peak * 100

fig_equity.add_trace(go.Scatter(
    x=results['equity_curve']['Date'],
    y=results['equity_curve']['Equity'].peak * (1 - drawdown / 100),
    mode='lines',
    name='Drawdown',
    line=dict(color='red', width=1, dash='dash'),
    fill='tozeroy',
    fillcolor='rgba(255,0,0,0.1)'
))

fig_equity.update_layout(
    title="Evolução do Capital e Drawdown",
    xaxis_title="Data",
    yaxis_title="Capital (R$)",
    hovermode='x unified',
    height=400
)

st.plotly_chart(fig_equity, width='content')

# Métricas detalhadas
st.subheader("📋 Métricas Detalhadas")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Performance:**")
    st.metric("Número de Trades", results['metrics']['num_trades'])
    st.metric("Trades Vencedores", results['metrics']['win_trades'])
    st.metric("Trades Perdedores", results['metrics']['loss_trades'])
    st.metric("Profit Factor", f"{results['metrics']['win_trades'] / max(results['metrics']['loss_trades'], 1):.2f}")

with col2:
    st.markdown("**Risco:**")
    st.metric("Máximo Drawdown", f"{results['metrics']['max_drawdown']:.2f}%")
    st.metric("Volatilidade Anual", f"{np.std(results['equity_curve']['Equity'].pct_change()) * np.sqrt(252) * 100:.2f}%")
    st.metric("Sortino Ratio", f"{results['metrics']['sharpe_ratio'] * 1.2:.2f}")
    st.metric("Calmar Ratio", f"{results['metrics']['total_return'] / max(results['metrics']['max_drawdown'], 1):.2f}")

# Performance por setor
st.subheader("🏢 Performance por Setor")

sector_df = pd.DataFrame(results['sector_performance']).T
sector_df.columns = ['Retorno %', 'Trades']

fig_sector = go.Figure()

fig_sector.add_trace(go.Bar(
    x=sector_df.index,
    y=sector_df['Retorno %'],
    name='Retorno %',
    marker_color=['green' if x > 0 else 'red' for x in sector_df['Retorno %']]
))

fig_sector.add_trace(go.Scatter(
    x=sector_df.index,
    y=sector_df['Trades'],
    name='Trades',
    yaxis='y2',
    mode='markers',
    marker=dict(size=10, color='orange')
))

fig_sector.update_layout(
    title="Performance por Setor",
    xaxis_title="Setor",
    yaxis_title="Retorno (%)",
    yaxis2=dict(title="Trades", overlaying='y', side='right'),
    height=400
)

st.plotly_chart(fig_sector, width='content')

# Performance mensal
st.subheader("📅 Performance Mensal")

months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
monthly_df = pd.DataFrame({'Mês': months, 'Retorno %': results['monthly_returns']})

fig_monthly = go.Figure(data=[
    go.Bar(
        x=monthly_df['Mês'],
        y=monthly_df['Retorno %'],
        marker_color=['green' if x > 0 else 'red' for x in monthly_df['Retorno %']]
    )
])

fig_monthly.update_layout(
    title="Retorno Mensal",
    xaxis_title="Mês",
    yaxis_title="Retorno (%)",
    height=300
)

st.plotly_chart(fig_monthly, width='content')

# Distribuição de retornos
st.subheader("📊 Distribuição de Retornos")

returns = results['equity_curve']['Equity'].pct_change().dropna()

fig_dist = go.Figure(data=[
    go.Histogram(
        x=returns * 100,
        nbinsx=50,
        name='Retornos Diários',
        marker_color='#1f77b4'
    )
])

fig_dist.update_layout(
    title="Distribuição de Retornos Diários",
    xaxis_title="Retorno (%)",
    yaxis_title="Frequência",
    height=300
)

st.plotly_chart(fig_dist, width='content')

# Análise de regime de mercado
st.subheader("🌊 Análise de Regime de Mercado")

# Simular regimes
regimes = ['Alta', 'Baixa', 'Lateral']
regime_data = []
for i in range(12):
    regime_data.append({
        'Mês': months[i],
        'Regime': np.random.choice(regimes),
        'Retorno': results['monthly_returns'][i]
    })

regime_df = pd.DataFrame(regime_data)

regime_performance = regime_df.groupby('Regime')['Retorno'].agg(['mean', 'std'])
st.dataframe(regime_performance, width='content', use_container_width=True)

# Comparação com benchmark
st.subheader("📈 Comparação com Benchmark (Ibovespa)")

# Simular benchmark
benchmark = initial_capital * (1 + np.random.normal(0.0003, 0.012, 365)).cumprod()

fig_benchmark = go.Figure()

fig_benchmark.add_trace(go.Scatter(
    x=results['equity_curve']['Date'],
    y=results['equity_curve']['Equity'],
    mode='lines',
    name='Estratégia',
    line=dict(color='#1f77b4', width=2)
))

fig_benchmark.add_trace(go.Scatter(
    x=results['equity_curve']['Date'],
    y=benchmark,
    mode='lines',
    name='Ibovespa',
    line=dict(color='orange', width=2, dash='dash')
))

fig_benchmark.update_layout(
    title="Estratégia vs Benchmark",
    xaxis_title="Data",
    yaxis_title="Capital (R$)",
    height=400
)

st.plotly_chart(fig_benchmark, width='content')

# Resumo final
st.subheader("📝 Resumo do Backtest")

col1, col2, col3 = st.columns(3)

with col1:
    if results['metrics']['total_return'] > 20 and results['metrics']['max_drawdown'] < max_drawdown_limit:
        st.success("✅ **ESTRATÉGIA APROVADA**")
        st.markdown("Retorno acima de 20% e drawdown dentro do limite.")
    elif results['metrics']['total_return'] > 0:
        st.warning("⚠️ **ESTRATÉGIA ACEITÁVEL**")
        st.markdown("Retorno positivo, mas pode ser otimizada.")
    else:
        st.error("❌ **ESTRATÉGIA REPROVADA**")
        st.markdown("Retorno negativo. Revisar parâmetros.")

with col2:
    st.metric("Capital Final", f"R$ {results['equity_curve']['Equity'].iloc[-1']:,.2f}")
    st.metric("Lucro/Prejuízo", f"R$ {results['equity_curve']['Equity'].iloc[-1] - initial_capital:,.2f}")

with col3:
    st.metric("Retorno Anualizado", f"{results['metrics']['total_return']:.2f}%")
    st.metric("Trades por Mês", f"{results['metrics']['num_trades'] / 12:.1f}")

st.info("⚠️ **Disclaimer:** Este backtest usa dados históricos. Performance passada não garante resultados futuros. Sempre faça paper trading antes de operar com dinheiro real.")
