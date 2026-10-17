"""
Página de Radar de Oportunidades
Lista ações com sinais de compra/venda baseados no modelo XGBoost
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Adicionar diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.model_integration import get_model_signals
except ImportError:
    from model_integration import get_model_signals

# Configuração da página
st.set_page_config(
    page_title="Radar de Oportunidades",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Radar de Oportunidades")
st.markdown("Sinais de compra/venda baseados no modelo XGBoost para swing trade")

# Sidebar - Filtros
st.sidebar.header("Filtros")

# Filtro de R:R mínimo
min_rr = st.sidebar.slider("R:R Mínimo", 1.0, 10.0, 3.0, 0.5)

# Filtro de confiança mínima
min_confidence = st.sidebar.slider("Confiança Mínima", 0.0, 1.0, 0.6, 0.1)

# Filtro de setor
sectors = ['Todos', 'Energia', 'Financeiro', 'Comercio', 'Mineracao', 'Tecnologia']
selected_sector = st.sidebar.selectbox("Setor", sectors)

# Filtro de tipo de sinal
signal_types = ['Todos', 'COMPRA', 'VENDA', 'NEUTRO']
selected_signal = st.sidebar.selectbox("Tipo de Sinal", signal_types)

# Botão de atualização
if st.sidebar.button("🔄 Atualizar Sinais"):
    st.rerun()

# Dados do modelo XGBoost
@st.cache_data(ttl=300)  # Cache por 5 minutos
def get_signals():
    """Retorna sinais do modelo XGBoost."""
    try:
        return get_model_signals()
    except Exception as e:
        st.error(f"Erro ao carregar sinais do modelo: {e}")
        # Fallback para dados simulados
        np.random.seed(42)
        
        tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'MGLU3', 'LREN3', 'AMER3', 
                   'PRIO3', 'HAPV3', 'CVCB3', 'CYRE3', 'MRVE3', 'PCAR3', 'ASAI3']
        
        data = []
        for ticker in tickers:
            price = np.random.uniform(10, 100)
            confidence = np.random.uniform(0.4, 0.95)
            
            if confidence > 0.7:
                signal = 'COMPRA'
            elif confidence < 0.5:
                signal = 'VENDA'
            else:
                signal = 'NEUTRO'
            
            rr = np.random.uniform(1.5, 6.0)
            risk = price * 0.05
            target = price + (risk * rr) if signal == 'COMPRA' else price - (risk * rr)
            stop = price - risk if signal == 'COMPRA' else price + risk
            sector = np.random.choice(['Energia', 'Financeiro', 'Comercio', 'Mineracao', 'Tecnologia'])
            
            data.append({
                'Ticker': ticker,
                'Preço': round(price, 2),
                'Sinal': signal,
                'Confiança': round(confidence, 2),
                'R:R': round(rr, 2),
                'Alvo': round(target, 2),
                'Stop': round(stop, 2),
                'Setor': sector,
                'Potencial %': round((target - price) / price * 100, 2) if signal == 'COMPRA' else round((price - target) / price * 100, 2),
                'Risco %': round(risk / price * 100, 2)
            })
        
        return pd.DataFrame(data)

# Carregar sinais
df_signals = get_signals()

# Aplicar filtros
if selected_sector != 'Todos':
    df_signals = df_signals[df_signals['Setor'] == selected_sector]

if selected_signal != 'Todos':
    df_signals = df_signals[df_signals['Sinal'] == selected_signal]

df_signals = df_signals[
    (df_signals['Confiança'] >= min_confidence) &
    (df_signals['R:R'] >= min_rr)
]

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total de Sinais", len(df_signals))

with col2:
    compras = len(df_signals[df_signals['Sinal'] == 'COMPRA'])
    st.metric("Sinais de Compra", compras, delta=f"{compras/len(df_signals)*100:.1f}%" if len(df_signals) > 0 else 0)

with col3:
    vendas = len(df_signals[df_signals['Sinal'] == 'VENDA'])
    st.metric("Sinais de Venda", vendas, delta=f"{vendas/len(df_signals)*100:.1f}%" if len(df_signals) > 0 else 0)

with col4:
    avg_rr = df_signals['R:R'].mean() if len(df_signals) > 0 else 0
    st.metric("R:R Médio", round(avg_rr, 2))

# Tabela de sinais
st.subheader("📋 Sinais Ativos")

if len(df_signals) > 0:
    # Formatar tabela com cores
    def highlight_signal(val):
        if val == 'COMPRA':
            return 'background-color: #d4edda'
        elif val == 'VENDA':
            return 'background-color: #f8d7da'
        else:
            return ''
    
    def highlight_confidence(val):
        if val >= 0.8:
            return 'background-color: #d1ecf1'
        elif val >= 0.6:
            return 'background-color: #fff3cd'
        else:
            return ''
    
    styled_df = df_signals.style.applymap(highlight_signal, subset=['Sinal'])
    styled_df = styled_df.applymap(highlight_confidence, subset=['Confiança'])
    
    st.dataframe(
        styled_df,
        width='content',
        height=400,
        use_container_width=True
    )
    
    # Gráfico de dispersão R:R vs Confiança
    st.subheader("📊 R:R vs Confiança")
    
    fig = go.Figure()
    
    for signal in ['COMPRA', 'VENDA', 'NEUTRO']:
        df_filtered = df_signals[df_signals['Sinal'] == signal]
        color = {'COMPRA': 'green', 'VENDA': 'red', 'NEUTRO': 'gray'}[signal]
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Confiança'],
            y=df_filtered['R:R'],
            mode='markers',
            name=signal,
            text=df_filtered['Ticker'],
            textposition='top center',
            marker=dict(
                size=12,
                color=color,
                line=dict(width=1, color='white')
            )
        ))
    
    fig.update_layout(
        title="Relação entre Confiança e Risco-Retorno",
        xaxis_title="Confiança do Modelo",
        yaxis_title="Risco-Retorno (R:R)",
        hovermode='closest',
        height=400
    )
    
    st.plotly_chart(fig, width='content')
    
    # Gráfico de distribuição por setor
    st.subheader("🏢 Sinais por Setor")
    
    sector_counts = df_signals.groupby('Setor').size()
    
    fig_sector = go.Figure(data=[
        go.Bar(
            x=sector_counts.index,
            y=sector_counts.values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        )
    ])
    
    fig_sector.update_layout(
        title="Distribuição de Sinais por Setor",
        xaxis_title="Setor",
        yaxis_title="Número de Sinais",
        height=400
    )
    
    st.plotly_chart(fig_sector, width='content')
    
else:
    st.info("📌 Nenhum sinal encontrado com os filtros atuais. Ajuste os filtros ou clique em 'Atualizar Sinais'.")

# Detalhes de uma ação selecionada
if len(df_signals) > 0:
    st.subheader("🔍 Análise Detalhada")
    
    selected_ticker = st.selectbox("Selecione uma ação para análise", df_signals['Ticker'].tolist())
    
    if selected_ticker:
        ticker_data = df_signals[df_signals['Ticker'] == selected_ticker].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Ticker", ticker_data['Ticker'])
            st.metric("Preço Atual", f"R$ {ticker_data['Preço']:.2f}")
        
        with col2:
            st.metric("Sinal", ticker_data['Sinal'])
            st.metric("Confiança", f"{ticker_data['Confiança']:.2%}")
        
        with col3:
            st.metric("R:R", ticker_data['R:R'])
            st.metric("Potencial", f"{ticker_data['Potencial %']:.2f}%")
        
        # Níveis de entrada/saída
        st.markdown("### 📊 Níveis de Operação")
        
        levels_df = pd.DataFrame({
            'Nível': ['Entrada', 'Alvo', 'Stop Loss'],
            'Preço': [ticker_data['Preço'], ticker_data['Alvo'], ticker_data['Stop']],
            'Tipo': ['Atual', 'Take Profit', 'Stop Loss']
        })
        
        st.dataframe(levels_df, width='content', use_container_width=True)
