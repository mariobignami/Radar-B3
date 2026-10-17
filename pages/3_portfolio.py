"""
Página de Portfolio
Gestão de posições com controle de drawdown
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="Portfolio",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Portfolio")
st.markdown("Gestão de posições com controle de risco e drawdown")

# Inicializar portfolio no session_state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': 10000.0,
        'positions': [],
        'total_capital': 10000.0,
        'peak_value': 10000.0,
        'current_drawdown': 0.0
    }

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Capital Total", f"R$ {st.session_state.portfolio['total_capital']:,.2f}")

with col2:
    st.metric("Cash Disponível", f"R$ {st.session_state.portfolio['cash']:,.2f}")

with col3:
    positions_value = st.session_state.portfolio['total_capital'] - st.session_state.portfolio['cash']
    st.metric("Valor em Posições", f"R$ {positions_value:,.2f}")

with col4:
    dd_pct = st.session_state.portfolio['current_drawdown'] * 100
    st.metric("Drawdown Atual", f"{dd_pct:.2f}%")

# Controle de drawdown
st.subheader("📊 Controle de Drawdown")

progress = st.session_state.portfolio['current_drawdown']
st.progress(progress, text=f"Drawdown: {progress*100:.2f}%")

if progress > 0.15:
    st.error("⚠️ Drawdown acima de 15% - Reduzir tamanho das posições")
elif progress > 0.10:
    st.warning("⚠️ Drawdown acima de 10% - Atenção necessária")
else:
    st.success("✅ Drawdown dentro dos limites aceitáveis")

# Posições atuais
st.subheader("📋 Posições Atuais")

if st.session_state.portfolio['positions']:
    positions_df = pd.DataFrame(st.session_state.portfolio['positions'])
    st.dataframe(positions_df, width='content', use_container_width=True)
else:
    st.info("📌 Nenhuma posição aberta. Use o Radar de Oportunidades para abrir novas posições.")

# Formulário para abrir nova posição
st.subheader("➕ Abrir Nova Posição")

with st.expander("Clique para abrir nova posição"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker_new = st.text_input("Ticker", value="PETR4").upper()
        entry_price = st.number_input("Preço de Entrada", value=30.0, min_value=0.01, step=0.01)
    
    with col2:
        confidence = st.slider("Confiança do Modelo", 0.0, 1.0, 0.7, 0.1)
        sector = st.selectbox("Setor", ['Energia', 'Financeiro', 'Comercio', 'Mineracao', 'Tecnologia'])
    
    with col3:
        quantity = st.number_input("Quantidade", value=100, min_value=100, step=100)
        notes = st.text_input("Notas", value="")
    
    if st.button("Abrir Posição"):
        risk = entry_price * 0.05
        stop_loss = entry_price - risk
        tp1 = entry_price + (risk * 3)
        tp2 = entry_price + (risk * 5)
        tp3 = entry_price + (risk * 8)
        
        new_position = {
            'ticker': ticker_new,
            'quantity': quantity,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'take_profit_3': tp3,
            'sector': sector,
            'notes': notes,
            'entry_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        st.session_state.portfolio['positions'].append(new_position)
        st.session_state.portfolio['cash'] -= entry_price * quantity
        st.session_state.portfolio['total_capital'] = st.session_state.portfolio['cash'] + sum(
            p['entry_price'] * p['quantity'] for p in st.session_state.portfolio['positions']
        )
        
        st.success(f"✅ Posição aberta: {ticker_new} - {quantity} ações @ R${entry_price:.2f}")
        st.rerun()

st.info("⚠️ **Disclaimer:** Este é um sistema de simulação. Não use para operações reais sem validação prévia.")
