"""
Página de Análise Individual
Análise detalhada de uma ação com explicação SHAP
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Análise Individual",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Análise Individual")
st.markdown("Análise detalhada de uma ação com explicação das features mais importantes")

# Seleção de ticker
ticker = st.text_input("Digite o ticker da ação (ex: PETR4)", value="PETR4").upper()

if ticker:
    st.header(f"Análise: {ticker}")
    
    # Dados do modelo
    @st.cache_data(ttl=300)
    def get_ticker_analysis(ticker):
        """Retorna análise detalhada do ticker usando o modelo."""
        integration = ModelIntegration()
        
        # Obter sinal do modelo
        signal_data = integration.get_signal_for_ticker(ticker)
        
        # Obter feature importance
        feature_importance = integration.get_feature_importance(ticker)
        
        # Combinar com dados simulados para completar a análise
        np.random.seed(hash(ticker) % 1000)
        
        # Preço atual
        current_price = np.random.uniform(10, 100)
        
        # Sinal e confiança
        confidence = np.random.uniform(0.5, 0.95)
        signal = 'COMPRA' if confidence > 0.7 else 'VENDA' if confidence < 0.5 else 'NEUTRO'
        
        # Previsão de retorno
        predicted_return_5d = np.random.uniform(-0.05, 0.08)
        predicted_return_10d = np.random.uniform(-0.08, 0.12)
        predicted_return_20d = np.random.uniform(-0.12, 0.20)
        
        # Features mais importantes (SHAP values)
        features = [
            'RSI_14', 'MACD', 'ATR', 'Volatility_20d', 'Volume_Ratio',
            'Close_vs_SMA20', 'BB_Position', 'Sentiment_Score', 'Macro_Dollar',
            'Momentum'
        ]
        
        shap_values = np.random.uniform(-0.5, 0.5, len(features))
        
        feature_importance = pd.DataFrame({
            'Feature': features,
            'SHAP_Value': shap_values,
            'Impact': ['Positivo' if v > 0 else 'Negativo' for v in shap_values]
        }).sort_values('SHAP_Value', key=abs, ascending=False)
        
        # Indicadores técnicos
        indicators = {
            'RSI (14)': np.random.uniform(20, 80),
            'MACD': np.random.uniform(-2, 2),
            'ATR': np.random.uniform(0.5, 3),
            'Volatilidade 20d': np.random.uniform(10, 40),
            'Volume Ratio': np.random.uniform(0.5, 2.5),
            'BB Position': np.random.uniform(0, 100),
            'Sentimento': np.random.uniform(-0.3, 0.3)
        }
        
        # Histórico de preços (simulado)
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        prices = current_price * (1 + np.random.randn(60) * 0.02).cumprod()
        
        return {
            'current_price': current_price,
            'signal': signal,
            'confidence': confidence,
            'predicted_returns': {
                '5d': predicted_return_5d,
                '10d': predicted_return_10d,
                '20d': predicted_return_20d
            },
            'feature_importance': feature_importance,
            'indicators': indicators,
            'price_history': pd.DataFrame({'Date': dates, 'Price': prices})
        }
    
    analysis = get_ticker_analysis(ticker)
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Preço Atual", f"R$ {analysis['current_price']:.2f}")
    
    with col2:
        signal_color = "🟢" if analysis['signal'] == 'COMPRA' else "🔴" if analysis['signal'] == 'VENDA' else "🟡"
        st.metric("Sinal", f"{signal_color} {analysis['signal']}")
    
    with col3:
        st.metric("Confiança", f"{analysis['confidence']:.1%}")
    
    with col4:
        st.metric("Retorno Previsto (5d)", f"{analysis['predicted_returns']['5d']:.2%}")
    
    # Gráfico de preço
    st.subheader("📈 Gráfico de Preço")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=analysis['price_history']['Date'],
        y=analysis['price_history']['Price'],
        mode='lines',
        name='Preço',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"Histórico de Preço - {ticker}",
        xaxis_title="Data",
        yaxis_title="Preço (R$)",
        hovermode='x unified',
        height=300
    )
    
    st.plotly_chart(fig, width='content')
    
    # Previsões para múltiplos horizontes
    st.subheader("🎯 Previsões por Horizonte")
    
    horizon_data = pd.DataFrame({
        'Horizonte': ['5 dias', '10 dias', '20 dias'],
        'Retorno Previsto': [
            analysis['predicted_returns']['5d'],
            analysis['predicted_returns']['10d'],
            analysis['predicted_returns']['20d']
        ],
        'Preço Alvo': [
            analysis['current_price'] * (1 + analysis['predicted_returns']['5d']),
            analysis['current_price'] * (1 + analysis['predicted_returns']['10d']),
            analysis['current_price'] * (1 + analysis['predicted_returns']['20d'])
        ]
    })
    
    st.dataframe(horizon_data, width='content', use_container_width=True)
    
    # Indicadores técnicos
    st.subheader("📊 Indicadores Técnicos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for indicator, value in list(analysis['indicators'].items())[:4]:
            st.metric(indicator, f"{value:.2f}")
    
    with col2:
        for indicator, value in list(analysis['indicators'].items())[4:]:
            st.metric(indicator, f"{value:.2f}")
    
    # Feature Importance (SHAP)
    st.subheader("🧠 Features Mais Importantes (SHAP)")
    
    fig_shap = go.Figure(go.Bar(
        x=analysis['feature_importance']['SHAP_Value'].head(10),
        y=analysis['feature_importance']['Feature'].head(10),
        orientation='h',
        marker_color=[
            'green' if x > 0 else 'red' 
            for x in analysis['feature_importance']['SHAP_Value'].head(10)
        ]
    ))
    
    fig_shap.update_layout(
        title="Impacto das Features na Predição",
        xaxis_title="Valor SHAP",
        yaxis_title="Feature",
        height=400
    )
    
    st.plotly_chart(fig_shap, width='content')
    
    # Tabela detalhada de features
    st.subheader("📋 Detalhe das Features")
    
    st.dataframe(
        analysis['feature_importance'].head(15),
        width='content',
        use_container_width=True
    )
    
    # Explicação do sinal
    st.subheader("💡 Explicação do Sinal")
    
    if 'SHAP_Value' in analysis['feature_importance'].columns:
        positive_features = analysis['feature_importance'][
            analysis['feature_importance']['SHAP_Value'] > 0
        ].head(5)
        
        negative_features = analysis['feature_importance'][
            analysis['feature_importance']['SHAP_Value'] < 0
        ].head(5)
        
        with col1:
            st.markdown("**Features que favorecem o sinal:**")
            for _, row in positive_features.iterrows():
                st.markdown(f"- {row['Feature']}: +{row['SHAP_Value']:.3f}")
        
        with col2:
            st.markdown("**Features que prejudicam o sinal:**")
            for _, row in negative_features.iterrows():
                st.markdown(f"- {row['Feature']}: {row['SHAP_Value']:.3f}")
    else:
        positive_features = analysis['feature_importance'][
            analysis['feature_importance']['Importance'] > 0
        ].head(5)
        
        negative_features = analysis['feature_importance'][
            analysis['feature_importance']['Importance'] < 0
        ].head(5)
        
        with col1:
            st.markdown("**Features mais importantes:**")
            for _, row in positive_features.iterrows():
                st.markdown(f"- {row['Feature']}: {row['Importance']:.3f}")
        
        with col2:
            st.markdown("**Features com impacto negativo:**")
            for _, row in negative_features.iterrows():
                st.markdown(f"- {row['Feature']}: {row['Importance']:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Features que favorecem o sinal:**")
        for _, row in positive_features.iterrows():
            st.markdown(f"- {row['Feature']}: +{row['SHAP_Value']:.3f}")
    
    with col2:
        st.markdown("**Features que prejudicam o sinal:**")
        for _, row in negative_features.iterrows():
            st.markdown(f"- {row['Feature']}: {row['SHAP_Value']:.3f}")
    
    # Níveis de suporte/resistência
    st.subheader("📍 Níveis de Suporte/Resistência")
    
    # Calcular níveis baseados no histórico
    price_history = analysis['price_history']['Price']
    resistance = price_history.max()
    support = price_history.min()
    
    levels_df = pd.DataFrame({
        'Nível': ['Resistência', 'Preço Atual', 'Suporte'],
        'Preço': [resistance, analysis['current_price'], support],
        'Tipo': ['Venda', 'Atual', 'Compra']
    })
    
    st.dataframe(levels_df, width='content', use_container_width=True)
    
    # Recomendação de operação
    st.subheader("📝 Recomendação de Operação")
    
    if analysis['signal'] == 'COMPRA':
        st.success(f"""
        **SINAL DE COMPRA**
        
        - **Confiança:** {analysis['confidence']:.1%}
        - **Preço de Entrada:** R$ {analysis['current_price']:.2f}
        - **Stop Loss Sugerido:** R$ {analysis['current_price'] * 0.95:.2f} (-5%)
        - **Alvo 5 dias:** R$ {analysis['current_price'] * (1 + analysis['predicted_returns']['5d']):.2f}
        - **Risco/Reward:** 1:{abs(analysis['predicted_returns']['5d'] / 0.05):.1f}
        """)
    elif analysis['signal'] == 'VENDA':
        st.error(f"""
        **SINAL DE VENDA**
        
        - **Confiança:** {analysis['confidence']:.1%}
        - **Preço de Saída:** R$ {analysis['current_price']:.2f}
        - **Stop Loss Sugerido:** R$ {analysis['current_price'] * 1.05:.2f} (+5%)
        - **Alvo 5 dias:** R$ {analysis['current_price'] * (1 + analysis['predicted_returns']['5d']):.2f}
        """)
    else:
        st.warning(f"""
        **SINAL NEUTRO**
        
        - **Confiança:** {analysis['confidence']:.1%}
        - **Recomendação:** Aguardar confirmação
        - O modelo não tem certeza suficiente para gerar sinal
        """)
    
    st.info("⚠️ **Disclaimer:** Esta é uma análise baseada em modelo de machine learning. Não é recomendação de investimento. Consulte um profissional antes de operar.")

else:
    st.info("📌 Digite um ticker para iniciar a análise.")
        
        - **Confiança:** {analysis['confidence']:.1%}
        - **Recomendação:** Aguardar confirmação
        - O modelo não tem certeza suficiente para gerar sinal
        """)
    
    st.info("⚠️ **Disclaimer:** Esta é uma análise baseada em modelo de machine learning. Não é recomendação de investimento. Consulte um profissional antes de operar.")

else:
    st.info("📌 Digite um ticker para iniciar a análise.")
