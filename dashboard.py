"""
Dashboard Streamlit para Previsão de Ações B3
Visualiza predições, backtest e métricas de performance
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path

from src.predict import StockPredictor
from src.utils_indicators import (
    DEFAULT_OPTIONAL_INDICATORS,
    calculate_optional_indicators,
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_optional_indicators(stock_code):
    return calculate_optional_indicators(stock_code)

# Configuração da página
st.set_page_config(
    page_title="B3 Stock Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success {
        color: #00aa00;
        font-weight: bold;
    }
    .warning {
        color: #ff9900;
        font-weight: bold;
    }
    .danger {
        color: #ff0000;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("📈 B3 Stock Prediction Dashboard")
st.markdown("Machine Learning para Previsão de Ações da Bolsa Brasileira")

# ⚠️ DISCLAIMER IMPORTANTE - SEMPRE VISÍVEL
st.warning("""
🚨 **AVISO LEGAL IMPORTANTE**
- ⚠️ Este é um PROJETO EDUCACIONAL. Não é recomendação de investimento.
- ⚠️ Backtesting em dados históricos ≠ Performance futura garantida
- ⚠️ RISCO: Você pode PERDER dinheiro seguindo essas predições
- ⚠️ Consulte um profissional antes de investir

**Leia os avisos em cada página para entender completamente os riscos.**
""")

# Sidebar
st.sidebar.header("⚙️ Configurações")
page = st.sidebar.radio(
    "Selecione a página:",
    ["🔮 Predições", "📊 Backtest", "📈 Análise", "🎯 Recomendações", "ℹ️ Sobre"]
)

# ==================== PÁGINA: PREDIÇÕES ====================
if page == "🔮 Predições":
    st.header("🔮 Faça uma Predição")

    dim_company_path = Path("data/raw/dimCompany.csv")
    if dim_company_path.exists():
        dim_company = pd.read_csv(dim_company_path)
    else:
        dim_company = pd.DataFrame()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dados da Ação")
        open_price = st.number_input("Preço de Abertura (R$)", min_value=1.0, value=28.5, step=0.1)
        high_price = st.number_input("Preço Máximo (R$)", min_value=1.0, value=29.2, step=0.1)
        low_price = st.number_input("Preço Mínimo (R$)", min_value=1.0, value=28.1, step=0.1)
        volume = st.number_input("Volume (milhões)", min_value=0.0, value=500.0, step=10.0)
    
    with col2:
        st.subheader("Características")
        if not dim_company.empty:
            stock_codes = dim_company['stockCodeCompany'].tolist()
            selected_code = st.selectbox("Ticker", stock_codes)
            sector_default = dim_company.loc[
                dim_company['stockCodeCompany'] == selected_code, 'sectorCompany'
            ].iloc[0]
            segment_default = dim_company.loc[
                dim_company['stockCodeCompany'] == selected_code, 'segmentCompany'
            ].iloc[0]
            sector_options = sorted(dim_company['sectorCompany'].unique().tolist())
            segment_options = sorted(dim_company['segmentCompany'].unique().tolist())
            sector = st.selectbox(
                "Setor",
                sector_options,
                index=sector_options.index(sector_default) if sector_default in sector_options else 0
            )
            segment = st.selectbox(
                "Segmento",
                segment_options,
                index=segment_options.index(segment_default) if segment_default in segment_options else 0
            )
        else:
            selected_code = None
            sector = st.selectbox("Setor", ["Energia", "Mineracao", "Financeiro", "Bebidas", "Comercio"])
            segment = st.selectbox("Segmento", ["Petroleo", "Ferro", "Banco", "Bebidas", "Varejo"])
        month = st.slider("Mês", 1, 12, 4)
        day_week = st.slider("Dia da Semana (0=Seg, 4=Sex)", 0, 6, 1)

    technical_indicators = None
    indicator_source = "automático"

    with st.expander("📈 Indicadores Técnicos (Opcional)", expanded=False):
        st.info(
            "Use o modo automático para buscar indicadores recentes via yfinance. "
            "Se quiser simular um cenário específico, alterne para preenchimento manual."
        )
        st.markdown("""
        **Como ler estes indicadores:**
        - **RSI 14**: mede forca compradora/vendedora nos ultimos 14 periodos. Abaixo de 30 tende a indicar sobrevenda; acima de 70, sobrecompra.
        - **Vol 20d**: oscilacao recente dos retornos. Quanto maior, maior o risco de o preco fugir da previsao.
        - **Drawdown**: maior queda desde um topo recente no historico usado. Numeros muito negativos indicam ativo muito castigado.
        - **Gap**: diferenca entre a abertura mais recente e o fechamento anterior. Gap positivo abre acima; negativo abre abaixo.
        - **Vol Rel**: volume atual contra a media de 20 dias. 100% e volume normal; acima disso indica negociacao mais intensa.
        """)

        indicator_mode = st.radio(
            "Origem dos indicadores",
            ["Calcular automaticamente", "Informar manualmente"],
            horizontal=True,
            key="technical_indicator_mode",
        )

        if indicator_mode == "Informar manualmente":
            indicator_source = "manual"
            tech_col1, tech_col2 = st.columns(2)
            with tech_col1:
                rsi_14 = st.number_input(
                    "RSI 14",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0,
                    help="Indice de Forca Relativa: acima de 70 sugere sobrecompra, abaixo de 30 sugere sobrevenda.",
                )
                volatility_20d = st.number_input(
                    "Volatilidade 20d (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=20.0,
                    step=0.5,
                    help="Desvio padrao dos retornos recentes, em percentual.",
                )
                max_drawdown = st.number_input(
                    "Maximo Drawdown (%)",
                    min_value=-100.0,
                    max_value=0.0,
                    value=-10.0,
                    step=0.5,
                    help="Pior queda acumulada no periodo analisado. Valor negativo.",
                )
            with tech_col2:
                open_gap = st.number_input(
                    "Gap de Abertura (%)",
                    min_value=-20.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.1,
                    help="Diferenca percentual entre a abertura informada e o fechamento anterior.",
                )
                volume_rel = st.number_input(
                    "Volume Relativo 20d (%)",
                    min_value=0.0,
                    max_value=500.0,
                    value=100.0,
                    step=5.0,
                    help="Volume atual em relacao a media de 20 dias. 100% equivale a media.",
                )

            technical_indicators = {
                "rsi_14": rsi_14,
                "volatility_20d_percent": volatility_20d,
                "max_drawdown_percent": max_drawdown,
                "open_gap_percent": open_gap,
                "volume_rel_20d_percent": volume_rel,
            }
        else:
            try:
                with st.spinner("Calculando indicadores via yfinance..."):
                    technical_indicators = get_cached_optional_indicators(selected_code)

                st.caption(
                    f"Calculado para {technical_indicators['ticker']} em "
                    f"{technical_indicators['last_date']}. "
                    f"Gap calculado com abertura de R$ {technical_indicators['open_used_for_gap']:.2f}."
                )
                metric_cols = st.columns(5)
                metric_cols[0].metric("RSI 14", f"{technical_indicators['rsi_14']:.1f}")
                metric_cols[1].metric("Vol 20d", f"{technical_indicators['volatility_20d_percent']:.2f}%")
                metric_cols[2].metric("Drawdown", f"{technical_indicators['max_drawdown_percent']:.2f}%")
                metric_cols[3].metric("Gap", f"{technical_indicators['open_gap_percent']:.2f}%")
                metric_cols[4].metric("Vol Rel", f"{technical_indicators['volume_rel_20d_percent']:.1f}%")
            except Exception as indicator_error:
                st.warning(
                    "Nao foi possivel calcular automaticamente agora. "
                    f"Usando defaults conservadores: {indicator_error}"
                )
                technical_indicators = DEFAULT_OPTIONAL_INDICATORS.copy()
                indicator_source = "default"

    if st.button("🚀 Fazer Predição", key="predict_btn"):
        try:
            predictor = StockPredictor()
            result = predictor.predict_single(
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                quantity=volume * 1_000_000,
                stock_code=selected_code,
                sector=sector,
                segment=segment,
                month=month,
                day_week=day_week,
                technical_indicators=technical_indicators
            )
            
            if result['predicted_price']:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Abertura", f"R$ {open_price:.2f}")
                
                with col2:
                    variacao = ((result['predicted_price'] - open_price) / open_price * 100)
                    st.metric("Fechamento Previsto", f"R$ {result['predicted_price']:.2f}", 
                             delta=f"{variacao:+.2f}%")
                
                with col3:
                    st.metric("Intervalo", f"R$ {high_price - low_price:.2f}")

                if technical_indicators:
                    st.caption(f"Indicadores técnicos usados: origem {indicator_source}.")
                    indicator_cols = st.columns(5)
                    indicator_cols[0].metric("RSI 14", f"{technical_indicators['rsi_14']:.1f}")
                    indicator_cols[1].metric("Vol 20d", f"{technical_indicators['volatility_20d_percent']:.2f}%")
                    indicator_cols[2].metric("Drawdown", f"{technical_indicators['max_drawdown_percent']:.2f}%")
                    indicator_cols[3].metric("Gap", f"{technical_indicators['open_gap_percent']:.2f}%")
                    indicator_cols[4].metric("Vol Rel", f"{technical_indicators['volume_rel_20d_percent']:.1f}%")
                
                # Gráfico
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=['Abertura', 'Fechamento\nPrevisto'],
                    y=[open_price, result['predicted_price']],
                    mode='lines+markers',
                    name='Preço',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    title="Predição de Preço",
                    xaxis_title="Período",
                    yaxis_title="Preço (R$)",
                    hovermode='x unified',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("❌ Erro ao fazer predição")
        except Exception as e:
            st.error(f"❌ Erro: {e}")

# ==================== PÁGINA: BACKTEST ====================
elif page == "📊 Backtest":
    st.header("📊 Resultados do Backtest")
    
    # Carregar resultados
    if Path("backtest_results.json").exists():
        with open("backtest_results.json", "r") as f:
            results = json.load(f)
        
        df = pd.DataFrame(results)
        
        # Estatísticas gerais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Acurácia Média", f"{df['accuracy'].mean():.2f}%", 
                     delta=f"{df['accuracy'].std():.2f}%")
        
        with col2:
            st.metric("Melhor Acerto", f"{df['accuracy'].max():.2f}%")
        
        with col3:
            st.metric("Pior Acerto", f"{df['accuracy'].min():.2f}%")
        
        with col4:
            st.metric("Erro Médio", f"R$ {df['error'].mean():.2f}")

        if 'smape' in df.columns:
            st.metric("SMAPE Médio", f"{df['smape'].mean():.2f}%")
        
        # Gráfico por data
        st.subheader("📅 Acurácia por Data")
        df_by_date = df.groupby('date')['accuracy'].agg(['mean', 'min', 'max'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_by_date.index,
            y=df_by_date['mean'],
            mode='lines+markers',
            name='Média',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_by_date.index,
            y=df_by_date['max'],
            mode='lines',
            name='Máxima',
            line=dict(color='green', width=1, dash='dash'),
            fill='tozeroy'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_by_date.index,
            y=df_by_date['min'],
            mode='lines',
            name='Mínima',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title="Acurácia ao Longo do Tempo",
            xaxis_title="Data",
            yaxis_title="Acurácia (%)",
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico por empresa
        st.subheader("🏢 Acurácia por Empresa")
        df_by_company = df.groupby('company')['accuracy'].agg(['mean', 'std'])
        
        fig = px.bar(
            df_by_company.reset_index(),
            x='company',
            y='mean',
            error_y='std',
            color='mean',
            color_continuous_scale='RdYlGn',
            title="Acurácia Média por Ação"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de resultados
        st.subheader("📋 Detalhes do Backtest")
        display_cols = ['date', 'company', 'real', 'predicted', 'error', 'accuracy']
        if 'smape' in df.columns:
            display_cols.append('smape')
        st.dataframe(
            df[display_cols].sort_values('date', ascending=False),
            use_container_width=True,
            height=400
        )
    else:
        st.info("📌 Nenhum resultado de backtest disponível. Execute `python backtest_model.py` primeiro.")

# ==================== PÁGINA: ANÁLISE ====================
elif page == "📈 Análise":
    st.header("📈 Análise do Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Informações do Modelo")
        st.markdown("""
        - **Tipo:** Regressão Linear
        - **Dados:** Periodo maximo disponivel
        - **Amostras:** Variavel (depende do numero de empresas)
        - **Features:** 11 características
        - **Empresas:** Varias (varejo, energia e outros nichos)
        - **R² Treino:** 0.9998
        - **R² Teste:** 0.9998
        """)
    
    with col2:
        st.subheader("🎯 Performance")
        st.markdown("""
        - **Acuracia Media:** Variavel (ver backtest)
        - **Erro Medio:** Variavel (ver backtest)
        - **Dias Testados:** Variavel
        - **Total de Predicoes:** Variavel
        - **Melhor Dia:** Variavel
        - **Status:** ✅ Modelo Validado
        """)
    
    st.subheader("📚 Como Funciona")
    st.markdown("""
    1. **Ingestão de Dados:** Carrega dados históricos de OHLC da B3
    2. **Processamento:** Normaliza, codifica e prepara features
    3. **Treinamento:** Treina modelo de Regressão Linear
    4. **Validacao:** Testa em dados nao vistos
    5. **Predição:** Prevê preço de fechamento com entrada de abertura
    
    ### Features Utilizadas:
    - Preço de Abertura (Open)
    - Preço Máximo (High)
    - Preço Mínimo (Low)
    - Volume Negociado
    - Valor da Moeda (Câmbio)
    - Dia do Mês
    - Dia da Semana
    - Mês
    - Ano
    - Setor da Empresa
    - Segmento
    """)

# ==================== PÁGINA: RECOMENDACOES ====================
elif page == "🎯 Recomendações":
    st.switch_page("pages/4_recomendacoes_amanha.py")

# ==================== PÁGINA: SOBRE ====================
elif page == "ℹ️ Sobre":
    st.header("ℹ️ Sobre este Projeto")
    
    st.markdown("""
    ## 🎯 Objetivo
    Prever o preço de fechamento de ações da B3 (Bolsa de Valores Brasileira) 
    usando Machine Learning com dados históricos e características do mercado.
    
    ## 🛠️ Tecnologias
    - **Python 3.12**
    - **scikit-learn** - Machine Learning
    - **Pandas** - Análise de dados
    - **Streamlit** - Dashboard web
    - **Plotly** - Visualizações
    - **yfinance** - Dados do Yahoo Finance
    
    ## 📊 Dataset
    - **Periodo:** maximo disponivel no Yahoo Finance
    - **Frequencia:** Diaria
    - **Acoes:** Varias (varejo, energia e outros nichos)
    - **Total de Registros:** Variavel (com dados completos)
    
    ## 📈 Performance
    - **Acuracia Media:** Variavel (ver backtest)
    - **Erro Medio:** Variavel (ver backtest)
    - **Melhor Performance:** Variavel (ver backtest)
    - **Robustez:** Modelo validado em backtests recentes
    
    ## 🚀 Como Usar
    
    ### Instalação
    ```bash
    pip install -r requirements.txt
    ```
    
    ### Scripts Disponíveis
    ```bash
    # Baixar dados da B3
    python get_b3_data.py
    
    # Treinar modelo
    python run_pipeline_simple.py
    
    # Fazer predição para uma data
    python predict_date.py 2026-04-30
    
    # Validar modelo
    python validate_model.py
    
    # Backtest em múltiplos dias
    python backtest_model.py
    
    # Dashboard web
    streamlit run dashboard.py
    ```
    
    ## 👨‍💻 Desenvolvimento
    **Projeto:** Radar B3  
    **Data:** Maio 2026  
    **Repositório:** github.com/mariobignami/Radar-B3
    
    ## ⚠️ DISCLAIMER CRÍTICO
    
    ### Este é um PROJETO EDUCACIONAL
    
    **NÃO é recomendação de investimento. O modelo pode estar ERRADO.**
    
    ### Limitações Críticas:
    
    1. **Backtesting ≠ Futuro Real**
       - Acurácia de 99.46% é com **dados históricos já conhecidos**
       - O futuro é desconhecido e pode ser COMPLETAMENTE diferente
       - Eventos inesperados podem quebrar o modelo
    
    2. **Coisas que o modelo NÃO consegue prever:**
       - 📰 Notícias de mercado (positivas ou negativas)
       - 🏛️ Decisões políticas e econômicas
       - 💥 Crises, guerras, pandemias
       - 📊 Mudanças de tendência do mercado
       - 🏢 Anúncios das empresas
       - 😟 Sentimento e pânico do mercado
    
    3. **Riscos Reais:**
       - ❌ Você pode PERDER DINHEIRO
       - ❌ Volatilidade extrema quebra o modelo
       - ❌ Uma notícia ruim muda tudo
       - ❌ Correlações históricas podem não se repetir
    
    ### Por que ainda assim 99.46% de acurácia?
    
    - ✓ Os dados históricos passados tendem a se repetir
    - ✓ Padrões curtos são previsíveis
    - ✓ Mas AMANHÃ pode ser diferente
    - ✓ Especialmente em crises ou eventos
    
    ### Antes de usar em DINHEIRO REAL:
    
    ✋ **PARAR. LEIA TUDO ABAIXO**
    
    1. 📚 **Estude**: Entenda como funciona o modelo
    2. 💬 **Consulte**: Fale com um analista financeiro profissional
    3. 🧪 **Teste**: Use valores MUITO pequenos primeiro
    4. 📈 **Diversifique**: Nunca coloque tudo em uma predição
    5. ⏰ **Monitore**: Acompanhe se as predições estão funcionando
    6. 🛑 **Pare se**: Começar a perder dinheiro
    
    ### Disclaimer Legal:
    
    Este software é fornecido "como está" sem garantias. Os autores não são responsáveis 
    por perdas financeiras. Você usa este projeto por sua conta e risco.
    
    **Consulte sempre um profissional antes de investir dinheiro real.**
    """)

st.sidebar.divider()
st.sidebar.markdown("""
    ---
    **B3 Stock Prediction v1.0**  
    Powered by Machine Learning & Streamlit
""")
