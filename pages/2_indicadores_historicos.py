"""
Página de Indicadores Técnicos Históricos
Visualiza SMA, RSI, ATR, volatilidade e outros indicadores em série temporal
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json

st.set_page_config(
    page_title="Indicadores Históricos",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Indicadores Técnicos Históricos")
st.write("Visualize a evolução dos indicadores técnicos ao longo do tempo")

# ==================== CARREGAR DADOS ====================

# Lista de ações disponíveis
data_dir = Path("data/raw")
def consolidated_files_available():
    required = ["dimCompany.csv", "dimTime.csv", "factStocks.csv"]
    return all((data_dir / file_name).exists() for file_name in required)

def load_stock_codes():
    if consolidated_files_available():
        dim_company = pd.read_csv(data_dir / "dimCompany.csv")
        if "stockCodeCompany" in dim_company.columns:
            return sorted(dim_company["stockCodeCompany"].dropna().astype(str).unique().tolist())

    prefixes = ("ABEV", "ALPA", "AMER", "ASAI", "BBDC", "BRAP", "CASH", "CEAB", "COGN", "CSAN", "CSNA", "CVCB", "CYRE", "GMAT", "ITUB", "LJQQ", "LREN", "MGLU", "MRVE", "OIBR", "PCAR", "PETR", "POMO", "PRIO", "QUAL", "RECV", "UGPA", "USIM", "VALE", "VBBR", "VIVA", "WEGE")
    return sorted([f.stem for f in data_dir.glob("*.csv") if f.stem.startswith(prefixes)])

def load_stock_history(stock_code):
    if consolidated_files_available():
        dim_company = pd.read_csv(data_dir / "dimCompany.csv")
        company = dim_company[dim_company["stockCodeCompany"].astype(str) == stock_code]

        if not company.empty:
            company_row = company.iloc[0]
            fact_stocks = pd.read_csv(data_dir / "factStocks.csv")
            dim_time = pd.read_csv(data_dir / "dimTime.csv", usecols=["keyTime", "datetime"])

            df_stock = fact_stocks[fact_stocks["keyCompany"] == company_row["keyCompany"]].copy()
            df_stock = df_stock.merge(dim_time, on="keyTime", how="left")
            df_stock["dateStock"] = pd.to_datetime(df_stock["datetime"])
            df_stock["ticker"] = f"{stock_code}.SA"
            df_stock["nameCompany"] = company_row.get("nameCompany", stock_code)

            columns = [
                "dateStock", "ticker", "nameCompany",
                "openValueStock", "highValueStock", "lowValueStock",
                "closeValueStock", "quantityStock",
            ]
            return df_stock[columns], "base consolidada atualizada"

    csv_path = data_dir / f"{stock_code}.csv"
    if not csv_path.exists():
        return pd.DataFrame(), "CSV individual"

    return pd.read_csv(csv_path), "CSV individual legado"

csv_files = load_stock_codes()

if not csv_files:
    st.error("Nenhum dado de acao encontrado em data/raw")
    st.stop()

default_stock_index = csv_files.index("MGLU3") if "MGLU3" in csv_files else 0

# Seletor de ação
stock_code = st.sidebar.selectbox("Selecione a ação:", csv_files, index=default_stock_index)

# ==================== CARREGAR E PROCESSAR DADOS ====================


# Carregar dados
df, data_source = load_stock_history(stock_code)

if df.empty:
    st.error(f"Dados de {stock_code} nao encontrados")
    st.stop()

df['dateStock'] = pd.to_datetime(df['dateStock'])
df = df.sort_values('dateStock').reset_index(drop=True)

last_update = df['dateStock'].max().strftime('%Y-%m-%d')
st.caption(f"Fonte dos dados: {data_source}. Ultima data carregada para {stock_code}: {last_update}.")

# Pegar apenas últimos 252 dias (1 ano de pregões)
df_viz = df.tail(252).copy()

if len(df_viz) < 20:
    st.warning(f"⚠️ Menos de 20 dias de dados disponíveis. Total: {len(df_viz)} dias")

# ==================== CALCULAR INDICADORES ====================

def calcular_rsi(series, periodo=14):
    """Calcula RSI (Relative Strength Index)"""
    delta = series.diff()
    ganhos = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perdas = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    rs = ganhos / perdas
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_atr(high, low, close, periodo=14):
    """Calcula ATR (Average True Range)"""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodo).mean()
    return atr

def calcular_volatilidade(returns, periodo=20):
    """Calcula volatilidade móvel"""
    vol_diaria = returns.rolling(window=periodo).std()
    vol_anualizada = vol_diaria * np.sqrt(252) * 100
    return vol_anualizada

def calcular_macd(close_series, fast=12, slow=26, signal=9):
    """Calcula MACD, linha de sinal e histograma."""
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calcular_obv(close_series, volume_series):
    """Calcula On Balance Volume."""
    obv = [0]
    for i in range(1, len(close_series)):
        if close_series.iloc[i] > close_series.iloc[i - 1]:
            obv.append(obv[-1] + volume_series.iloc[i])
        elif close_series.iloc[i] < close_series.iloc[i - 1]:
            obv.append(obv[-1] - volume_series.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close_series.index)

def calcular_bollinger(close_series, periodo=20, desvios=2):
    """Calcula Banda de Bollinger."""
    media = close_series.rolling(window=periodo).mean()
    std = close_series.rolling(window=periodo).std()
    superior = media + (std * desvios)
    inferior = media - (std * desvios)
    return media, superior, inferior

def preparar_timeframe(base_df, rule):
    """Agrupa OHLCV em um novo horizonte temporal."""
    frame = base_df.set_index('dateStock').resample(rule).agg({
        'openValueStock': 'first',
        'highValueStock': 'max',
        'lowValueStock': 'min',
        'closeValueStock': 'last',
        'quantityStock': 'sum',
    }).dropna().reset_index()
    return frame

def enriquecer_indicadores(frame):
    frame = frame.copy()
    frame['SMA_20'] = frame['closeValueStock'].rolling(window=20).mean()
    frame['SMA_50'] = frame['closeValueStock'].rolling(window=50).mean()
    frame['SMA_200'] = frame['closeValueStock'].rolling(window=200).mean()
    frame['RSI_14'] = calcular_rsi(frame['closeValueStock'], 14)
    frame['ATR_14'] = calcular_atr(frame['highValueStock'], frame['lowValueStock'], frame['closeValueStock'], 14)
    frame['Returns'] = frame['closeValueStock'].pct_change()
    frame['Volatilidade'] = calcular_volatilidade(frame['Returns'], 20)
    frame['MACD'], frame['MACD_SIGNAL'], frame['MACD_HIST'] = calcular_macd(frame['closeValueStock'])
    frame['OBV'] = calcular_obv(frame['closeValueStock'], frame['quantityStock'])
    frame['BB_MID'], frame['BB_UPPER'], frame['BB_LOWER'] = calcular_bollinger(frame['closeValueStock'])
    frame['MACD_CROSS_UP'] = (frame['MACD'].shift(1) <= frame['MACD_SIGNAL'].shift(1)) & (frame['MACD'] > frame['MACD_SIGNAL'])
    frame['MACD_CROSS_DOWN'] = (frame['MACD'].shift(1) >= frame['MACD_SIGNAL'].shift(1)) & (frame['MACD'] < frame['MACD_SIGNAL'])
    frame['BB_TOUCH_UPPER'] = frame['closeValueStock'] >= frame['BB_UPPER']
    frame['BB_TOUCH_LOWER'] = frame['closeValueStock'] <= frame['BB_LOWER']
    return frame

def obter_resumo_horizonte(frame):
    ultimo = frame.iloc[-1]
    return {
        'Preço': float(ultimo['closeValueStock']),
        'RSI': float(ultimo['RSI_14']) if pd.notna(ultimo['RSI_14']) else np.nan,
        'MACD': float(ultimo['MACD']) if pd.notna(ultimo['MACD']) else np.nan,
        'OBV': float(ultimo['OBV']) if pd.notna(ultimo['OBV']) else np.nan,
        'Volatilidade': float(ultimo['Volatilidade']) if pd.notna(ultimo['Volatilidade']) else np.nan,
    }

# Calcular indicadores para diário, semanal e mensal
timeframes = {
    'Diário': enriquecer_indicadores(df_viz),
    'Semanal': enriquecer_indicadores(preparar_timeframe(df, 'W-FRI').tail(156).copy()),
    'Mensal': enriquecer_indicadores(preparar_timeframe(df, 'ME').tail(120).copy()),
}

df_viz = timeframes['Diário']

tab_diario, tab_semanal, tab_mensal = st.tabs(["Diário", "Semanal", "Mensal"])


def render_timeframe_dashboard(frame, label):
    st.subheader(f"📊 {label}: preço, MACD, OBV e Bollinger")

    with st.expander("Legenda rapida dos indicadores", expanded=False):
        st.markdown("""
        **Preco**: valor de fechamento da acao no periodo selecionado.
        **SMA 20/50/200**: medias moveis de curto, medio e longo prazo; ajudam a ver a tendencia.
        **MACD**: compara medias moveis. Acima da linha de sinal sugere forca compradora; abaixo sugere perda de forca.
        **Histograma MACD**: distancia entre MACD e linha de sinal. Barras maiores indicam diferenca maior entre as duas linhas.
        **OBV**: soma volume em dias de alta e subtrai em dias de queda. Se sobe junto com o preco, o volume confirma o movimento.
        **Bandas de Bollinger**: faixa em torno da media de 20 periodos. Perto da superior pode estar esticado para cima; perto da inferior, esticado para baixo.
        **RSI 14**: mede forca recente. Acima de 70 costuma indicar sobrecompra; abaixo de 30, sobrevenda.
        """)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Preço", f"R$ {frame['closeValueStock'].iloc[-1]:.2f}")
    with col_b:
        rsi_atual = frame['RSI_14'].iloc[-1]
        st.metric("RSI 14", f"{rsi_atual:.1f}" if pd.notna(rsi_atual) else "n/d")
    with col_c:
        macd_atual = frame['MACD'].iloc[-1]
        st.metric("MACD", f"{macd_atual:.3f}" if pd.notna(macd_atual) else "n/d")
    with col_d:
        vol_atual = frame['Volatilidade'].iloc[-1]
        st.metric("Volatilidade", f"{vol_atual:.1f}%" if pd.notna(vol_atual) else "n/d")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.42, 0.18, 0.2, 0.2],
        specs=[[{}], [{}], [{}], [{}]]
    )

    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['closeValueStock'], name='Preço', line=dict(color='black', width=2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['SMA_50'], name='SMA 50', line=dict(color='blue', dash='dash')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['SMA_200'], name='SMA 200', line=dict(color='red', dash='dash')
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['BB_UPPER'], name='BB Superior', line=dict(color='#999999', width=1)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['BB_LOWER'], name='BB Inferior', line=dict(color='#999999', width=1), fill='tonexty', fillcolor='rgba(120,120,120,0.08)'
    ), row=1, col=1)

    macd_up = frame[frame['MACD_CROSS_UP']]
    macd_down = frame[frame['MACD_CROSS_DOWN']]
    bb_upper = frame[frame['BB_TOUCH_UPPER']]
    bb_lower = frame[frame['BB_TOUCH_LOWER']]

    if not macd_up.empty:
        fig.add_trace(go.Scatter(
            x=macd_up['dateStock'], y=macd_up['closeValueStock'], mode='markers', name='MACD cruzou para alta',
            marker=dict(symbol='triangle-up', size=11, color='#2ecc71', line=dict(color='white', width=1))
        ), row=1, col=1)
    if not macd_down.empty:
        fig.add_trace(go.Scatter(
            x=macd_down['dateStock'], y=macd_down['closeValueStock'], mode='markers', name='MACD cruzou para baixa',
            marker=dict(symbol='triangle-down', size=11, color='#e74c3c', line=dict(color='white', width=1))
        ), row=1, col=1)
    if not bb_upper.empty:
        fig.add_trace(go.Scatter(
            x=bb_upper['dateStock'], y=bb_upper['closeValueStock'], mode='markers', name='Toque banda superior',
            marker=dict(symbol='circle-open', size=10, color='#f39c12', line=dict(color='#f39c12', width=2))
        ), row=1, col=1)
    if not bb_lower.empty:
        fig.add_trace(go.Scatter(
            x=bb_lower['dateStock'], y=bb_lower['closeValueStock'], mode='markers', name='Toque banda inferior',
            marker=dict(symbol='circle-open', size=10, color='#3498db', line=dict(color='#3498db', width=2))
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=frame['dateStock'], y=frame['MACD_HIST'], name='Histograma MACD', marker_color='#8e44ad'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['MACD'], name='MACD', line=dict(color='#16a085')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['MACD_SIGNAL'], name='Sinal MACD', line=dict(color='#c0392b')
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['OBV'], name='OBV', line=dict(color='#2c3e50')
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=frame['dateStock'], y=frame['RSI_14'], name='RSI 14', line=dict(color='green'), fill='tozeroy'
    ), row=4, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Sobrecompra', row=4, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='blue', annotation_text='Sobrevenda', row=4, col=1)

    fig.update_yaxes(title_text='Preço', row=1, col=1)
    fig.update_yaxes(title_text='MACD', row=2, col=1)
    fig.update_yaxes(title_text='OBV', row=3, col=1)
    fig.update_yaxes(title_text='RSI', row=4, col=1, range=[0, 100])
    fig.update_layout(title=f"{stock_code} - {label}", hovermode='x unified', height=900, legend_orientation='h')

    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ultimo = frame.iloc[-1]
        acima_bb = ultimo['closeValueStock'] > ultimo['BB_UPPER'] if pd.notna(ultimo['BB_UPPER']) else False
        abaixo_bb = ultimo['closeValueStock'] < ultimo['BB_LOWER'] if pd.notna(ultimo['BB_LOWER']) else False
        if acima_bb:
            st.warning("Preço acima da banda superior")
        elif abaixo_bb:
            st.info("Preço abaixo da banda inferior")
        else:
            st.success("Preço dentro das bandas")
    with col2:
        macd_atual = frame['MACD'].iloc[-1]
        sinal_atual = frame['MACD_SIGNAL'].iloc[-1]
        if pd.notna(macd_atual) and pd.notna(sinal_atual):
            if macd_atual > sinal_atual:
                st.success("MACD acima da linha de sinal")
            else:
                st.warning("MACD abaixo da linha de sinal")
    with col3:
        obv_delta = frame['OBV'].diff().iloc[-1]
        if pd.notna(obv_delta) and obv_delta > 0:
            st.success("OBV em alta: volume confirma movimento")
        else:
            st.info("OBV estável ou em queda")

    recent_macd_up = frame[frame['MACD_CROSS_UP']].tail(1)
    recent_macd_down = frame[frame['MACD_CROSS_DOWN']].tail(1)
    recent_bb_lower = frame[frame['BB_TOUCH_LOWER']].tail(1)
    recent_bb_upper = frame[frame['BB_TOUCH_UPPER']].tail(1)

    st.markdown("#### Alertas Visuais")
    alert_cols = st.columns(4)
    with alert_cols[0]:
        if not recent_macd_up.empty:
            data_alerta = recent_macd_up['dateStock'].iloc[-1].strftime('%Y-%m-%d')
            st.success(f"Cruzamento de alta no MACD em {data_alerta}")
        else:
            st.info("Sem cruzamento de alta recente")
    with alert_cols[1]:
        if not recent_macd_down.empty:
            data_alerta = recent_macd_down['dateStock'].iloc[-1].strftime('%Y-%m-%d')
            st.warning(f"Cruzamento de baixa no MACD em {data_alerta}")
        else:
            st.info("Sem cruzamento de baixa recente")
    with alert_cols[2]:
        if not recent_bb_lower.empty:
            data_alerta = recent_bb_lower['dateStock'].iloc[-1].strftime('%Y-%m-%d')
            st.success(f"Toque na banda inferior em {data_alerta}")
        else:
            st.info("Sem toque recente na banda inferior")
    with alert_cols[3]:
        if not recent_bb_upper.empty:
            data_alerta = recent_bb_upper['dateStock'].iloc[-1].strftime('%Y-%m-%d')
            st.warning(f"Toque na banda superior em {data_alerta}")
        else:
            st.info("Sem toque recente na banda superior")


with tab_diario:
    render_timeframe_dashboard(timeframes['Diário'], 'Diário')

with tab_semanal:
    render_timeframe_dashboard(timeframes['Semanal'], 'Semanal')

with tab_mensal:
    render_timeframe_dashboard(timeframes['Mensal'], 'Mensal')

# ==================== TABELA RESUMO ====================

st.subheader("📋 Últimos 10 Dias - Resumo")

df_resumo = df_viz[['dateStock', 'closeValueStock', 'SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'Volatilidade', 'MACD', 'MACD_SIGNAL', 'OBV']].tail(10).copy()
df_resumo.columns = ['Data', 'Preço (R$)', 'SMA 20', 'SMA 50', 'RSI 14', 'ATR 14', 'Vol Anualizada (%)', 'MACD', 'MACD Sinal', 'OBV']
df_resumo['Data'] = df_resumo['Data'].dt.strftime('%Y-%m-%d')

st.dataframe(df_resumo.style.format({
    'Preço (R$)': '{:.2f}',
    'SMA 20': '{:.2f}',
    'SMA 50': '{:.2f}',
    'RSI 14': '{:.1f}',
    'ATR 14': '{:.2f}',
    'Vol Anualizada (%)': '{:.1f}',
    'MACD': '{:.3f}',
    'MACD Sinal': '{:.3f}',
    'OBV': '{:.0f}',
}), use_container_width=True)

# ==================== INTERPRETAÇÃO GERAL ====================

st.subheader("📖 Interpretação dos Indicadores")

interpretation = """
**SMA (Simple Moving Average)**
- **SMA 20**: Tendência de curto prazo
- **SMA 50**: Tendência de médio prazo  
- **SMA 200**: Tendência de longo prazo
- **Sinal**: Quando SMA 20 > SMA 50 > SMA 200 → Tendência ALTA
- **Sinal**: Quando SMA 20 < SMA 50 < SMA 200 → Tendência BAIXA

**RSI (Relative Strength Index)**
- **RSI > 70**: Ação em SOBRECOMPRA (possível queda)
- **RSI < 30**: Ação em SOBREVENDA (possível alta)
- **RSI 30-70**: Normal, sem sinais extremos
- **Divergências**: RSI sobe enquanto preço cai = possível reversão

**ATR (Average True Range)**
- **ATR alto**: Volatilidade alta (maior risco/oportunidade)
- **ATR baixo**: Volatilidade baixa (movimento previsível)
- **Uso**: Define níveis de stop-loss e take-profit

**Volatilidade Anualizada**
- Expressa em %, indica quanto o preço pode variar em 1 ano
- **Volatilidade alta (>60%)**: Ação volátil, maior risco
- **Volatilidade baixa (<30%)**: Ação estável, menor risco
"""

interpretation += """

**MACD**
- **MACD acima do sinal**: Momento positivo no curto prazo
- **MACD abaixo do sinal**: Momento mais fraco ou negativo
- **Cruzamentos**: Podem marcar mudanca de tendencia, mas precisam de confirmacao

**OBV (On Balance Volume)**
- **OBV subindo**: Volume acompanha dias de alta
- **OBV caindo**: Volume pesa mais em dias de queda
- **Divergencia**: Preco sobe e OBV cai pode indicar alta sem apoio do volume

**Bandas de Bollinger**
- **Banda superior**: Preco esticado para cima em relacao a media recente
- **Banda inferior**: Preco esticado para baixo em relacao a media recente
- **Bandas abertas**: Volatilidade maior; **bandas fechadas**: volatilidade menor
"""

st.markdown(interpretation)

# ==================== COMBINAÇÃO DE SINAIS ====================

st.subheader("🎯 Combinação de Sinais")

sinais = []

# Sinal 1: Tendência SMA
if df_viz['SMA_20'].iloc[-1] > df_viz['SMA_50'].iloc[-1] > df_viz['SMA_200'].iloc[-1]:
    sinais.append("✅ Tendência ALTA (SMA 20 > 50 > 200)")
elif df_viz['SMA_20'].iloc[-1] < df_viz['SMA_50'].iloc[-1] < df_viz['SMA_200'].iloc[-1]:
    sinais.append("❌ Tendência BAIXA (SMA 20 < 50 < 200)")
else:
    sinais.append("⚪ Tendência MISTA (SMAs cruzadas)")

# Sinal 2: RSI
if df_viz['RSI_14'].iloc[-1] > 70:
    sinais.append("🔴 RSI Sobrecompra (>70) - Possível queda")
elif df_viz['RSI_14'].iloc[-1] < 30:
    sinais.append("🟢 RSI Sobrevenda (<30) - Possível alta")

# Sinal 3: Preço vs SMA 200
if df_viz['closeValueStock'].iloc[-1] > df_viz['SMA_200'].iloc[-1]:
    sinais.append("📈 Preço acima da SMA 200 (suporte de longo prazo)")
else:
    sinais.append("📉 Preço abaixo da SMA 200 (sem suporte)")

# Mostrar sinais
for sinal in sinais:
    st.write(sinal)
