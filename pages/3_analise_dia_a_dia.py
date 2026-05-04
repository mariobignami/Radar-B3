"""
Dashboard Web para Análise Dia-a-Dia
Mostra predição do dia anterior vs resultado real do dia atual
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Análise Dia-a-Dia",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análise Dia-a-Dia")
st.write("Compare predições vs resultados reais em pares de dias consecutivos")

# ⚠️ DISCLAIMER IMPORTANTE
st.error("""
🚨 **DISCLAIMER CRÍTICO - LEIA COM ATENÇÃO**

Este é um **PROJETO EDUCACIONAL** apenas. NÃO é recomendação de investimento.

**Limitações Importantes:**
- ✗ Modelo treina com dados HISTÓRICOS (passado já aconteceu)
- ✗ Futuro pode ser COMPLETAMENTE diferente do passado
- ✗ Não prevê: notícias, crises, eventos econômicos, decisões políticas
- ✗ Modelo é LINEAR (muito simples) e pode falhar em crises
- ✗ 99%+ de acurácia NO PASSADO ≠ 99%+ no futuro

**Riscos Reais:**
- ⚠️ Você pode PERDER dinheiro se seguir essas recomendações
- ⚠️ Volatilidade extrema pode quebrar o modelo completamente
- ⚠️ Uma notícia ruim e tudo muda

**Antes de investir:**
- 🔸 Consulte um profissional (consultor, analista)
- 🔸 Faça sua própria análise fundamentalista
- 🔸 Teste com valores pequenos primeiro
- 🔸 Nunca coloque todo seu dinheiro em uma predição

**Fim do Disclaimer** ⚠️
""")

st.info("""
📌 **Como funciona:**
- **Preço Real**: Preço de FECHAMENTO (close) da ação ao final do dia (dados históricos)
- **Preço Predito**: Preço de FECHAMENTO predito pelo modelo para aquele dia
- **Erro**: Diferença absoluta em reais entre predição e realidade
- **Acurácia**: Calculada a partir do SMAPE (0 a 100, maior = melhor)
""")

# Carregar dados do backtest
backtest_file = Path('backtest_results.json')
if not backtest_file.exists():
    st.error("❌ Arquivo backtest_results.json não encontrado. Execute backtest_model.py primeiro.")
    st.stop()

with open(backtest_file) as f:
    backtest_data = json.load(f)

if not backtest_data:
    st.error("❌ Nenhum dado de backtest disponível.")
    st.stop()

df_backtest = pd.DataFrame(backtest_data)

# Converter para DataFrame estruturado
df_backtest['date'] = pd.to_datetime(df_backtest['date'])
df_backtest = df_backtest.sort_values('date')

st.sidebar.header("Filtros")

# Selecionar intervalo de datas
min_date = df_backtest['date'].min()
max_date = df_backtest['date'].max()

date_range = st.sidebar.date_input(
    "Selecione o intervalo",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    df_filtered = df_backtest[
        (df_backtest['date'].dt.date >= date_range[0]) &
        (df_backtest['date'].dt.date <= date_range[1])
    ]
else:
    df_filtered = df_backtest

# Selecionar empresa
companies = sorted(df_filtered['company'].unique())
selected_companies = st.sidebar.multiselect(
    "Selecione as empresas",
    companies,
    default=companies
)

if selected_companies:
    df_filtered = df_filtered[df_filtered['company'].isin(selected_companies)]

# ========== MÉTRICAS PRINCIPAIS ==========
st.header("📈 Métricas Gerais")

col1, col2, col3, col4 = st.columns(4)

with col1:
    mean_accuracy = df_filtered['accuracy'].mean()
    st.metric("Acurácia Média", f"{mean_accuracy:.2f}%")

with col2:
    min_accuracy = df_filtered['accuracy'].min()
    st.metric("Acurácia Mínima", f"{min_accuracy:.2f}%")

with col3:
    max_accuracy = df_filtered['accuracy'].max()
    st.metric("Acurácia Máxima", f"{max_accuracy:.2f}%")

with col4:
    mean_error = df_filtered['error'].mean()
    st.metric("Erro Médio (R$)", f"{mean_error:.2f}")

# ========== GRÁFICOS ==========
st.header("📊 Visualizações")

# Gráfico 1: Acurácia por Data
st.subheader("Acurácia ao longo do tempo")

df_by_date = df_filtered.groupby('date')['accuracy'].mean().reset_index()
df_by_date['date_str'] = df_by_date['date'].dt.strftime('%d/%m/%Y')

fig1 = px.line(
    df_by_date,
    x='date',
    y='accuracy',
    markers=True,
    labels={'date': 'Data', 'accuracy': 'Acurácia (%)'},
    title='Acurácia Média por Dia'
)
fig1.update_layout(height=400)
st.plotly_chart(fig1, use_container_width=True)

# Gráfico 2: Acurácia por Empresa
st.subheader("Acurácia por Empresa")

df_by_company = df_filtered.groupby('company')['accuracy'].mean().sort_values(ascending=False)

fig2 = px.bar(
    x=df_by_company.index,
    y=df_by_company.values,
    labels={'x': 'Empresa', 'y': 'Acurácia (%)'},
    title='Acurácia Média por Empresa',
    color=df_by_company.values,
    color_continuous_scale='Viridis'
)
fig2.update_layout(height=400)
st.plotly_chart(fig2, use_container_width=True)

# ========== TABELA DE RESULTADOS ==========
st.header("📋 Resultados Detalhados")

# Preparar dados para exibição
df_display = df_filtered.copy()
df_display['date_str'] = df_display['date'].dt.strftime('%d/%m/%Y')
df_display = df_display[[
    'date_str', 'company', 'real', 'predicted', 
    'error', 'accuracy'
]].rename(columns={
    'date_str': 'Data',
    'company': 'Empresa',
    'real': 'Fechamento Real (R$)',
    'predicted': 'Fechamento Predito (R$)',
    'error': 'Diferença (R$)',
    'accuracy': 'Acurácia (%)'
})

# Formatar números
df_display['Fechamento Real (R$)'] = df_display['Fechamento Real (R$)'].apply(lambda x: f'{x:.2f}')
df_display['Fechamento Predito (R$)'] = df_display['Fechamento Predito (R$)'].apply(lambda x: f'{x:.2f}')
df_display['Diferença (R$)'] = df_display['Diferença (R$)'].apply(lambda x: f'{x:.2f}')
df_display['Acurácia (%)'] = df_display['Acurácia (%)'].apply(lambda x: f'{x:.2f}%')

st.dataframe(df_display, use_container_width=True, hide_index=True)

# ========== ANÁLISE ESPECÍFICA ==========
st.header("🔍 Análise Específica")

dates = sorted(df_filtered['date'].unique())
if len(dates) > 0:
    selected_date = st.selectbox(
        "Selecione uma data para análise detalhada",
        dates,
        format_func=lambda x: x.strftime('%d/%m/%Y')
    )
    
    df_date_detail = df_filtered[df_filtered['date'] == selected_date].copy()
    
    st.subheader(f"📅 Análise para {selected_date.strftime('%d de %B de %Y')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Data", selected_date.strftime('%d/%m/%Y'))
        st.metric("Acurácia Média", f"{df_date_detail['accuracy'].mean():.2f}%")
    
    with col2:
        st.metric("Predições", len(df_date_detail))
        st.metric("Erro Médio", f"R$ {df_date_detail['error'].mean():.2f}")
    
    st.write("**Detalhes por Ação:**")
    
    for idx, row in df_date_detail.iterrows():
        with st.expander(f"{row['company']} - Acurácia: {row['accuracy']:.2f}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fechamento Real", f"R$ {row['real']:.2f}")
            with col2:
                st.metric("Fechamento Predito", f"R$ {row['predicted']:.2f}")
            with col3:
                st.metric("Diferença", f"R$ {row['error']:.2f}")
            
            # Barra de acurácia visual
            progress_value = min(row['accuracy'] / 100, 1.0)
            st.progress(progress_value)

# ========== INSIGHTS ==========
st.header("💡 Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Melhores Performances")
    top_5 = df_filtered.nlargest(5, 'accuracy')[['date', 'company', 'accuracy']].copy()
    top_5['date'] = top_5['date'].dt.strftime('%d/%m/%Y')
    st.table(top_5.rename(columns={'date': 'Data', 'company': 'Empresa', 'accuracy': 'Acurácia (%)'}))

with col2:
    st.subheader("Piores Performances")
    bottom_5 = df_filtered.nsmallest(5, 'accuracy')[['date', 'company', 'accuracy']].copy()
    bottom_5['date'] = bottom_5['date'].dt.strftime('%d/%m/%Y')
    st.table(bottom_5.rename(columns={'date': 'Data', 'company': 'Empresa', 'accuracy': 'Acurácia (%)'}))

st.write("---")
st.write("""
### Como Usar

1. **Filtros**: Use a barra lateral para filtrar por intervalo de datas e empresas
2. **Métricas**: Veja as estatísticas gerais no topo
3. **Gráficos**: Visualize tendências de acurácia ao longo do tempo
4. **Tabela**: Consulte todos os resultados em formato tabular
5. **Análise Específica**: Clique em uma data para ver detalhes de cada ação

### O que Significa?

- **Preço Real**: Preço de FECHAMENTO (close) da ação ao final do dia útil
- **Preço Predito**: O que nosso modelo predisse que seria o preço de fechamento
- **Acurácia**: Porcentagem de proximidade entre predição e preço real
- **Erro**: Diferença absoluta em reais (R$) entre predição e realidade
- **Meta**: Manter acurácia > 99% (muito próximo do preço real)

### Exemplos de Interpretação

- **Acurácia 99.62%**: Ótimo! Predição muito próxima do resultado real
- **Acurácia 98.50%**: Bom, mas com pequeno desvio
- **Erro R$ 0.18**: Diferença pequena entre predição (R$ 46.20) e real (R$ 46.38)
""")
