# Radar B3

Dashboard em Streamlit para acompanhar ações da B3 com foco em triagem de oportunidades por curto, médio e longo prazo.

O sistema combina dados históricos, indicadores técnicos, backtests e scores de risco para ajudar na leitura de possíveis candidatos de compra ou venda. Ele é um projeto educacional e não substitui análise profissional de investimentos.

## Funcionalidades

- Candidatos de compra por prazo: curto, médio e longo.
- Recomendações para o próximo pregão com probabilidade de alta/queda.
- Radar de swing trade com distribuição sugerida de capital.
- Simulação histórica de entradas e saídas.
- Painel detalhado por ação com indicadores técnicos, cotação recente e fundamentos quando disponíveis.
- Backtests e métricas de desempenho do modelo.

## Como Rodar Localmente

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Para atualizar dados, backtests e previsões antes de abrir o dashboard:

```bash
python run_update_and_open.py
```

## Deploy no Streamlit Cloud

1. Suba este projeto para o GitHub.
2. Acesse https://streamlit.io/cloud.
3. Crie um novo app usando o repositório `mariobignami/Radar-B3`.
4. Defina o arquivo principal como `dashboard.py`.
5. Faça o deploy.

## Aviso

Este projeto é apenas educacional. As previsões são baseadas em dados históricos e podem estar erradas. O passado não garante desempenho futuro. Consulte um profissional antes de investir.
