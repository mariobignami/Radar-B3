# 🎯 COMO USAR O DASHBOARD WEB

## 📲 Instalação Rápida

### Passo 1: Instalar Streamlit
```bash
pip install streamlit plotly -q
```
streamlit run dashboard.py
streamlit run das


### Passo 2: Rodar o Dashboard
```bash
streamlit run dashboard.py
```

### Passo 3: Acessar no Navegador
O dashboard abrirá automaticamente em:
```
http://localhost:8501
```

---

## 🎨 Funcionalidades do Dashboard

### 1️⃣ **Página: Predições** (🔮)
- Digite os parâmetros da ação:
  - Preço de Abertura
  - Preço Máximo
  - Preço Mínimo
  - Volume Negociado
  - Setor e Segmento
  - Data (mês e dia da semana)

- Clique em "🚀 Fazer Predição"
- Veja o resultado instantâneo com gráfico

**Exemplo:**
```
Abertura: R$ 28.50
Previsão: R$ 29.22 (+2.53%)
Gráfico: Mostra a evolução prevista
```

### 2️⃣ **Página: Backtest** (📊)
- Visualiza os resultados de 30 predições
- Gráficos interativos:
  - Acurácia ao longo do tempo
  - Performance por empresa
  - Tabela completa de resultados

**Métricas Mostradas:**
```
Acurácia Média:    99.46%
Melhor Acerto:     100.00%
Pior Acerto:       98.16%
Erro Médio:        R$ 0.21
```

### 3️⃣ **Página: Análise** (📈)
- Informações do modelo
- Performance geral
- Como funciona o sistema
- Features utilizadas

### 4️⃣ **Página: Sobre** (ℹ️)
- Objetivo do projeto
- Tecnologias utilizadas
- Dataset e período
- Instruções de uso
- Disclaimer e avisos

---

## 🖼️ Estrutura Visual

```
┌─────────────────────────────────────────────────────┐
│  📈 B3 Stock Prediction Dashboard                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│  [🔮 Predições] [📊 Backtest] [📈 Análise]        │
│  [ℹ️ Sobre]                                         │
│                                                     │
├─────────────────────────────────────────────────────┤
│  🔮 Faça uma Predição                              │
│                                                     │
│  Dados da Ação        Características              │
│  ┌──────────────┐    ┌────────────────┐           │
│  │Abertura: 28.5│    │Setor: Energia  │           │
│  │Máximo:  29.2 │    │Mês: Abril      │           │
│  │Mínimo:  28.1 │    │Dia: Seg        │           │
│  │Volume:  500M │    │                │           │
│  └──────────────┘    └────────────────┘           │
│                                                     │
│  [🚀 Fazer Predição]                              │
│                                                     │
│  Resultado:                                         │
│  ┌──────────────────────────────────────────┐     │
│  │ Abertura: R$ 28.50                       │     │
│  │ Fechamento Previsto: R$ 29.22 (+2.53%)   │     │
│  │ Intervalo: R$ 1.10                       │     │
│  └──────────────────────────────────────────┘     │
│                                                     │
│  [Gráfico interativo mostrando a predição]        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🔧 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit -q
```

### Erro: "ModuleNotFoundError: No module named 'plotly'"
```bash
pip install plotly -q
```

### Dashboard não abre no navegador
Abra manualmente em:
```
http://localhost:8501
```

### Porta 8501 já está em uso
```bash
streamlit run dashboard.py --server.port 8502
```

---

## 📊 Dados do Backtest

O arquivo `backtest_results.json` contém:
- Datas testadas
- Empresas avaliadas
- Preços reais vs preditos
- Erros e acurácias

Estrutura:
```json
[
  {
    "date": "2026-04-28",
    "company": "PETR4",
    "real": 47.52,
    "predicted": 47.86,
    "error": 0.34,
    "accuracy": 99.3
  },
  ...
]
```

---

## 💡 Dicas de Uso

1. **Comparar Empresas**
   - Vá para a aba "Backtest"
   - Veja qual ação teve melhor performance
   - ABEV3 teve 99.7% de acurácia média

2. **Entender o Modelo**
   - Acesse a aba "Análise"
   - Leia sobre features e performance
   - Veja como funciona a predição

3. **Fazer Predições**
   - Use valores realistas (preços atuais)
   - Preencha todos os campos
   - Clique no botão de predição

4. **Validar Resultados**
   - Compare com dados históricos
   - Veja as métricas de performance
   - Analise as tendências

---

## 🚀 Próximas Features

- [ ] Integração com dados em tempo real (WebSocket)
- [ ] Gráficos de série temporal (LSTM)
- [ ] API REST para integração
- [ ] Alertas de anomalias
- [ ] Relatórios em PDF
- [ ] Exportar predições em CSV

---

**Dashboard Status:** ✅ Ativo e Testado  
**Última Atualização:** 29 de Abril de 2026
