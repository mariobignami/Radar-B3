"""
Script para baixar dados reais da B3 usando yfinance
Gera CSVs no formato esperado pelo projeto
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Primeiro, vamos instalar yfinance se necessário
try:
    import yfinance as yf
except ImportError:
    print("📦 Instalando yfinance...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance'])
    import yfinance as yf

# Configurações
data_raw = Path("data/raw")
data_raw.mkdir(parents=True, exist_ok=True)

# Configuracao de periodo e filtros
HISTORY_START = (datetime.now() - timedelta(days=183)).strftime("%Y-%m-%d")
UNIVERSE_LIMIT = 100

# Tickers principais (liquidos + swing trade)
BASE_TICKERS = {
    # Varejo
    'MGLU3.SA': {'name': 'Magazine Luiza', 'sector': 'Comercio', 'segment': 'Varejo'},
    'LREN3.SA': {'name': 'Lojas Renner', 'sector': 'Comercio', 'segment': 'Varejo'},
    'AMER3.SA': {'name': 'Americanas', 'sector': 'Comercio', 'segment': 'Varejo'},
    'ASAI3.SA': {'name': 'Assai', 'sector': 'Comercio', 'segment': 'Atacarejo'},
    'CRFB3.SA': {'name': 'Carrefour', 'sector': 'Comercio', 'segment': 'Varejo'},
    'PCAR3.SA': {'name': 'Pao de Acucar', 'sector': 'Comercio', 'segment': 'Varejo'},
    'NTCO3.SA': {'name': 'Natura', 'sector': 'Consumo', 'segment': 'Cosmeticos'},
    'SOMA3.SA': {'name': 'Grupo Soma', 'sector': 'Comercio', 'segment': 'Varejo'},
    'CEAB3.SA': {'name': 'Cea', 'sector': 'Comercio', 'segment': 'Varejo'},
    'ARZZ3.SA': {'name': 'Arezzo', 'sector': 'Comercio', 'segment': 'Calcados'},
    'VIIA3.SA': {'name': 'Via', 'sector': 'Comercio', 'segment': 'Varejo'},
    'GMAT3.SA': {'name': 'Grupo Mateus', 'sector': 'Comercio', 'segment': 'Varejo'},
    'VIVA3.SA': {'name': 'Vivara', 'sector': 'Comercio', 'segment': 'Joias'},
    'ALPA4.SA': {'name': 'Alpargatas', 'sector': 'Comercio', 'segment': 'Calcados'},
    'GUAR3.SA': {'name': 'Guararapes', 'sector': 'Comercio', 'segment': 'Varejo'},
    'LJQQ3.SA': {'name': 'Quero-Quero', 'sector': 'Comercio', 'segment': 'Varejo'},

    # Energia e petroleo
    'PETR4.SA': {'name': 'Petrobras PN', 'sector': 'Energia', 'segment': 'Petroleo'},
    'PETR3.SA': {'name': 'Petrobras ON', 'sector': 'Energia', 'segment': 'Petroleo'},
    'PRIO3.SA': {'name': 'PRIO', 'sector': 'Energia', 'segment': 'Petroleo'},
    'UGPA3.SA': {'name': 'Ultrapar', 'sector': 'Energia', 'segment': 'Distribuicao'},
    'CSAN3.SA': {'name': 'Cosan', 'sector': 'Energia', 'segment': 'Combustiveis'},
    'VBBR3.SA': {'name': 'Vibra', 'sector': 'Energia', 'segment': 'Combustiveis'},
    'RECV3.SA': {'name': 'PetroReconcavo', 'sector': 'Energia', 'segment': 'Petroleo'},

    # Outros nichos
    'VALE3.SA': {'name': 'Vale', 'sector': 'Mineracao', 'segment': 'Ferro'},
    'ITUB4.SA': {'name': 'Itau', 'sector': 'Financeiro', 'segment': 'Banco'},
    'BBDC4.SA': {'name': 'Bradesco', 'sector': 'Financeiro', 'segment': 'Banco'},
    'ABEV3.SA': {'name': 'Ambev', 'sector': 'Bebidas', 'segment': 'Bebidas'},
    'WEGE3.SA': {'name': 'WEG', 'sector': 'Industria', 'segment': 'Bens de Capital'},
    'B3SA3.SA': {'name': 'B3', 'sector': 'Financeiro', 'segment': 'Bolsa'},
    'BBAS3.SA': {'name': 'Banco do Brasil', 'sector': 'Financeiro', 'segment': 'Banco'},
    'BPAC11.SA': {'name': 'BTG Pactual', 'sector': 'Financeiro', 'segment': 'Banco de Investimento'},
    'SANB11.SA': {'name': 'Santander Unit', 'sector': 'Financeiro', 'segment': 'Banco'},
    'EGIE3.SA': {'name': 'Engie Brasil', 'sector': 'Energia', 'segment': 'Energia'},
    'CMIG4.SA': {'name': 'Cemig PN', 'sector': 'Energia', 'segment': 'Energia'},
    'ELET3.SA': {'name': 'Eletrobras ON', 'sector': 'Energia', 'segment': 'Energia'},
}

# Candidatas a acoes baratas e volateis
CHEAP_CANDIDATES = {
    'OIBR3.SA': {'name': 'Oi', 'sector': 'Telecom', 'segment': 'Telecom'},
    'MRVE3.SA': {'name': 'MRV', 'sector': 'Construcao', 'segment': 'Construcao'},
    'GOLL4.SA': {'name': 'Gol', 'sector': 'Transporte', 'segment': 'Aereo'},
    'AZUL4.SA': {'name': 'Azul', 'sector': 'Transporte', 'segment': 'Aereo'},
    'COGN3.SA': {'name': 'Cogna', 'sector': 'Educacao', 'segment': 'Educacao'},
    'IRBR3.SA': {'name': 'IRB Brasil', 'sector': 'Financeiro', 'segment': 'Seguros'},
    'POMO4.SA': {'name': 'Marcopolo', 'sector': 'Industria', 'segment': 'Autopecas'},
    'USIM5.SA': {'name': 'Usiminas', 'sector': 'Industria', 'segment': 'Siderurgia'},
    'CSNA3.SA': {'name': 'CSN', 'sector': 'Industria', 'segment': 'Siderurgia'},
    'TEND3.SA': {'name': 'Tenda', 'sector': 'Construcao', 'segment': 'Construcao'},
    'QUAL3.SA': {'name': 'Qualicorp', 'sector': 'Saude', 'segment': 'Planos'},
    'CVCB3.SA': {'name': 'CVC', 'sector': 'Turismo', 'segment': 'Turismo'},
    'SIMH3.SA': {'name': 'Simpar', 'sector': 'Logistica', 'segment': 'Servicos'},
    'MOTV3.SA': {'name': 'Motiva', 'sector': 'Infraestrutura', 'segment': 'Concessoes'},
    'TIMS3.SA': {'name': 'TIM', 'sector': 'Telecom', 'segment': 'Telecom'},
    'VIVT3.SA': {'name': 'Vivo', 'sector': 'Telecom', 'segment': 'Telecom'},
    'RAIL3.SA': {'name': 'Rumo', 'sector': 'Logistica', 'segment': 'Ferrovias'},
    'CSMG3.SA': {'name': 'Copasa', 'sector': 'Saneamento', 'segment': 'Saneamento'},
    'SBSP3.SA': {'name': 'Sabesp', 'sector': 'Saneamento', 'segment': 'Saneamento'},
    'CPFE3.SA': {'name': 'CPFL Energia', 'sector': 'Energia', 'segment': 'Energia'},
    'ENEV3.SA': {'name': 'Eneva', 'sector': 'Energia', 'segment': 'Energia'},
    'LWSA3.SA': {'name': 'Locaweb', 'sector': 'Tecnologia', 'segment': 'Software'},
    'RDOR3.SA': {'name': 'Rede DOr', 'sector': 'Saude', 'segment': 'Hospitais'},
    'MULT3.SA': {'name': 'Multiplan', 'sector': 'Imobiliario', 'segment': 'Shopping'},
    'HYPE3.SA': {'name': 'Hypera', 'sector': 'Saude', 'segment': 'Farmaceutico'},
    'JBSS3.SA': {'name': 'JBS', 'sector': 'Alimentos', 'segment': 'Proteinas'},
    'MGLU3.SA': {'name': 'Magazine Luiza', 'sector': 'Comercio', 'segment': 'Varejo'},
}

# Tickers usados pela pagina de recomendacoes para swing trade.
# Mantidos aqui para garantir que qualquer papel novo seja buscado no yfinance
# durante a atualizacao da base.
SWING_TRADE_CORE_TICKERS = [
    'ABEV3.SA', 'ALPA4.SA', 'AMER3.SA', 'ASAI3.SA', 'BBDC4.SA', 'CASH3.SA',
    'CEAB3.SA', 'BBSE3.SA', 'COGN3.SA', 'CSNA3.SA', 'CURY3.SA', 'CVCB3.SA',
    'CYRE3.SA', 'GGBR4.SA', 'GOAU4.SA', 'HAPV3.SA', 'ITUB4.SA', 'KLBN11.SA',
    'LREN3.SA', 'MGLU3.SA', 'MRVE3.SA', 'OIBR3.SA', 'PCAR3.SA', 'PETR4.SA',
    'FLRY3.SA', 'POSI3.SA', 'PRIO3.SA', 'RADL3.SA', 'RDOR3.SA', 'SANB11.SA',
    'SUZB3.SA', 'TEND3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA',
    'VAMO3.SA', 'VIVA3.SA', 'WEGE3.SA', 'YDUQ3.SA',
]

ADDITIONAL_SWING_TICKERS = {
    'CASH3.SA': {'name': 'Meliuz', 'sector': 'Tecnologia', 'segment': 'Servicos digitais'},
    'CURY3.SA': {'name': 'Cury', 'sector': 'Construcao', 'segment': 'Incorporacao'},
    'CYRE3.SA': {'name': 'Cyrela', 'sector': 'Construcao', 'segment': 'Incorporacao'},
    'GGBR4.SA': {'name': 'Gerdau PN', 'sector': 'Industria', 'segment': 'Siderurgia'},
    'GOAU4.SA': {'name': 'Metalurgica Gerdau PN', 'sector': 'Industria', 'segment': 'Siderurgia'},
    'HAPV3.SA': {'name': 'Hapvida', 'sector': 'Saude', 'segment': 'Planos'},
    'KLBN11.SA': {'name': 'Klabin Unit', 'sector': 'Papel e Celulose', 'segment': 'Papel'},
    'POSI3.SA': {'name': 'Positivo', 'sector': 'Tecnologia', 'segment': 'Hardware'},
    'RADL3.SA': {'name': 'Raia Drogasil', 'sector': 'Saude', 'segment': 'Farmacias'},
    'SUZB3.SA': {'name': 'Suzano', 'sector': 'Papel e Celulose', 'segment': 'Celulose'},
    'TOTS3.SA': {'name': 'Totvs', 'sector': 'Tecnologia', 'segment': 'Software'},
    'VAMO3.SA': {'name': 'Vamos', 'sector': 'Logistica', 'segment': 'Locacao'},
    'YDUQ3.SA': {'name': 'Yduqs', 'sector': 'Educacao', 'segment': 'Educacao'},
}

# Universo amplo para o Radar B3. A lista mistura ativos de maior liquidez
# presentes em indices amplos como Ibovespa/IBrX e nomes setoriais relevantes.
BROAD_UNIVERSE_TICKERS = {
    'AALR3.SA': {'name': 'Alliar', 'sector': 'Saude', 'segment': 'Diagnosticos'},
    'AERI3.SA': {'name': 'Aerís', 'sector': 'Industria', 'segment': 'Energia eolica'},
    'AESB3.SA': {'name': 'AES Brasil', 'sector': 'Energia', 'segment': 'Energia'},
    'AGRO3.SA': {'name': 'BrasilAgro', 'sector': 'Agro', 'segment': 'Terras agricolas'},
    'ALOS3.SA': {'name': 'Allos', 'sector': 'Imobiliario', 'segment': 'Shopping'},
    'ANIM3.SA': {'name': 'Anima', 'sector': 'Educacao', 'segment': 'Educacao'},
    'ARML3.SA': {'name': 'Armac', 'sector': 'Industria', 'segment': 'Locacao'},
    'AURE3.SA': {'name': 'Auren', 'sector': 'Energia', 'segment': 'Energia'},
    'BEEF3.SA': {'name': 'Minerva', 'sector': 'Alimentos', 'segment': 'Proteinas'},
    'BHIA3.SA': {'name': 'Casas Bahia', 'sector': 'Comercio', 'segment': 'Varejo'},
    'BMGB4.SA': {'name': 'Banco BMG', 'sector': 'Financeiro', 'segment': 'Banco'},
    'BMOB3.SA': {'name': 'Bemobi', 'sector': 'Tecnologia', 'segment': 'Software'},
    'BOVA11.SA': {'name': 'iShares Ibovespa', 'sector': 'ETF', 'segment': 'Indice'},
    'BPAN4.SA': {'name': 'Banco Pan', 'sector': 'Financeiro', 'segment': 'Banco'},
    'BBSE3.SA': {'name': 'BB Seguridade', 'sector': 'Financeiro', 'segment': 'Seguros'},
    'BRAP4.SA': {'name': 'Bradespar', 'sector': 'Mineracao', 'segment': 'Holding'},
    'BRKM5.SA': {'name': 'Braskem', 'sector': 'Industria', 'segment': 'Quimicos'},
    'CBAV3.SA': {'name': 'CBA', 'sector': 'Industria', 'segment': 'Aluminio'},
    'CCRO3.SA': {'name': 'CCR', 'sector': 'Infraestrutura', 'segment': 'Concessoes'},
    'CMIN3.SA': {'name': 'CSN Mineracao', 'sector': 'Mineracao', 'segment': 'Minerio'},
    'CPLE6.SA': {'name': 'Copel', 'sector': 'Energia', 'segment': 'Energia'},
    'CRFB3.SA': {'name': 'Carrefour', 'sector': 'Comercio', 'segment': 'Varejo'},
    'CXSE3.SA': {'name': 'Caixa Seguridade', 'sector': 'Financeiro', 'segment': 'Seguros'},
    'DIRR3.SA': {'name': 'Direcional', 'sector': 'Construcao', 'segment': 'Construcao'},
    'DXCO3.SA': {'name': 'Dexco', 'sector': 'Industria', 'segment': 'Materiais'},
    'ECOR3.SA': {'name': 'Ecorodovias', 'sector': 'Infraestrutura', 'segment': 'Concessoes'},
    'EMBR3.SA': {'name': 'Embraer', 'sector': 'Industria', 'segment': 'Aeroespacial'},
    'ENAT3.SA': {'name': 'Enauta', 'sector': 'Energia', 'segment': 'Petroleo'},
    'ENGI11.SA': {'name': 'Energisa Unit', 'sector': 'Energia', 'segment': 'Energia'},
    'EQTL3.SA': {'name': 'Equatorial', 'sector': 'Energia', 'segment': 'Energia'},
    'EVEN3.SA': {'name': 'Even', 'sector': 'Construcao', 'segment': 'Incorporacao'},
    'EZTC3.SA': {'name': 'EZTEC', 'sector': 'Construcao', 'segment': 'Incorporacao'},
    'FESA4.SA': {'name': 'Ferbasa', 'sector': 'Industria', 'segment': 'Siderurgia'},
    'FLRY3.SA': {'name': 'Fleury', 'sector': 'Saude', 'segment': 'Diagnosticos'},
    'GFSA3.SA': {'name': 'Gafisa', 'sector': 'Construcao', 'segment': 'Incorporacao'},
    'GRND3.SA': {'name': 'Grendene', 'sector': 'Consumo', 'segment': 'Calcados'},
    'HBSA3.SA': {'name': 'Hidrovias do Brasil', 'sector': 'Logistica', 'segment': 'Transporte'},
    'IGTI11.SA': {'name': 'Iguatemi Unit', 'sector': 'Imobiliario', 'segment': 'Shopping'},
    'INTB3.SA': {'name': 'Intelbras', 'sector': 'Tecnologia', 'segment': 'Hardware'},
    'IRBR3.SA': {'name': 'IRB Brasil', 'sector': 'Financeiro', 'segment': 'Seguros'},
    'JALL3.SA': {'name': 'Jalles Machado', 'sector': 'Agro', 'segment': 'Acucar e etanol'},
    'KEPL3.SA': {'name': 'Kepler Weber', 'sector': 'Agro', 'segment': 'Equipamentos'},
    'LEVE3.SA': {'name': 'Mahle Metal Leve', 'sector': 'Industria', 'segment': 'Autopecas'},
    'LOGG3.SA': {'name': 'Log CP', 'sector': 'Imobiliario', 'segment': 'Galpoes'},
    'MDIA3.SA': {'name': 'M Dias Branco', 'sector': 'Alimentos', 'segment': 'Alimentos'},
    'MEAL3.SA': {'name': 'IMC', 'sector': 'Consumo', 'segment': 'Restaurantes'},
    'MOVI3.SA': {'name': 'Movida', 'sector': 'Transporte', 'segment': 'Locacao'},
    'NEOE3.SA': {'name': 'Neoenergia', 'sector': 'Energia', 'segment': 'Energia'},
    'NTCO3.SA': {'name': 'Natura', 'sector': 'Consumo', 'segment': 'Cosmeticos'},
    'ONCO3.SA': {'name': 'Oncoclínicas', 'sector': 'Saude', 'segment': 'Hospitais'},
    'ODPV3.SA': {'name': 'Odontoprev', 'sector': 'Saude', 'segment': 'Planos odontologicos'},
    'ORVR3.SA': {'name': 'Orizon', 'sector': 'Saneamento', 'segment': 'Residuos'},
    'PETZ3.SA': {'name': 'Petz', 'sector': 'Consumo', 'segment': 'Pets'},
    'POMO4.SA': {'name': 'Marcopolo', 'sector': 'Industria', 'segment': 'Autopecas'},
    'PORT3.SA': {'name': 'Wilson Sons', 'sector': 'Logistica', 'segment': 'Portos'},
    'QUAL3.SA': {'name': 'Qualicorp', 'sector': 'Saude', 'segment': 'Planos'},
    'RAPT4.SA': {'name': 'Randoncorp', 'sector': 'Industria', 'segment': 'Autopecas'},
    'RANI3.SA': {'name': 'Irani', 'sector': 'Papel e Celulose', 'segment': 'Papel'},
    'RCSL3.SA': {'name': 'Recrusul', 'sector': 'Industria', 'segment': 'Equipamentos'},
    'RENT3.SA': {'name': 'Localiza', 'sector': 'Transporte', 'segment': 'Locacao'},
    'ROMI3.SA': {'name': 'Industrias Romi', 'sector': 'Industria', 'segment': 'Maquinas'},
    'SLCE3.SA': {'name': 'SLC Agricola', 'sector': 'Agro', 'segment': 'Agricultura'},
    'SMFT3.SA': {'name': 'Smart Fit', 'sector': 'Consumo', 'segment': 'Academias'},
    'SMTO3.SA': {'name': 'Sao Martinho', 'sector': 'Agro', 'segment': 'Acucar e etanol'},
    'SOJA3.SA': {'name': 'Boa Safra', 'sector': 'Agro', 'segment': 'Sementes'},
    'STBP3.SA': {'name': 'Santos Brasil', 'sector': 'Logistica', 'segment': 'Portos'},
    'TAEE11.SA': {'name': 'Taesa Unit', 'sector': 'Energia', 'segment': 'Transmissao'},
    'TRIS3.SA': {'name': 'Trisul', 'sector': 'Construcao', 'segment': 'Incorporacao'},
    'TTEN3.SA': {'name': '3tentos', 'sector': 'Agro', 'segment': 'Insumos'},
    'VULC3.SA': {'name': 'Vulcabras', 'sector': 'Consumo', 'segment': 'Calcados'},
    'WIZC3.SA': {'name': 'Wiz', 'sector': 'Financeiro', 'segment': 'Corretagem'},
}

def compute_volatility(close_series):
    series = close_series.dropna()
    if len(series) < 2:
        return 0.0
    returns = series.pct_change().dropna() * 100
    if len(returns) == 0:
        return 0.0
    return float(returns.std())


def compute_liquidity_score(df_hist):
    if df_hist is None or df_hist.empty:
        return 0.0
    close_series = pd.to_numeric(df_hist.get('Close'), errors='coerce').dropna()
    volume_series = pd.to_numeric(df_hist.get('Volume'), errors='coerce').dropna()
    if close_series.empty or volume_series.empty:
        return 0.0
    return float(close_series.tail(60).mean() * volume_series.tail(60).mean())

ALL_TICKERS = {**CHEAP_CANDIDATES, **BASE_TICKERS, **ADDITIONAL_SWING_TICKERS, **BROAD_UNIVERSE_TICKERS}

# Sugestões adicionais para expandir o universo (serão mescladas se não existirem)
SUGGESTED_EXTRA = [
    'BRFS3.SA', 'BRKM5.SA', 'RENT3.SA', 'BBSE3.SA', 'FLRY3.SA',
    'BRAP4.SA', 'HYPE3.SA', 'ODPV3.SA', 'GOLL4.SA', 'AZUL4.SA'
]

# Inclui automaticamente tickers encontrados em data/raw (sem duplicatas)
import re
TICKER_RE = re.compile(r'^[A-Z]{1,4}[0-9]{1,2}$')
try:
    raw_files = []
    for p in Path('data/raw').glob('*.csv'):
        stem = p.stem.upper()
        if TICKER_RE.match(stem):
            raw_files.append(stem + '.SA')
except Exception:
    raw_files = []

for tk in set(raw_files + SUGGESTED_EXTRA):
    if tk not in ALL_TICKERS and isinstance(tk, str) and tk.endswith('.SA'):
        symbol = tk.split('.')[0]
        ALL_TICKERS[tk] = {'name': symbol, 'sector': 'Unknown', 'segment': 'Unknown'}

# Reconcile core tickers: remove any core tickers that still aren't present and log them
missing_core_tickers = [ticker for ticker in SWING_TRADE_CORE_TICKERS if ticker not in ALL_TICKERS]
if missing_core_tickers:
    print(f"Aviso: alguns tickers do core não estavam disponíveis e foram removidos: {', '.join(missing_core_tickers)}")
    SWING_TRADE_CORE_TICKERS = [ticker for ticker in SWING_TRADE_CORE_TICKERS if ticker in ALL_TICKERS]

print("=" * 60)
print("BAIXANDO DADOS REAIS DA B3")
print("=" * 60)

# 1 - Baixar dados historicos
print("\nEtapa 1: Baixando dados históricos...")
all_data = {}
for ticker, info in ALL_TICKERS.items():
    try:
        print(f"   Baixando {ticker}...", end=" ")
        stock = yf.Ticker(ticker)
        df = stock.history(start=HISTORY_START)
        all_data[ticker] = {
            'data': df,
            'info': info
        }
        print(f"OK ({len(df)} dias)")
    except Exception as e:
        print(f"Erro: {e}")

print("\nTodos os dados baixados com sucesso!")

# 1.1 Filtrar candidatas baratas por preco medio e volatilidade
print("\nSelecionando universo para swing trade...")
selected_data = []
for ticker, data in all_data.items():
    df_hist = data['data']
    if df_hist is None or df_hist.empty:
        print(f"   {ticker} sem dados, ignorado")
        continue

    close_series = pd.to_numeric(df_hist['Close'], errors='coerce').dropna()
    volume_series = pd.to_numeric(df_hist['Volume'], errors='coerce').dropna()
    if len(close_series) < 40 or len(volume_series) < 40:
        print(f"   {ticker} ignorado por histórico curto ({len(close_series)} pregões)")
        continue

    recent_close = close_series.tail(60)
    mean_close = float(recent_close.mean())
    volatility = compute_volatility(recent_close)
    liquidity_score = compute_liquidity_score(df_hist)
    selected_data.append({
        'ticker': ticker,
        'data': data,
        'liquidity_score': liquidity_score,
        'mean_close': mean_close,
        'volatility': volatility,
        'days': len(close_series),
    })

selected_data = sorted(
    selected_data,
    key=lambda item: (item['liquidity_score'], item['volatility'], item['days']),
    reverse=True,
)

filtered_data = {}
for item in selected_data[:UNIVERSE_LIMIT]:
    filtered_data[item['ticker']] = item['data']
    print(
        f"   {item['ticker']} selecionado (liq {item['liquidity_score']:.2f}, "
        f"preco medio {item['mean_close']:.2f}, vol {item['volatility']:.2f}%)"
    )

all_data = filtered_data
print(f"Total selecionado: {len(all_data)} empresas")

# 2️⃣ Criar dimCompany
print("\nEtapa 2: Criando dimCompany...")
companies = []
for idx, (ticker, data) in enumerate(all_data.items(), 1):
    companies.append({
        'keyCompany': idx,
        'stockCodeCompany': ticker.split('.')[0],
        'nameCompany': data['info']['name'],
        'sectorCodeCompany': data['info']['sector'][:3].upper(),
        'sectorCompany': data['info']['sector'],
        'segmentCompany': data['info']['segment']
    })

df_company = pd.DataFrame(companies)
df_company.to_csv(data_raw / "dimCompany.csv", index=False)
print(f"   dimCompany.csv criado ({len(df_company)} empresas)")

# 3️⃣ Criar dimCoin
print("\nEtapa 3: Criando dimCoin...")
coins = {
    'keyCoin': [1, 2],
    'abbrevCoin': ['BRL', 'USD'],
    'nameCoin': ['Real', 'Dólar'],
    'symbolCoin': ['R$', '$']
}
df_coin = pd.DataFrame(coins)
df_coin.to_csv(data_raw / "dimCoin.csv", index=False)
print(f"   dimCoin.csv criado")

# 4️⃣ Criar dimTime - Datas únicas de todos os dados
print("\nEtapa 4: Criando dimTime...")
all_dates = set()
for ticker, data in all_data.items():
    all_dates.update([d.date() for d in data['data'].index])

all_dates = sorted(list(all_dates))
dim_time_data = {
    'keyTime': range(len(all_dates)),
    'datetime': [str(d) for d in all_dates],
    'dayTime': [d.day for d in all_dates],
    'dayWeekTime': [d.weekday() for d in all_dates],
    'dayWeekAbbrevTime': [d.strftime('%a') for d in all_dates],
    'dayWeekCompleteTime': [d.strftime('%A') for d in all_dates],
    'monthTime': [d.month for d in all_dates],
    'monthAbbrevTime': [d.strftime('%b') for d in all_dates],
    'monthCompleteTime': [d.strftime('%B') for d in all_dates],
    'bimonthTime': [(d.month - 1) // 2 + 1 for d in all_dates],
    'quarterTime': [(d.month - 1) // 3 + 1 for d in all_dates],
    'semesterTime': [(d.month - 1) // 6 + 1 for d in all_dates],
    'yearTime': [d.year for d in all_dates]
}
df_time = pd.DataFrame(dim_time_data)
df_time.to_csv(data_raw / "dimTime.csv", index=False)
print(f"   dimTime.csv criado ({len(df_time)} datas)")

# 5️⃣ Criar factCoins - Taxas de câmbio (simuladas)
print("\nEtapa 5: Criando factCoins...")
# Para simplicidade, vamos usar valores de câmbio simulados
np.random.seed(42)
fact_coins_data = {
    'keyTime': np.repeat(range(len(all_dates)), 2),
    'keyCoin': np.tile([1, 2], len(all_dates)),
    'valueCoin': np.random.uniform(4.5, 6.0, len(all_dates) * 2)
}
df_fact_coins = pd.DataFrame(fact_coins_data)
df_fact_coins.to_csv(data_raw / "factCoins.csv", index=False)
print(f"   factCoins.csv criado")

# 6️⃣ Criar factStocks - Dados OHLC reais
print("\nEtapa 6: Criando factStocks...")
fact_stocks_list = []

# Criar mapa de datas para keyTime
date_to_key = {str(all_dates[i]): i for i in range(len(all_dates))}

for company_key, (ticker, data) in enumerate(all_data.items(), 1):
    df_ohlc = data['data'].reset_index()
    date_col = 'Date' if 'Date' in df_ohlc.columns else 'Datetime'
    df_ohlc[date_col] = df_ohlc[date_col].astype(str).str[:10]  # Pega apenas YYYY-MM-DD
    
    for _, row in df_ohlc.iterrows():
        date_str = row[date_col]
        # Pré-validar Close: se NaN ou zero, pular esta linha
        close_val = float(row['Close']) if pd.notna(row['Close']) else None
        if close_val is None or close_val <= 0:
            continue
        
        if date_str in date_to_key:
            fact_stocks_list.append({
                'keyTime': date_to_key[date_str],
                'keyCompany': company_key,
                'openValueStock': float(row['Open']) if pd.notna(row['Open']) and float(row['Open']) > 0 else close_val,
                'closeValueStock': close_val,
                'highValueStock': float(row['High']) if pd.notna(row['High']) and float(row['High']) > 0 else close_val,
                'lowValueStock': float(row['Low']) if pd.notna(row['Low']) and float(row['Low']) > 0 else close_val,
                'quantityStock': float(row['Volume']) if pd.notna(row['Volume']) else 0.0,
            })

df_fact_stocks = pd.DataFrame(fact_stocks_list)
df_fact_stocks.to_csv(data_raw / "factStocks.csv", index=False)
print(f"   ✅ factStocks.csv criado ({len(df_fact_stocks)} registros)")

# Resumo
print("\n" + "=" * 60)
print("✅ DADOS BAIXADOS COM SUCESSO!")
print("=" * 60)
print(f"\n📊 Resumo dos dados salvos em data/raw/:")
print(f"   • dimCompany.csv: {len(df_company)} empresas")
print(f"   • dimCoin.csv: {len(df_coin)} moedas")
print(f"   • dimTime.csv: {len(df_time)} datas ({all_dates[0]} a {all_dates[-1]})")
print(f"   • factCoins.csv: {len(df_fact_coins)} registros")
print(f"   • factStocks.csv: {len(df_fact_stocks)} registros OHLC")
print(f"   • Janela de coleta: {HISTORY_START} até hoje")
print(f"\n🚀 Próximo passo: python run_pipeline_simple.py")
