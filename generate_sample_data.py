import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Definir caminho
data_raw = Path("data/raw")
data_raw.mkdir(parents=True, exist_ok=True)

# 1️⃣ dimCompany - Empresas da B3
companies = {
    'keyCompany': [1, 2, 3, 4, 5],
    'stockCodeCompany': ['PETR4', 'VALE3', 'ITUB4', 'ABEV3', 'WEGE3'],
    'nameCompany': ['Petrobras', 'Vale', 'Itaú', 'Ambev', 'WEG'],
    'sectorCodeCompany': ['EN', 'MM', 'FIN', 'BEV', 'IND'],
    'sectorCompany': ['Energia', 'Mineração', 'Financeiro', 'Bebidas', 'Indústria'],
    'segmentCompany': ['Petróleo', 'Ferro', 'Banco', 'Bebidas', 'Motores']
}
pd.DataFrame(companies).to_csv(data_raw / "dimCompany.csv", index=False)
print("✅ dimCompany.csv criado")

# 2️⃣ dimCoin - Moedas
coins = {
    'keyCoin': [1, 2],
    'abbrevCoin': ['USD', 'EUR'],
    'nameCoin': ['Dólar', 'Euro'],
    'symbolCoin': ['$', '€']
}
pd.DataFrame(coins).to_csv(data_raw / "dimCoin.csv", index=False)
print("✅ dimCoin.csv criado")

# 3️⃣ dimTime - Datas
dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
dim_time_data = {
    'keyTime': range(len(dates)),
    'datetime': dates.strftime('%Y-%m-%d'),
    'dayTime': dates.day,
    'dayWeekTime': dates.dayofweek,
    'dayWeekAbbrevTime': dates.strftime('%a'),
    'dayWeekCompleteTime': dates.strftime('%A'),
    'monthTime': dates.month,
    'monthAbbrevTime': dates.strftime('%b'),
    'monthCompleteTime': dates.strftime('%B'),
    'bimonthTime': (dates.month - 1) // 2 + 1,
    'quarterTime': dates.quarter,
    'semesterTime': ((dates.month - 1) // 6) + 1,
    'yearTime': dates.year
}
pd.DataFrame(dim_time_data).to_csv(data_raw / "dimTime.csv", index=False)
print("✅ dimTime.csv criado")

# 4️⃣ factCoins - Taxas de câmbio
np.random.seed(42)
fact_coins_data = {
    'keyTime': np.repeat(range(len(dates)), 2),
    'keyCoin': np.tile([1, 2], len(dates)),
    'valueCoin': np.random.uniform(4.5, 6.0, len(dates) * 2)
}
pd.DataFrame(fact_coins_data).to_csv(data_raw / "factCoins.csv", index=False)
print("✅ factCoins.csv criado")

# 5️⃣ factStocks - Dados OHLC
np.random.seed(42)
num_records = len(dates) * 5  # 5 empresas x cada data
fact_stocks_data = {
    'keyTime': np.repeat(range(len(dates)), 5),
    'keyCompany': np.tile(range(1, 6), len(dates)),
    'openValueStock': np.random.uniform(10, 100, num_records),
    'closeValueStock': np.random.uniform(10, 100, num_records),
    'highValueStock': np.random.uniform(10, 100, num_records),
    'lowValueStock': np.random.uniform(10, 100, num_records),
    'quantityStock': np.random.uniform(1000000, 10000000, num_records)
}
pd.DataFrame(fact_stocks_data).to_csv(data_raw / "factStocks.csv", index=False)
print("✅ factStocks.csv criado")

print("\n🎉 Todos os CSVs foram gerados em data/raw/")
