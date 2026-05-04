from src.data_ingestion import DataIngestor
import pandas as pd

df = DataIngestor().load_and_merge()
print(f'Shape: {df.shape}')
print(f'Colunas: {list(df.columns)}')
print(f'Tickers unicos: {df["stockCodeCompany"].nunique()}')
print(f'\nPrimeiras linhas:')
print(df[['datetime', 'stockCodeCompany', 'closeValueStock']].head(10))
print(f'\nUltimas linhas:')
print(df[['datetime', 'stockCodeCompany', 'closeValueStock']].tail(10))
print(f'\nSummary of closeValueStock:')
print(df['closeValueStock'].describe())
print(f'\nZeros count: {(df["closeValueStock"] == 0).sum()}')
