# Classe de carga e merge
import pandas as pd
from .config import Config

class DataIngestor:
    def __init__(self):
        self.path = Config.DATA_RAW

    def load_and_merge(self) -> pd.DataFrame:
        """Carrega os CSVs e realiza o merge relacional (Estrela/Floco de Neve)."""
        print("🚀 Iniciando ingestão de dados...")
        
        # Carregando tabelas
        dim_company = pd.read_csv(self.path / "dimCompany.csv")
        dim_coin = pd.read_csv(self.path / "dimCoin.csv")
        dim_time = pd.read_csv(self.path / "dimTime.csv")
        fact_coins = pd.read_csv(self.path / "factCoins.csv")
        fact_stocks = pd.read_csv(self.path / "factStocks.csv")

        # Merge factStocks e dimCompany
        merged = fact_stocks.merge(dim_company, how="inner", on="keyCompany")

        # Merge com dimTime
        merged = merged.merge(dim_time, how="inner", on="keyTime")

        # Merge factCoins e dimCoin para obter valor da moeda
        fact_coins_merged = fact_coins.merge(dim_coin, how="inner", on="keyCoin")
        fact_coins_merged = fact_coins_merged[fact_coins_merged['abbrevCoin'] == 'BRL']

        # Merge final para adicionar taxa de câmbio
        # Selecionamos apenas keyTime e valueCoin para evitar duplicidade de colunas
        final_df = merged.merge(
            fact_coins_merged[['keyTime', 'valueCoin']], 
            how="left", 
            on="keyTime"
        )

        print(f"✅ Ingestão concluída. Shape total: {final_df.shape}")
        return final_df