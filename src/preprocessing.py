# Classe de limpeza e encoding
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from .config import Config

class DataProcessor:
    def __init__(self, save_artifacts=True):
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        self.save_artifacts = save_artifacts

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra colunas e remove valores ausentes críticos."""
        print("🧹 Iniciando limpeza de dados...")
        
        # Seleção de colunas de interesse conforme Config
        df_filtered = df[Config.COLUNAS_INTERESSE].copy()
        
        # Removendo linhas onde o alvo (target) é nulo
        df_filtered = df_filtered.dropna(subset=[Config.TARGET])
        
        # Removendo NaNs remanescentes nas colunas críticas
        df_filtered.dropna(inplace=True)
        
        # Tratamento de infinitos (comum em dados financeiros)
        df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_filtered.dropna(inplace=True)

        # OHLC zerado aparece em pregoes incompletos; usa o fechamento como ancora.
        price_cols = ['openValueStock', 'highValueStock', 'lowValueStock', Config.TARGET]
        for col in price_cols:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        close = df_filtered[Config.TARGET]
        invalid_open = df_filtered['openValueStock'].isna() | (df_filtered['openValueStock'] <= 0)
        invalid_high = df_filtered['highValueStock'].isna() | (df_filtered['highValueStock'] <= 0)
        invalid_low = df_filtered['lowValueStock'].isna() | (df_filtered['lowValueStock'] <= 0)
        df_filtered.loc[invalid_open, 'openValueStock'] = close[invalid_open]
        df_filtered.loc[invalid_high, 'highValueStock'] = df_filtered.loc[invalid_high, ['openValueStock', Config.TARGET]].max(axis=1)
        df_filtered.loc[invalid_low, 'lowValueStock'] = df_filtered.loc[invalid_low, ['openValueStock', Config.TARGET]].min(axis=1)
        df_filtered['highValueStock'] = df_filtered[['highValueStock', 'openValueStock', Config.TARGET]].max(axis=1)
        df_filtered['lowValueStock'] = df_filtered[['lowValueStock', 'openValueStock', Config.TARGET]].min(axis=1)

        return df_filtered

    def transform(self, df: pd.DataFrame):
        """Aplica Encoding, Imputação e Escalonamento."""
        print("🧪 Transformando atributos...")
        
        # Encoding de Variáveis Categóricas
        for col in Config.COLS_CATEGORICAS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            # Salva o encoder para uso futuro no modelo
            if self.save_artifacts:
                joblib.dump(le, Config.MODEL_DIR / f"encoder_{col}.pkl")

        # Separação X e y
        X = df.drop(columns=[Config.TARGET])
        y = df[Config.TARGET]

        # Imputação de valores nulos remanescentes
        X_imputed = self.imputer.fit_transform(X)
        if self.save_artifacts:
            joblib.dump(self.imputer, Config.MODEL_DIR / "imputer.pkl")

        # Normalização (Scaling)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Salva o scaler para normalizar dados futuros da mesma forma
        if self.save_artifacts:
            joblib.dump(self.scaler, Config.MODEL_DIR / "scaler.pkl")

        return X_scaled, y
