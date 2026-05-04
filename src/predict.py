"""
Módulo de Previsão de Preços de Ações
Utiliza o modelo treinado para fazer predições de preços de fechamento
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from .config import Config
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

class StockPredictor:
    """Classe para fazer predições de preços de ações."""
    
    def __init__(self, model_name='linear_regression_model.pkl'):
        """
        Inicializa o preditor carregando o modelo treinado.
        
        Args:
            model_name (str): Nome do arquivo do modelo salvo em models/
        """
        self.model_path = Config.MODEL_DIR / model_name
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(Config.MODEL_DIR / "scaler.pkl")
            imputer_path = Config.MODEL_DIR / "imputer.pkl"
            self.imputer = joblib.load(imputer_path) if imputer_path.exists() else None
            
            # Carregar encoders
            self.label_encoders = {}
            for col in Config.COLS_CATEGORICAS:
                encoder_path = Config.MODEL_DIR / f"encoder_{col}.pkl"
                if encoder_path.exists():
                    self.label_encoders[col] = joblib.load(encoder_path)
            
            print(f"Modelo carregado: {model_name}")
        except FileNotFoundError as e:
            print(f"Erro: Arquivo nao encontrado - {e}")
            print("Execute primeiro: python run_pipeline_simple.py")
            raise
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            raise
    
    def _expected_features(self):
        """Return the feature order expected by the saved preprocessing artifacts."""
        if self.imputer is not None and hasattr(self.imputer, "feature_names_in_"):
            return list(self.imputer.feature_names_in_)
        return [col for col in Config.COLUNAS_INTERESSE if col != Config.TARGET]

    @staticmethod
    def _normalize_technical_indicators(technical_indicators):
        if not technical_indicators:
            return {}

        indicators = dict(technical_indicators)
        normalized = {}

        rsi = indicators.get("rsi_14")
        if rsi is not None:
            normalized["rsi_14"] = float(rsi)

        volatility_percent = indicators.get(
            "volatility_20d_percent", indicators.get("volatility")
        )
        if volatility_percent is not None:
            normalized["volatility"] = float(volatility_percent)
            normalized["vol_20d"] = float(volatility_percent) / 100

        drawdown_percent = indicators.get("max_drawdown_percent")
        if drawdown_percent is not None:
            normalized["max_drawdown_percent"] = float(drawdown_percent)
            normalized["drawdown_20d"] = float(drawdown_percent) / 100

        open_gap_percent = indicators.get(
            "open_gap_percent", indicators.get("open_gap_pct")
        )
        if open_gap_percent is not None:
            normalized["open_gap_percent"] = float(open_gap_percent)
            normalized["open_gap_pct"] = float(open_gap_percent) / 100

        volume_rel_percent = indicators.get("volume_rel_20d_percent")
        if volume_rel_percent is None:
            volume_rel_percent = indicators.get("volume_rel_20d")
        if volume_rel_percent is not None:
            normalized["volume_rel_20d_percent"] = float(volume_rel_percent)
            normalized["volume_rel_20d"] = (float(volume_rel_percent) / 100) - 1

        return normalized

    def predict_single(self, open_price, high_price, low_price, quantity, 
                      stock_code=None, sector='Energia', segment='Petróleo',
                      month=4, day_week=0, technical_indicators=None):
        """
        Faz uma predição para um único ponto de dados.
        """
        try:
            # Criar DataFrame com TODAS as colunas esperadas, na ordem correta
            normalized_indicators = self._normalize_technical_indicators(technical_indicators)
            input_values = {
                'stockCodeCompany': [stock_code] if stock_code else ["UNKNOWN"],
                'sectorCompany': [sector],
                'segmentCompany': [segment],
                'dayTime': [15],
                'dayWeekTime': [day_week],
                'monthTime': [month],
                'yearTime': [2026],
                'openValueStock': [float(open_price)],
                'highValueStock': [float(high_price)],
                'lowValueStock': [float(low_price)],
                'quantityStock': [float(quantity)],
                'valueCoin': [5.0]
            }

            for key, value in normalized_indicators.items():
                input_values[key] = [float(value)]

            input_data = pd.DataFrame(input_values)
            
            # Selecionar features na ordem esperada (sem target)
            features_order = self._expected_features()
            for col in features_order:
                if col not in input_data.columns:
                    input_data[col] = np.nan
            X = input_data[features_order].copy()
            
            # Codificar variáveis categóricas
            for col in Config.COLS_CATEGORICAS:
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    try:
                        X[col] = encoder.transform(X[col].astype(str))
                    except ValueError:
                        # Usar primeira classe como padrão
                        X[col] = encoder.transform([encoder.classes_[0]])[0]
            
            # Normalizar com as mesmas dimensões esperadas
            if self.imputer is not None:
                X_values = self.imputer.transform(X)
            else:
                X_values = X
            X_scaled = self.scaler.transform(X_values)
            
            # Fazer predição
            prediction = self.model.predict(X_scaled)[0]
            
            return {
                'status': 'Sucesso',
                'predicted_price': round(float(prediction), 2),
                'input': {
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'volume': quantity,
                    'sector': sector,
                    'technical_indicators': normalized_indicators
                }
            }
        except Exception as e:
            return {
                'status': 'Erro',
                'error': str(e),
                'predicted_price': None
            }
    
    def predict_batch(self, df):
        """
        Faz predições em lote a partir de um DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame com as colunas esperadas
        
        Returns:
            np.array: Predições
        """
        try:
            X = df.copy()
            
            # Codificar categóricas
            for col in Config.COLS_CATEGORICAS:
                if col in self.label_encoders:
                    X[col] = self.label_encoders[col].transform(X[col].astype(str))
            
            # Remover target se existir
            if Config.TARGET in X.columns:
                X = X.drop(columns=[Config.TARGET])
            
            # Normalizar e prever
            if self.imputer is not None:
                X_values = self.imputer.transform(X)
            else:
                X_values = X
            X_scaled = self.scaler.transform(X_values)
            predictions = self.model.predict(X_scaled)
            return predictions
        except Exception as e:
            print(f"Erro na predicao em lote: {e}")
            return None

    def predict(self, input_data):
        prepared_data = self._prepare_input(input_data)
        prediction = self.model.predict(prepared_data, verbose=0)
        
        # Retorno tratado conforme o tipo de modelo
        return float(prediction[0][0]) if self.model_type == 'neural_network' else float(prediction[0])

if __name__ == "__main__":
    # Exemplo de uso com dados fictícios de uma ação
    try:
        predictor = StockPredictor()
        result = predictor.predict_single(
            open_price=34.50,
            high_price=35.10,
            low_price=34.20,
            quantity=1_500_000,
            stock_code="ITUB4",
            sector="Financeiro",
            segment="Banco",
            month=1,
            day_week=4
        )
        print("\n--- Resultado da Predicao ---")
        print(f"Preco de Fechamento Estimado: R$ {result['predicted_price']:.2f}")
    except Exception as e:
        print(e)
