"""
Modelo XGBoost para Previsão de Swing Trade
Com Optuna para hyperparameter tuning e SHAP para interpretabilidade
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import shap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib

try:
    from .config import Config
except ImportError:
    try:
        from config import Config
    except ImportError:
        Config = None

try:
    from .feature_engineering import FeatureEngineer
except ImportError:
    from feature_engineering import FeatureEngineer


class XGBoostSwingTrader:
    """Modelo XGBoost para previsão de swing trade."""
    
    def __init__(self, target_horizon: int = 5):
        """
        Inicializa o modelo.
        
        Args:
            target_horizon: Horizonte de previsão em dias (5, 10 ou 20)
        """
        self.target_horizon = target_horizon
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.shap_values = None
        self.feature_names = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara dados para treinamento.
        
        Args:
            df: DataFrame com features e targets
        
        Returns:
            X (features), y (target)
        """
        # Remover colunas não usadas
        exclude_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Target_Return_5d', 'Target_Return_10d', 'Target_Return_20d',
            'Target_Direction_5d', 'Target_Direction_10d', 'Target_Direction_20d',
            'Target_Drawdown_20d', 'Ticker'
        ]
        
        # Selecionar target baseado no horizonte
        if self.target_horizon == 5:
            target_col = 'Target_Return_5d'
        elif self.target_horizon == 10:
            target_col = 'Target_Return_10d'
        else:
            target_col = 'Target_Return_20d'
        
        # Features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remover NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, 
                 y_train: pd.Series, cv: TimeSeriesSplit) -> float:
        """
        Função objetivo para Optuna optimization.
        
        Args:
            trial: Trial do Optuna
            X_train: Features de treino
            y_train: Target de treino
            cv: Cross-validation splitter
        
        Returns:
            MSE médio da validação
        """
        # Hiperparâmetros para otimizar
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        
        cv_scores = []
        
        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            cv_scores.append(mse)
        
        return np.mean(cv_scores)
    
    def train_with_optuna(self, df: pd.DataFrame, n_trials: int = 50) -> Dict:
        """
        Treina modelo com Optuna hyperparameter tuning.
        
        Args:
            df: DataFrame com features e targets
            n_trials: Número de trials do Optuna
        
        Returns:
            Dicionário com métricas de treinamento
        """
        X, y = self.prepare_data(df)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X, y, tscv),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        print(f"\n✅ Melhores parâmetros: {self.best_params}")
        
        # Treinar modelo final com melhores parâmetros
        final_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1,
            **self.best_params
        }
        
        self.model = xgb.XGBRegressor(**final_params)
        self.model.fit(X, y)
        
        # Calcular feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calcular SHAP values
        print("Calculando SHAP values...")
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        
        # Métricas
        y_pred = self.model.predict(X)
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': self.model.score(X, y)
        }
        
        return metrics
    
    def train_simple(self, df: pd.DataFrame) -> Dict:
        """
        Treina modelo sem Optuna (usa parâmetros padrão razoáveis).
        
        Args:
            df: DataFrame com features e targets
        
        Returns:
            Dicionário com métricas de treinamento
        """
        X, y = self.prepare_data(df)
        
        # Parâmetros padrão para swing trade
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # SHAP values
        print("Calculando SHAP values...")
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)
        
        # Métricas
        y_pred = self.model.predict(X)
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': self.model.score(X, y)
        }
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Faz predições.
        
        Args:
            X: Features
        
        Returns:
            Predições
        """
        if self.model is None:
            raise ValueError("Modelo não treinado. Chame train() primeiro.")
        
        return self.model.predict(X)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Faz predições com intervalo de confiança baseado em SHAP.
        
        Args:
            X: Features
        
        Returns:
            Predições e incerteza estimada
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        predictions = self.model.predict(X)
        
        # Estimar incerteza baseada em SHAP values
        if self.shap_values is not None:
            shap_explainer = shap.TreeExplainer(self.model)
            shap_vals = shap_explainer.shap_values(X)
            uncertainty = np.abs(shap_vals).mean(axis=1)
        else:
            uncertainty = np.zeros(len(predictions))
        
        return predictions, uncertainty
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna as features mais importantes.
        
        Args:
            top_n: Número de features a retornar
        
        Returns:
            DataFrame com feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Modelo não treinado.")
        
        return self.feature_importance.head(top_n)
    
    def explain_prediction(self, X: pd.DataFrame, idx: int = 0) -> Dict:
        """
        Explica uma predição específica usando SHAP.
        
        Args:
            X: Features
            idx: Índice da predição a explicar
        
        Returns:
            Dicionário com explicação
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(X.iloc[[idx]])
        
        # Features com maior impacto
        feature_impact = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': shap_vals[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return {
            'prediction': self.model.predict(X.iloc[[idx]])[0],
            'top_features': feature_impact.head(10).to_dict('records')
        }
    
    def save_model(self, path: str):
        """Salva o modelo em disco."""
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'target_horizon': self.target_horizon,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, path)
        print(f"✅ Modelo salvo em {path}")
    
    def load_model(self, path: str):
        """Carrega o modelo do disco."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_names = model_data['feature_names']
        self.target_horizon = model_data['target_horizon']
        self.feature_importance = model_data['feature_importance']
        
        print(f"✅ Modelo carregado de {path}")


def train_ensemble_model(df: pd.DataFrame) -> Dict[str, XGBoostSwingTrader]:
    """
    Treina ensemble de modelos para diferentes horizontes.
    
    Args:
        df: DataFrame com features e targets
    
    Returns:
        Dicionário com modelos para cada horizonte
    """
    models = {}
    
    for horizon in [5, 10, 20]:
        print(f"\n=== Treinando modelo para horizonte {horizon} dias ===")
        model = XGBoostSwingTrader(target_horizon=horizon)
        metrics = model.train_simple(df)
        
        models[f'horizon_{horizon}'] = model
        
        print(f"Métricas: {metrics}")
        print(f"Top 10 features:")
        print(model.get_feature_importance(10))
    
    

if __name__ == "__main__":
    # Teste com dados fictícios
    from feature_engineering import FeatureEngineer
    
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
    close_prices = np.cumsum(np.random.randn(500) * 0.02) + 100
    
    test_df = pd.DataFrame({
        'Open': close_prices + np.random.randn(500) * 0.5,
        'High': close_prices + np.random.rand(500) * 2,
        'Low': close_prices - np.random.rand(500) * 2,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 10000000, 500)
    }, index=dates)
    
    engineer = FeatureEngineer()
    features = engineer.create_features_for_stock(test_df, 'TEST4',
                                                 include_sentiment=False,
                                                 include_macro=False)
    
    print("\n=== Teste XGBoost Model ===")
    model = XGBoostSwingTrader(target_horizon=5)
    metrics = model.train_simple(features)
    
    print(f"\nMétricas de treinamento:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTop 10 features:")
    print(model.get_feature_importance(10))
