"""
Script para validação do modelo
Treina SEM dados de 28 de abril e depois compara a predição com o valor real
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

from src.data_ingestion import DataIngestor
from src.preprocessing import DataProcessor
from src.trainer import LinearRegressionTrainer
from src.config import Config

def validate_prediction_for_date(target_date_str):
    """
    Faz validação: treina SEM o dia especificado e depois prediz para esse dia.
    
    Args:
        target_date_str (str): Data no formato YYYY-MM-DD (ex: 2026-04-28)
    """
    print("=" * 70)
    print(f"🔬 VALIDAÇÃO DO MODELO - Predição para {target_date_str}")
    print("=" * 70)
    
    # 1️⃣ INGESTÃO E PROCESSAMENTO
    print("\n📥 Etapa 1: Carregando dados...")
    ingestor = DataIngestor()
    df_raw = ingestor.load_and_merge()
    
    # Extrair data ANTES do processamento
    df_raw['date_str'] = pd.to_datetime(df_raw['datetime']).dt.strftime('%Y-%m-%d')
    
    processor = DataProcessor(save_artifacts=False)
    
    # Separar dados ANTES do processamento completo
    mask_validation = df_raw['date_str'] == target_date_str
    df_validation_raw = df_raw[mask_validation].copy()
    df_train_raw = df_raw[~mask_validation].copy()
    
    # Extrair y real E NOMES DE EMPRESAS ANTES do processamento
    y_validation_real = df_validation_raw[Config.TARGET].values
    validation_companies = df_validation_raw[['stockCodeCompany']].values.flatten()
    
    # Processar separadamente
    df_train_clean = processor.clean_data(df_train_raw)
    df_validation_clean = processor.clean_data(df_validation_raw)
    
    # 2️⃣ SEPARAR DADOS DE VALIDAÇÃO
    print(f"\n🔍 Etapa 2: Separando dados de {target_date_str}...")
    df_validation = df_validation_clean.copy()
    df_train_data = df_train_clean.copy()
    
    print(f"   ✅ Dados de treino: {len(df_train_data)} registros")
    print(f"   ✅ Dados de validação: {len(df_validation)} registros")
    
    # 3️⃣ PROCESSAR DADOS DE TREINO
    print(f"\n🔧 Etapa 3: Processando dados de treino...")
    processor_train = DataProcessor(save_artifacts=False)
    X_train, y_train = processor_train.transform(df_train_data)
    
    # Dividir em treino/teste
    X_train_split, X_test, y_train_split, y_test = train_test_split(
        X_train, y_train, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # 4️⃣ TREINAR MODELO (SEM DADOS DE 28 DE ABRIL)
    print(f"\n🤖 Etapa 4: Treinando modelo (SEM {target_date_str})...")
    lr_trainer = LinearRegressionTrainer()
    lr_trainer.train(X_train_split, y_train_split)
    
    # Métricas
    y_pred_test = lr_trainer.model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"   ✅ Modelo treinado!")
    print(f"   📈 MSE: {mse_test:.4f}")
    print(f"   📈 R²:  {r2_test:.4f}")
    
    # 5️⃣ FAZER PREDIÇÃO PARA 28 DE ABRIL
    print(f"\n🔮 Etapa 5: Fazendo predição para {target_date_str}...")
    
    # Processar dados de validação usando MESMOS encoders/scalers
    X_validation = df_validation.copy()
    
    # Aplicar os mesmos codificadores do treino
    for col in Config.COLS_CATEGORICAS:
        if col in processor_train.label_encoders:
            encoder = processor_train.label_encoders[col]
            try:
                X_validation[col] = encoder.transform(X_validation[col].astype(str))
            except ValueError:
                X_validation[col] = encoder.transform([encoder.classes_[0]])[0]
    
    # Aplicar o mesmo scaler do treino
    X_validation_features = X_validation[[c for c in Config.COLUNAS_INTERESSE if c != Config.TARGET]]
    X_validation_scaled = processor_train.scaler.transform(X_validation_features)
    
    # Extrair y real
    y_validation_real = df_validation[Config.TARGET].values
    
    # Fazer predição
    y_validation_pred = lr_trainer.model.predict(X_validation_scaled)
    
    # 6️⃣ COMPARAÇÃO
    print(f"\n📊 Etapa 6: Comparando Predição vs Realidade")
    print(f"\n{'Empresa':<20} {'Real':<12} {'Predito':<12} {'Erro':<12} {'Acerto':<10}")
    print("-" * 70)
    
    total_erro = 0
    for idx, (pred, real, ticker) in enumerate(zip(y_validation_pred, y_validation_real, validation_companies)):
        erro = abs(real - pred)
        total_erro += erro
        acerto_pct = (1 - (erro / real)) * 100 if real != 0 else 0
        
        sinal = "✅" if acerto_pct > 95 else "⚠️" if acerto_pct > 90 else "❌"
        
        print(f"{str(ticker):<20} R$ {real:<10.2f} R$ {pred:<10.2f} R$ {erro:<10.2f} {acerto_pct:>6.1f}% {sinal}")
    
    print("-" * 70)
    mean_erro = total_erro / len(y_validation_pred)
    print(f"\n📈 Erro Médio: R$ {mean_erro:.4f}")
    print(f"✅ Acerto Médio: {(1 - (mean_erro / np.mean(y_validation_real))) * 100:.2f}%")
    
    print("\n" + "=" * 70)
    print("✅ Validação concluída!")
    print("=" * 70)

if __name__ == "__main__":
    # Validar para 28 de abril
    validate_prediction_for_date("2026-04-28")
