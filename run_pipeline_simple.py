"""
Pipeline simples sem Keras (para evitar problemas de compatibilidade)
Roda apenas com Regressão Linear + Scikit-learn
"""
from src.data_ingestion import DataIngestor
from src.preprocessing import DataProcessor
from src.trainer import LinearRegressionTrainer
from src.config import Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def run_simple_pipeline():
    print("=" * 60)
    print("🚀 PIPELINE DE PREVISÃO DE AÇÕES B3 (Versão Simplificada)")
    print("=" * 60)
    
    # 1️⃣ INGESTÃO DE DADOS
    print("\n📥 Etapa 1: Ingestão de Dados")
    ingestor = DataIngestor()
    df_raw = ingestor.load_and_merge()
    print(f"   ✅ Dados carregados: {df_raw.shape}")
    print(f"   📊 Colunas: {df_raw.columns.tolist()}")
    
    # 2️⃣ PROCESSAMENTO
    print("\n🔧 Etapa 2: Processamento de Dados")
    processor = DataProcessor()
    df_clean = processor.clean_data(df_raw)
    X, y = processor.transform(df_clean)
    print(f"   ✅ Dados processados: X={X.shape}, y={y.shape}")
    print(f"   📊 Features: {X.shape[1]} características")
    
    # 3️⃣ SPLIT
    print("\n✂️  Etapa 3: Dividindo Treino/Teste")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    print(f"   ✅ Treino: {X_train.shape[0]} amostras")
    print(f"   ✅ Teste: {X_test.shape[0]} amostras")
    
    # 4️⃣ TREINO - REGRESSÃO LINEAR
    print("\n🤖 Etapa 4: Treinamento - Regressão Linear")
    lr_trainer = LinearRegressionTrainer()
    lr_trainer.train(X_train, y_train)
    
    # Predições
    y_pred_train = lr_trainer.model.predict(X_train)
    y_pred_test = lr_trainer.model.predict(X_test)
    
    # Métricas
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print(f"   ✅ Modelo treinado com sucesso!")
    print(f"\n   📈 Métricas de Treino:")
    print(f"      • MSE:  {mse_train:.4f}")
    print(f"      • R²:   {r2_train:.4f}")
    print(f"\n   📈 Métricas de Teste:")
    print(f"      • MSE:  {mse_test:.4f}")
    print(f"      • R²:   {r2_test:.4f}")
    
    # Salvar modelo
    lr_trainer.save_model("linear_regression_model.pkl")
    
    # 5️⃣ EXEMPLO DE PREDIÇÃO
    print("\n🔮 Etapa 5: Exemplo de Predição")
    sample = X_test[:1]
    pred = lr_trainer.model.predict(sample)[0]
    real = y_test.iloc[0] if hasattr(y_test, 'iloc') else y_test[0]
    print(f"   Valor Real: R$ {real:.2f}")
    print(f"   Predito:    R$ {pred:.2f}")
    print(f"   Erro:       R$ {abs(real - pred):.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline finalizada com sucesso!")
    print("=" * 60)

if __name__ == "__main__":
    run_simple_pipeline()
