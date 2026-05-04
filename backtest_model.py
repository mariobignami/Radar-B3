"""
Script de Backtest - Testa o modelo em múltiplos dias
Treina sem dados até o dia N e prediz para o dia N
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json

from src.data_ingestion import DataIngestor
from src.preprocessing import DataProcessor
from src.trainer import LinearRegressionTrainer
from src.config import Config

def run_backtest(start_date_str, end_date_str, step_days=1, df_raw=None):
    """
    Executa backtest para múltiplos dias.
    
    Args:
        start_date_str: Data inicial (YYYY-MM-DD)
        end_date_str: Data final (YYYY-MM-DD)
        step_days: Intervalo entre testes (padrão: 1 = diário)
    """
    print("=" * 80)
    print(f"🔬 BACKTEST DO MODELO")
    print(f"   Período: {start_date_str} a {end_date_str}")
    print("=" * 80)
    
    # 1️⃣ CARREGAR TODOS OS DADOS
    if df_raw is None:
        print("\n📥 Carregando dados históricos...")
        ingestor = DataIngestor()
        df_raw = ingestor.load_and_merge()
    df_raw['date_str'] = pd.to_datetime(df_raw['datetime']).dt.strftime('%Y-%m-%d')
    
    # Converter datas
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Gerar datas do teste
    test_dates = []
    current_date = start_date
    while current_date <= end_date:
        test_dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=step_days)
    
    print(f"✅ {len(test_dates)} datas para testar")
    
    # 2️⃣ EXECUTAR TESTES
    results = []
    
    for idx, test_date in enumerate(test_dates, 1):
        print(f"\n[{idx}/{len(test_dates)}] Testando {test_date}...", end=" ")
        
        try:
            # Separar dados: treino (até dia anterior) e teste (dia atual)
            mask_test = df_raw['date_str'] == test_date
            mask_train = ~mask_test & (df_raw['date_str'] < test_date)
            
            df_train_raw = df_raw[mask_train].copy()
            df_test_raw = df_raw[mask_test].copy()
            
            if len(df_test_raw) == 0:
                print("❌ (sem dados)")
                continue
            
            # Extrair dados reais
            y_real = df_test_raw[Config.TARGET].values
            companies = df_test_raw[['stockCodeCompany']].values.flatten()
            
            # Processar e treinar
            processor = DataProcessor(save_artifacts=False)
            df_train_clean = processor.clean_data(df_train_raw)
            
            X_train, y_train = processor.transform(df_train_clean)
            X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
                X_train, y_train, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
            )
            
            # Treinar modelo
            trainer = LinearRegressionTrainer()
            trainer.train(X_train_split, y_train_split)
            
            # Fazer predição
            df_test_clean = processor.clean_data(df_test_raw)
            feature_cols = [c for c in Config.COLUNAS_INTERESSE if c != Config.TARGET]
            X_test_features = df_test_clean[feature_cols].copy()
            
            # Aplicar transformações
            for col in Config.COLS_CATEGORICAS:
                if col in processor.label_encoders:
                    encoder = processor.label_encoders[col]
                    try:
                        X_test_features[col] = encoder.transform(X_test_features[col].astype(str))
                    except ValueError:
                        X_test_features[col] = encoder.transform([encoder.classes_[0]])[0]
            
            X_test_imputed = processor.imputer.transform(X_test_features)
            X_test_scaled = processor.scaler.transform(X_test_imputed)
            y_pred = trainer.model.predict(X_test_scaled)
            
            # Calcular métricas
            erro = np.abs(y_real - y_pred)
            denom = np.abs(y_real) + np.abs(y_pred) + 1e-6
            smape = (200 * erro) / denom
            acerto = 100 - (smape / 2)
            acerto = np.clip(acerto, 0, 100)
            
            # Salvar resultados
            for comp, real, pred, err, acc, smape_val in zip(companies, y_real, y_pred, erro, acerto, smape):
                results.append({
                    'date': test_date,
                    'company': comp,
                    'real': round(float(real), 2),
                    'predicted': round(float(pred), 2),
                    'error': round(float(err), 2),
                    'accuracy': round(float(acc), 2),
                    'smape': round(float(smape_val), 2)
                })
            
            # Média do dia
            mean_acc = np.mean(acerto)
            print(f"✅ (Acerto: {mean_acc:.2f}%)")
            
        except Exception as e:
            print(f"❌ ({str(e)[:30]})")
    
    # 3️⃣ GERAR RELATÓRIO
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DO BACKTEST")
    print("=" * 80)
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("❌ Nenhum resultado para mostrar")
        return None
    
    # Estatísticas por data (resumo)
    print("\n📅 Acurácia por Data:")
    print(f"{'Data':<12} {'Empresas':<10} {'Media':<10} {'Min':<10} {'Max':<10}")
    print("-" * 56)

    for date in df_results['date'].unique():
        df_date = df_results[df_results['date'] == date]
        count_companies = df_date['company'].nunique()
        mean_acc = df_date['accuracy'].mean()
        min_acc = df_date['accuracy'].min()
        max_acc = df_date['accuracy'].max()
        print(f"{date:<12} {count_companies:<10} {mean_acc:>7.2f}% {min_acc:>8.2f}% {max_acc:>8.2f}%")
    
    # Estatísticas gerais
    print("\n📈 Estatísticas Gerais:")
    print(f"   Acurácia Média:    {df_results['accuracy'].mean():.2f}%")
    print(f"   Acurácia Mínima:   {df_results['accuracy'].min():.2f}%")
    print(f"   Acurácia Máxima:   {df_results['accuracy'].max():.2f}%")
    print(f"   Desvio Padrão:     {df_results['accuracy'].std():.2f}%")
    print(f"   Erro Médio:        R$ {df_results['error'].mean():.2f}")
    
    # Salvar resultados em JSON
    with open('backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Resultados salvos em backtest_results.json")
    
    return df_results

if __name__ == "__main__":
    # Backtest automatico dos ultimos 180 dias disponiveis
    ingestor = DataIngestor()
    df_all = ingestor.load_and_merge()
    df_all['date_str'] = pd.to_datetime(df_all['datetime']).dt.date
    max_date = df_all['date_str'].max()
    start_date = max_date - timedelta(days=180)

    run_backtest(
        start_date.strftime("%Y-%m-%d"),
        max_date.strftime("%Y-%m-%d"),
        step_days=1,
        df_raw=df_all
    )
