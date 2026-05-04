"""
Análise Dia-a-Dia: Treina no dia N, prevê para dia N+1, compara resultado real
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
import json

from src.config import Config
from src.data_ingestion import DataIngestor
from src.preprocessing import DataProcessor

TARGET = Config.TARGET
COLS_CATEGORICAS = Config.COLS_CATEGORICAS
COLS_NUMERICAS = Config.COLS_NUMERICAS
COLS_TEMPO = ['dayTime', 'dayWeekTime', 'monthTime', 'yearTime']


def analyze_consecutive_days():
    """
    Análise: Treina em um dia, prevém para o próximo, compara com resultado real
    """
    
    print("[ANALISE] Análise Dia-a-Dia: Predição e Validação")
    print("=" * 60)
    
    # Carregar dados
    data_ingestor = DataIngestor()
    df_merged = data_ingestor.load_and_merge()
    
    # Converter datetime
    df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
    
    # Datas únicas de trading
    unique_dates = df_merged['datetime'].dt.date.unique()
    print(f"\n[DATAS] Datas de trading disponíveis: {len(unique_dates)}")
    print(f"   Periodo: {unique_dates[0]} a {unique_dates[-1]}")
    
    # Guardar análises
    analysis_results = []
    
    # Pares de dias consecutivos
    for i in range(len(unique_dates) - 1):
        training_date = unique_dates[i]
        prediction_date = unique_dates[i + 1]
        
        # Dados de treinamento (até o dia i)
        df_train = df_merged[df_merged['datetime'].dt.date <= training_date].copy()
        
        # Dados de predição (dia i+1)
        df_predict = df_merged[df_merged['datetime'].dt.date == prediction_date].copy()
        
        if len(df_train) == 0 or len(df_predict) == 0:
            continue
        
        print(f"\n{'='*60}")
        print(f"[TREINO] Treinamento: {training_date} -> Predicao: {prediction_date}")
        print(f"{'='*60}")
        print(f"Dados treino: {len(df_train)} | Predicao: {len(df_predict)}")
        
        try:
            # Processar dados de treinamento
            processor_train = DataProcessor(save_artifacts=False)
            df_train_clean = processor_train.clean_data(df_train)
            
            if len(df_train_clean) < 10:
                print(f"[ERRO] Dados insuficientes")
                continue
            
            # Separar features e target
            X_train = df_train_clean[COLS_CATEGORICAS + COLS_TEMPO + COLS_NUMERICAS].copy()
            y_train = df_train_clean[TARGET].copy()
            
            # Transformar dados de treinamento
            X_train_transformed, encoders_dict, scaler = processor_train.transform(X_train)
            
            if X_train_transformed is None:
                print(f"[ERRO] Erro na transformacao")
                continue
            
            # Treinar modelo
            model = LinearRegression()
            model.fit(X_train_transformed, y_train)
            model_r2 = model.score(X_train_transformed, y_train)
            
            # Processar dados de predição
            df_predict_clean = processor_train.clean_data(df_predict)
            
            if len(df_predict_clean) == 0:
                print(f"[ERRO] Sem dados validos")
                continue
            
            # Preparar features para predição
            X_predict = df_predict_clean[COLS_CATEGORICAS + COLS_TEMPO + COLS_NUMERICAS].copy()
            y_predict_real = df_predict_clean[TARGET].copy()
            
            # Aplicar transformações de treino
            X_predict_transformed = X_predict.copy()
            
            for col in COLS_CATEGORICAS:
                if col in X_predict_transformed.columns and col in encoders_dict:
                    encoded_values = []
                    for val in X_predict_transformed[col]:
                        try:
                            encoded = encoders_dict[col].transform([val])[0]
                        except:
                            encoded = encoders_dict[col].transform([encoders_dict[col].classes_[0]])[0]
                        encoded_values.append(encoded)
                    X_predict_transformed[col] = encoded_values
            
            # Escalar features numéricas
            X_predict_transformed[COLS_NUMERICAS] = scaler.transform(
                X_predict_transformed[COLS_NUMERICAS]
            )
            
            # Fazer predições
            y_predict = model.predict(X_predict_transformed)
            
            # Comparar resultados
            errors = np.abs(y_predict - y_predict_real.values)
            accuracies = 100 * (1 - errors / np.abs(y_predict_real.values))
            
            mean_accuracy = np.mean(accuracies)
            mean_error = np.mean(errors)
            
            print(f"[OK] R2={model_r2:.4f} | Acuracia={mean_accuracy:.2f}% | Erro=R${mean_error:.2f}")
            
            # Detalhe por ação
            print(f"\n   Acoes:")
            
            for idx, (row_idx, row) in enumerate(df_predict_clean.iterrows()):
                company = row.get('nameCompany', 'N/A')
                real = row[TARGET]
                pred = y_predict[idx]
                err = errors[idx]
                acc = accuracies[idx]
                
                print(f"   {company:10} | Real=R${real:7.2f} | Pred=R${pred:7.2f} | "
                      f"Erro=R${err:6.2f} | Acuracia={acc:6.2f}%")
                
                # Guardar no resultado
                analysis_results.append({
                    'training_date': str(training_date),
                    'prediction_date': str(prediction_date),
                    'company': company,
                    'real_price': float(real),
                    'predicted_price': float(pred),
                    'error_reais': float(err),
                    'accuracy_percent': float(acc),
                    'model_r2': float(model_r2)
                })
        
        except Exception as e:
            print(f"[ERRO] {str(e)[:80]}")
            continue
    
    # Salvar resultados
    output_file = Path('analysis_consecutive_days.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n[OK] Analise concluida!")
    print(f"[ARQUIVO] Salvos em: {output_file}")
    print(f"[TOTAL] Total: {len(analysis_results)}")
    
    # Estatísticas
    if analysis_results:
        df_results = pd.DataFrame(analysis_results)
        print(f"\n[STATS] ESTATISTICAS:")
        print(f"   Acuracia Media: {df_results['accuracy_percent'].mean():.2f}%")
        print(f"   Acuracia Min: {df_results['accuracy_percent'].min():.2f}%")
        print(f"   Acuracia Max: {df_results['accuracy_percent'].max():.2f}%")
        print(f"   Erro Medio: R$ {df_results['error_reais'].mean():.2f}")


if __name__ == "__main__":
    analyze_consecutive_days()
