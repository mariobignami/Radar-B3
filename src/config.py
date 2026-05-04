# Configurações dinâmicas e caminhos
from pathlib import Path

class Config:
    # Caminhos dinâmicos com Pathlib
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_RAW = BASE_DIR / "data" / "raw"
    DATA_PROCESSED = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "models"

    # Garante que as pastas existam
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configurações do Dataset
    COLUNAS_INTERESSE = [
        'stockCodeCompany', 'sectorCompany', 'segmentCompany', 'dayTime', 'dayWeekTime', 
        'monthTime', 'yearTime', 'openValueStock', 'highValueStock', 
        'lowValueStock', 'quantityStock', 'valueCoin', 'closeValueStock'
    ]
    TARGET = 'closeValueStock'
    COLS_CATEGORICAS = ['stockCodeCompany', 'sectorCompany', 'segmentCompany']
    COLS_NUMERICAS = ['openValueStock', 'highValueStock', 'lowValueStock', 'quantityStock', 'valueCoin']

    # Hiperparâmetros
    TEST_SIZE = 0.3
    RANDOM_STATE = 99
    LEARNING_RATE = 0.001
    EPOCHS = 50
    BATCH_SIZE = 32