# Ponto de entrada (CLI do projeto)
from .data_ingestion import DataIngestor
from .preprocessing import DataProcessor
from .trainer import LinearRegressionTrainer, NeuralNetworkTrainer
from .config import Config
from sklearn.model_selection import train_test_split

def run_pipeline():
    # Ingestão
    ingestor = DataIngestor()
    df_raw = ingestor.load_and_merge()
    
    # Processamento
    processor = DataProcessor()
    df_clean = processor.clean_data(df_raw)
    X, y = processor.transform(df_clean)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )

    # Treino - Regressão Linear
    lr_trainer = LinearRegressionTrainer()
    lr_trainer.train(X_train, y_train)
    lr_trainer.save_model("linear_regression_model.pkl")

    # Treino - Rede Neural
    nn_trainer = NeuralNetworkTrainer()
    nn_trainer.build_model(input_dim=X_train.shape[1])
    nn_trainer.train(X_train, y_train, X_test, y_test)
    nn_trainer.save_model("neural_network_model.keras")

    print("\n✅ Pipeline finalizada com sucesso!")

if __name__ == "__main__":
    run_pipeline()