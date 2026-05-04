# Classes para treinar Regressão e NN
import joblib
from sklearn.linear_model import LinearRegression
from .config import Config

class BaseTrainer:
    """Classe base para treinadores de modelos."""
    def __init__(self):
        self.model = None

    def save_model(self, name):
        path = Config.MODEL_DIR / name
        if name.endswith('.pkl'):
            joblib.dump(self.model, path)
        elif name.endswith('.h5') or name.endswith('.keras'):
            self.model.save(path)
        print(f"Modelo salvo em: {path}")

class LinearRegressionTrainer(BaseTrainer):
    def train(self, X_train, y_train):
        print("Treinando Regressão Linear...")
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        return self.model

class NeuralNetworkTrainer(BaseTrainer):
    def build_model(self, input_dim):
        model = Sequential([
            Dense(64, input_dim=input_dim, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss='mse', metrics=['mse'])
        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val):
        print("Treinando Rede Neural...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            verbose=1
        )
        return history