import numpy as np
import pandas as pd
import pickle
import mlflow
import mlflow.pyfunc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import yaml
from src.logger import logging
import joblib
import os
from src.custom_model_wrapper import CustomModelWrapper  # Custom wrapper for MLflow

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    try:
        clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=5)
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info('Model evaluation completed with accuracy: %.4f', accuracy)
        print(f'Model Accuracy: {accuracy:.4f}')
        return accuracy
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def main():
    try:
        train_data = load_data('./data_fol/processed/train_scaled.csv')
        test_data = load_data('./data_fol/processed/test_scaled.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        accuracy = evaluate_model(clf, X_test, y_test)

        # Save model and scaler locally
        model_path = 'models/model.pkl'
        scaler_path = 'models/scaler.pkl'  # Ensure this file exists
        joblib.dump(clf, model_path)

        if not os.path.exists(scaler_path):
            logging.warning('Scaler.pkl not found at expected path.')
            return

        # Start MLflow run
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)

            artifacts = {
                "model": model_path,
                "scaler": scaler_path
            }

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=CustomModelWrapper(),
                artifacts=artifacts,
                code_path=["src"]
            )

            logging.info('Model and scaler logged as custom PyFunc model')

    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
