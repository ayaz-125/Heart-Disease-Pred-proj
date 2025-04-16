import numpy as np
import pandas as pd
import os
import yaml
from sklearn.preprocessing import StandardScaler
from src.logger import logging
import joblib


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, columns_to_scale: list):
    """
    Apply StandardScaler to selected numerical columns.
    """
    scaler = StandardScaler()
    
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])

    return train_df, test_df, scaler


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params('params.yaml')
        columns_to_scale = params['feature_engineering']['columns_to_scale']

        train_data = load_data('./data_fol/interim/train_processed.csv')
        test_data = load_data('./data_fol/interim/test_processed.csv')

        train_df, test_df, scaler = scale_features(train_data, test_data, columns_to_scale)

        save_data(train_df, os.path.join("./data_fol", "processed", "train_scaled.csv"))
        save_data(test_df, os.path.join("./data_fol", "processed", "test_scaled.csv"))

        
        scaler_path = os.path.join("./models", "scaler.pkl")
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info('StandardScaler object saved to %s', scaler_path)

    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
