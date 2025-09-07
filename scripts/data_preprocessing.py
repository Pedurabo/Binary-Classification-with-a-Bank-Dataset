#!/usr/bin/env python3
"""
Data preprocessing script for the bank classification dataset.
This script handles data cleaning, missing value imputation, and feature preparation.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

class DataPreprocessor:
    """Data preprocessing class for bank classification dataset."""
    
    def __init__(self, config: Config, drop_id: bool = True, log1p_balance: bool = False, cap_pct: float = None):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)
        self.models_dir = Path(config.MODELS_DIR)
        self.preprocessors = {}
        self.drop_id = drop_id
        self.log1p_balance = log1p_balance
        self.cap_pct = cap_pct
        
    def load_data(self):
        """Load training and test datasets."""
        try:
            train_path = self.data_dir / "train.csv"
            test_path = self.data_dir / "test.csv"
            
            if not train_path.exists():
                raise FileNotFoundError(f"Training data not found at {train_path}")
            
            self.train_data = pd.read_csv(train_path)
            logger.info(f"Loaded training data: {self.train_data.shape}")
            
            if test_path.exists():
                self.test_data = pd.read_csv(test_path)
                logger.info(f"Loaded test data: {self.test_data.shape}")
            else:
                self.test_data = None
                logger.warning("Test data not found")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def handle_missing_values(self, data, is_training=True):
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        # Check for missing values
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        # Separate numerical and categorical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target column from numerical columns if present
        if 'y' in numerical_cols:
            numerical_cols.remove('y')
        
        # Handle numerical missing values
        if numerical_cols:
            if is_training:
                self.preprocessors['numerical_imputer'] = SimpleImputer(strategy='median')
                data[numerical_cols] = self.preprocessors['numerical_imputer'].fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.preprocessors['numerical_imputer'].transform(data[numerical_cols])
        
        # Handle categorical missing values
        if categorical_cols:
            if is_training:
                self.preprocessors['categorical_imputer'] = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = self.preprocessors['categorical_imputer'].fit_transform(data[categorical_cols])
            else:
                data[categorical_cols] = self.preprocessors['categorical_imputer'].transform(data[categorical_cols])
        
        return data
    
    def handle_outliers(self, data, is_training=True):
        """Handle outliers using IQR method."""
        logger.info("Handling outliers...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'y' in numerical_cols:
            numerical_cols.remove('y')
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        return data
    
    def encode_categorical_variables(self, data, is_training=True):
        """Encode categorical variables."""
        logger.info("Encoding categorical variables...")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            if is_training:
                self.preprocessors['label_encoders'] = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.preprocessors['label_encoders'][col] = le
            else:
                for col in categorical_cols:
                    if col in self.preprocessors['label_encoders']:
                        le = self.preprocessors['label_encoders'][col]
                        # Handle unseen categories
                        data[col] = data[col].astype(str)
                        data[col] = data[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                        data[col] = le.transform(data[col])
        
        return data

    def clean_features(self, data):
        """Apply dataset-specific cleaning/transforms before scaling.
        - Optionally drop non-informative identifiers (id)
        - Optionally cap heavy-tailed columns (duration, campaign)
        - Optionally log1p-transform balance
        """
        logger.info("Applying data cleaning transforms...")

        # Drop id if present
        if self.drop_id and 'id' in data.columns:
            data = data.drop(columns=['id'])

        # Cap tails
        if isinstance(self.cap_pct, float) and 0.0 < self.cap_pct < 1.0:
            for col in ['duration', 'campaign']:
                if col in data.columns:
                    cap_val = data[col].quantile(self.cap_pct)
                    data[col] = np.clip(data[col], None, cap_val)

        # Log transform balance
        if self.log1p_balance and 'balance' in data.columns:
            data['balance'] = np.log1p(data['balance'])

        return data
    
    def scale_numerical_features(self, data, is_training=True):
        """Scale numerical features."""
        logger.info("Scaling numerical features...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'y' in numerical_cols:
            numerical_cols.remove('y')
        
        if numerical_cols:
            if is_training:
                self.preprocessors['scaler'] = StandardScaler()
                data[numerical_cols] = self.preprocessors['scaler'].fit_transform(data[numerical_cols])
            else:
                data[numerical_cols] = self.preprocessors['scaler'].transform(data[numerical_cols])
        
        return data
    
    def save_preprocessors(self):
        """Save fitted preprocessors for later use."""
        preprocessors_path = self.models_dir / "preprocessors.joblib"
        joblib.dump(self.preprocessors, preprocessors_path)
        logger.info(f"Preprocessors saved to {preprocessors_path}")
    
    def save_processed_data(self, train_data, test_data=None):
        """Save processed data."""
        # Save processed training data
        train_processed_path = self.data_dir / "train_processed.csv"
        train_data.to_csv(train_processed_path, index=False)
        logger.info(f"Processed training data saved to {train_processed_path}")
        
        # Save processed test data if available
        if test_data is not None:
            test_processed_path = self.data_dir / "test_processed.csv"
            test_data.to_csv(test_processed_path, index=False)
            logger.info(f"Processed test data saved to {test_processed_path}")
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        # Load data
        self.load_data()
        
        # Process training data
        logger.info("Processing training data...")
        train_processed = self.train_data.copy()
        train_processed = self.handle_missing_values(train_processed, is_training=True)
        train_processed = self.handle_outliers(train_processed, is_training=True)
        train_processed = self.clean_features(train_processed)
        train_processed = self.encode_categorical_variables(train_processed, is_training=True)
        train_processed = self.scale_numerical_features(train_processed, is_training=True)
        
        # Process test data if available
        test_processed = None
        if self.test_data is not None:
            logger.info("Processing test data...")
            test_processed = self.test_data.copy()
            test_processed = self.handle_missing_values(test_processed, is_training=False)
            test_processed = self.handle_outliers(test_processed, is_training=False)
            test_processed = self.clean_features(test_processed)
            test_processed = self.encode_categorical_variables(test_processed, is_training=False)
            test_processed = self.scale_numerical_features(test_processed, is_training=False)
        
        # Save preprocessors and processed data
        self.save_preprocessors()
        self.save_processed_data(train_processed, test_processed)
        
        logger.info("Data preprocessing completed successfully")
        return train_processed, test_processed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data preprocessing with optional cleaning")
    parser.add_argument("--no-drop-id", action="store_true", help="Do not drop 'id' column if present")
    parser.add_argument("--log1p-balance", action="store_true", help="Apply log1p transform to 'balance'")
    parser.add_argument("--cap-pct", type=float, default=None, help="Cap 'duration' and 'campaign' at this percentile (e.g., 0.99)")
    return parser.parse_args()


def main():
    """Main function to run data preprocessing."""
    try:
        args = parse_args()
        config = Config()
        preprocessor = DataPreprocessor(
            config,
            drop_id=not args.no_drop_id,
            log1p_balance=args.log1p_balance,
            cap_pct=args.cap_pct,
        )
        train_processed, test_processed = preprocessor.run_preprocessing()
        
        print("Data preprocessing completed successfully!")
        print(f"Processed training data shape: {train_processed.shape}")
        if test_processed is not None:
            print(f"Processed test data shape: {test_processed.shape}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
