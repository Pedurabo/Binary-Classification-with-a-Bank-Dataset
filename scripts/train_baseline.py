#!/usr/bin/env python3
"""
Baseline model training script for the bank classification dataset.
This script trains simple baseline models for comparison.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

class BaselineTrainer:
    """Baseline model trainer for bank classification dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)
        self.models_dir = Path(config.MODELS_DIR)
        self.models = {}
        self.results = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        
    def load_data(self):
        """Load processed training data."""
        try:
            train_processed_path = self.data_dir / "train_processed.csv"
            
            if not train_processed_path.exists():
                raise FileNotFoundError(f"Processed training data not found at {train_processed_path}")
            
            self.data = pd.read_csv(train_processed_path)
            logger.info(f"Loaded processed training data: {self.data.shape}")
            
            # Separate features and target
            self.X = self.data.drop('y', axis=1)
            self.y = self.data['y']
            
            # Split data
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                self.X, self.y, 
                test_size=self.config.TEST_SIZE, 
                random_state=self.config.RANDOM_STATE,
                stratify=self.y
            )
            
            logger.info(f"Training set: {self.X_train.shape}")
            logger.info(f"Validation set: {self.X_val.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def define_baseline_models(self):
        """Define baseline models to train."""
        self.baseline_models = {
            'logistic_regression': LogisticRegression(
                random_state=self.config.RANDOM_STATE,
                max_iter=1000,
                n_jobs=self.config.N_JOBS
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.RANDOM_STATE,
                n_jobs=self.config.N_JOBS
            ),
            'svm': SVC(
                probability=True,
                random_state=self.config.RANDOM_STATE
            )
        }
        
        logger.info(f"Defined {len(self.baseline_models)} baseline models")
    
    def train_model(self, model_name, model):
        """Train a single model and log results."""
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=f"baseline_{model_name}"):
            # Log model parameters
            mlflow.log_params(model.get_params())
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_val)
            y_pred_proba = model.predict_proba(self.X_val)[:, 1]
            
            # Calculate metrics
            roc_auc = roc_auc_score(self.y_val, y_pred_proba)
            accuracy = accuracy_score(self.y_val, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=self.config.CV_FOLDS, 
                scoring='roc_auc',
                n_jobs=self.config.N_JOBS
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log metrics
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("cv_roc_auc_mean", cv_mean)
            mlflow.log_metric("cv_roc_auc_std", cv_std)
            
            # Log model
            mlflow.sklearn.log_model(model, f"baseline_{model_name}")
            
            # Store results
            self.results[model_name] = {
                'model': model,
                'roc_auc': roc_auc,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{model_name} - ROC AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}")
            logger.info(f"{model_name} - CV ROC AUC: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    def train_all_models(self):
        """Train all baseline models."""
        logger.info("Training all baseline models...")
        
        for model_name, model in self.baseline_models.items():
            try:
                self.train_model(model_name, model)
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
    
    def save_models(self):
        """Save trained models."""
        logger.info("Saving trained models...")
        
        for model_name, result in self.results.items():
            model_path = self.models_dir / f"baseline_{model_name}.joblib"
            joblib.dump(result['model'], model_path)
            logger.info(f"Saved {model_name} to {model_path}")
    
    def generate_baseline_report(self):
        """Generate baseline model comparison report."""
        logger.info("Generating baseline model report...")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'ROC AUC': result['roc_auc'],
                'Accuracy': result['accuracy'],
                'CV ROC AUC Mean': result['cv_mean'],
                'CV ROC AUC Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)
        
        # Save comparison report
        report_path = self.config.REPORTS_DIR / "baseline_model_comparison.csv"
        comparison_df.to_csv(report_path, index=False)
        logger.info(f"Baseline comparison report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("BASELINE MODEL COMPARISON")
        print("="*50)
        print(comparison_df.to_string(index=False))
        print("="*50)
        
        # Find best model
        best_model = comparison_df.iloc[0]
        print(f"\nBest baseline model: {best_model['Model']}")
        print(f"Best ROC AUC: {best_model['ROC AUC']:.4f}")
        
        return comparison_df
    
    def run_training(self):
        """Run complete baseline training pipeline."""
        logger.info("Starting baseline model training...")
        
        # Load data
        self.load_data()
        
        # Define models
        self.define_baseline_models()
        
        # Train models
        self.train_all_models()
        
        # Save models
        self.save_models()
        
        # Generate report
        comparison_df = self.generate_baseline_report()
        
        logger.info("Baseline model training completed successfully")
        return self.results, comparison_df

def main():
    """Main function to run baseline training."""
    try:
        config = Config()
        trainer = BaselineTrainer(config)
        results, comparison_df = trainer.run_training()
        
        print("\nBaseline training completed successfully!")
        print(f"Trained {len(results)} baseline models")
        
    except Exception as e:
        logger.error(f"Baseline training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
