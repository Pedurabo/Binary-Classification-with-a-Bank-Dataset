#!/usr/bin/env python3
"""
Data validation script for the bank classification dataset.
This script validates the data quality and generates validation reports.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from utils.logging_config import setup_logging

# Setup logging
logger = setup_logging(__name__)

class DataValidator:
    """Data validation class for bank classification dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.DATA_DIR)
        self.reports_dir = Path(config.REPORTS_DIR) / "data_validation"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def basic_validation(self):
        """Perform basic data validation checks."""
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {}
        }
        
        # Check for missing values
        missing_values = self.train_data.isnull().sum()
        validation_results['checks']['missing_values'] = {
            'columns_with_missing': missing_values[missing_values > 0].to_dict(),
            'total_missing': missing_values.sum()
        }
        
        # Check for duplicates
        duplicates = self.train_data.duplicated().sum()
        validation_results['checks']['duplicates'] = {
            'duplicate_rows': int(duplicates)
        }
        
        # Check data types
        data_types = self.train_data.dtypes.to_dict()
        validation_results['checks']['data_types'] = data_types
        
        # Check target distribution
        if 'y' in self.train_data.columns:
            target_dist = self.train_data['y'].value_counts().to_dict()
            validation_results['checks']['target_distribution'] = target_dist
        
        # Check for outliers in numerical columns
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numerical_cols:
            if col != 'y':  # Skip target column
                Q1 = self.train_data[col].quantile(0.25)
                Q3 = self.train_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((self.train_data[col] < (Q1 - 1.5 * IQR)) | 
                               (self.train_data[col] > (Q3 + 1.5 * IQR))).sum()
                outliers[col] = int(outlier_count)
        
        validation_results['checks']['outliers'] = outliers
        
        return validation_results
    
    def create_validation_report(self, validation_results):
        """Create HTML validation report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .check {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; }}
                .error {{ background-color: #ffebee; border-left: 4px solid #f44336; }}
                .warning {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
                .success {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Validation Report</h1>
                <p>Generated on: {validation_results['timestamp']}</p>
            </div>
        """
        
        # Add validation checks
        for check_name, check_data in validation_results['checks'].items():
            html_content += f"""
            <div class="section">
                <h2>{check_name.replace('_', ' ').title()}</h2>
            """
            
            if check_name == 'missing_values':
                if check_data['total_missing'] == 0:
                    html_content += '<div class="check success">No missing values found</div>'
                else:
                    html_content += f'<div class="check warning">Total missing values: {check_data["total_missing"]}</div>'
                    if check_data['columns_with_missing']:
                        html_content += '<table><tr><th>Column</th><th>Missing Count</th></tr>'
                        for col, count in check_data['columns_with_missing'].items():
                            html_content += f'<tr><td>{col}</td><td>{count}</td></tr>'
                        html_content += '</table>'
            
            elif check_name == 'duplicates':
                if check_data['duplicate_rows'] == 0:
                    html_content += '<div class="check success">No duplicate rows found</div>'
                else:
                    html_content += f'<div class="check warning">Duplicate rows: {check_data["duplicate_rows"]}</div>'
            
            elif check_name == 'target_distribution':
                html_content += '<table><tr><th>Class</th><th>Count</th><th>Percentage</th></tr>'
                total = sum(check_data.values())
                for class_val, count in check_data.items():
                    percentage = (count / total) * 100
                    html_content += f'<tr><td>{class_val}</td><td>{count}</td><td>{percentage:.2f}%</td></tr>'
                html_content += '</table>'
            
            elif check_name == 'outliers':
                html_content += '<table><tr><th>Column</th><th>Outlier Count</th></tr>'
                for col, count in check_data.items():
                    html_content += f'<tr><td>{col}</td><td>{count}</td></tr>'
                html_content += '</table>'
            
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = self.reports_dir / "validation_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path
    
    def run_validation(self):
        """Run complete validation pipeline."""
        logger.info("Starting data validation...")
        
        # Load data
        self.load_data()
        
        # Perform validation
        validation_results = self.basic_validation()
        
        # Create report
        report_path = self.create_validation_report(validation_results)
        
        logger.info("Data validation completed successfully")
        return validation_results, report_path

def main():
    """Main function to run data validation."""
    try:
        config = Config()
        validator = DataValidator(config)
        validation_results, report_path = validator.run_validation()
        
        print(f"Validation completed. Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
