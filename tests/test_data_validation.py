"""
Unit tests for data validation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from scripts.data_validation import DataValidator
from config import Config

class TestDataValidator:
    """Test cases for DataValidator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def validator(self, config):
        """Create DataValidator instance."""
        return DataValidator(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'job': np.random.choice(['admin', 'technician', 'services', 'management'], n_samples),
            'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
            'education': np.random.choice(['primary', 'secondary', 'tertiary'], n_samples),
            'balance': np.random.normal(1000, 5000, n_samples),
            'housing': np.random.choice(['yes', 'no'], n_samples),
            'loan': np.random.choice(['yes', 'no'], n_samples),
            'y': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        return pd.DataFrame(data)
    
    def test_validator_initialization(self, validator):
        """Test DataValidator initialization."""
        assert validator is not None
        assert hasattr(validator, 'config')
        assert hasattr(validator, 'data_dir')
        assert hasattr(validator, 'reports_dir')
    
    def test_basic_validation_no_issues(self, validator, sample_data, tmp_path):
        """Test basic validation with clean data."""
        # Set up test data
        validator.train_data = sample_data
        
        # Run validation
        results = validator.basic_validation()
        
        # Check results structure
        assert 'timestamp' in results
        assert 'checks' in results
        assert 'missing_values' in results['checks']
        assert 'duplicates' in results['checks']
        assert 'data_types' in results['checks']
        assert 'target_distribution' in results['checks']
        assert 'outliers' in results['checks']
        
        # Check specific results
        assert results['checks']['missing_values']['total_missing'] == 0
        assert results['checks']['duplicates']['duplicate_rows'] == 0
        assert len(results['checks']['target_distribution']) == 2  # 0 and 1
    
    def test_basic_validation_with_missing_values(self, validator, sample_data):
        """Test basic validation with missing values."""
        # Add missing values
        sample_data.loc[0, 'age'] = np.nan
        sample_data.loc[1, 'balance'] = np.nan
        validator.train_data = sample_data
        
        # Run validation
        results = validator.basic_validation()
        
        # Check missing values
        assert results['checks']['missing_values']['total_missing'] == 2
        assert 'age' in results['checks']['missing_values']['columns_with_missing']
        assert 'balance' in results['checks']['missing_values']['columns_with_missing']
    
    def test_basic_validation_with_duplicates(self, validator, sample_data):
        """Test basic validation with duplicate rows."""
        # Add duplicate row
        duplicate_row = sample_data.iloc[0].copy()
        sample_data = pd.concat([sample_data, pd.DataFrame([duplicate_row])], ignore_index=True)
        validator.train_data = sample_data
        
        # Run validation
        results = validator.basic_validation()
        
        # Check duplicates
        assert results['checks']['duplicates']['duplicate_rows'] == 1
    
    def test_basic_validation_with_outliers(self, validator, sample_data):
        """Test basic validation with outliers."""
        # Add outliers
        sample_data.loc[0, 'balance'] = 1000000  # Extreme outlier
        sample_data.loc[1, 'age'] = 150  # Outlier
        validator.train_data = sample_data
        
        # Run validation
        results = validator.basic_validation()
        
        # Check outliers
        assert results['checks']['outliers']['balance'] > 0
        assert results['checks']['outliers']['age'] > 0
    
    def test_create_validation_report(self, validator, sample_data, tmp_path):
        """Test validation report creation."""
        # Set up test data
        validator.train_data = sample_data
        validator.reports_dir = tmp_path
        
        # Run validation
        results = validator.basic_validation()
        
        # Create report
        report_path = validator.create_validation_report(results)
        
        # Check report file exists
        assert report_path.exists()
        assert report_path.suffix == '.html'
        
        # Check report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert 'Data Validation Report' in content
            assert 'Missing Values' in content
            assert 'Duplicates' in content
            assert 'Target Distribution' in content
    
    def test_run_validation_pipeline(self, validator, sample_data, tmp_path, monkeypatch):
        """Test complete validation pipeline."""
        # Mock data loading
        def mock_load_data():
            validator.train_data = sample_data
            validator.test_data = None
        
        monkeypatch.setattr(validator, 'load_data', mock_load_data)
        validator.reports_dir = tmp_path
        
        # Run validation pipeline
        results, report_path = validator.run_validation()
        
        # Check results
        assert results is not None
        assert report_path is not None
        assert report_path.exists()

if __name__ == "__main__":
    pytest.main([__file__])
