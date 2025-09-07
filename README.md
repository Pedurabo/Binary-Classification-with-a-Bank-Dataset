# Bank Binary Classification - CI/CD Pipeline

This project implements a comprehensive CI/CD pipeline for the Kaggle Playground Series - Season 5, Episode 8: Binary Classification with a Bank Dataset.

## 🎯 Project Overview

The goal is to predict whether a client will subscribe to a bank term deposit using machine learning. The project includes a complete CI/CD pipeline with Jenkins for automated model training, evaluation, and deployment.

## 🏗️ Architecture

```
├── Jenkinsfile              # Main CI/CD pipeline
├── docker-compose.yml       # Local development environment
├── Dockerfile              # Application containerization
├── requirements.txt        # Python dependencies
├── scripts/               # Pipeline scripts
│   ├── data_validation.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_baseline.py
│   ├── train_advanced_models.py
│   ├── evaluate_models.py
│   ├── generate_predictions.py
│   └── performance_test.py
├── src/                   # Source code
│   ├── config.py         # Configuration management
│   ├── api/              # FastAPI application
│   ├── models/           # ML model implementations
│   ├── features/         # Feature engineering
│   └── utils/            # Utility functions
├── tests/                # Unit tests
├── monitoring/           # Monitoring configuration
└── data/                 # Dataset storage
```

## 🚀 CI/CD Pipeline Stages

### 1. **Checkout**
- Clones the repository from version control

### 2. **Setup Environment**
- Creates Python virtual environment
- Installs dependencies from `requirements.txt`

### 3. **Data Validation**
- Validates dataset quality and integrity
- Generates HTML validation reports
- Checks for missing values, duplicates, and outliers

### 4. **Data Preprocessing**
- Cleans and prepares the dataset
- Handles missing values and outliers
- Performs data type conversions

### 5. **Feature Engineering**
- Creates new features from existing data
- Applies feature selection techniques
- Handles categorical encoding

### 6. **Model Training** (Parallel)
- **Baseline Model**: Trains simple baseline models
- **Advanced Models**: Trains complex models (XGBoost, LightGBM, etc.)

### 7. **Model Evaluation**
- Evaluates models using multiple metrics
- Generates performance reports and visualizations
- Compares model performance

### 8. **Generate Predictions**
- Creates predictions for test dataset
- Formats output for Kaggle submission

### 9. **Build Docker Image** (Main branch only)
- Builds and tags Docker images
- Prepares for deployment

### 10. **Run Tests**
- Executes unit tests with coverage reporting
- Generates test coverage reports

### 11. **Security Scan**
- Runs security vulnerability scans
- Checks for known security issues

### 12. **Deploy to Staging** (Main branch only)
- Deploys to staging environment
- Performs integration testing

### 13. **Performance Testing** (Main branch only)
- Runs performance benchmarks
- Validates system performance

## 🛠️ Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Jenkins (for CI/CD pipeline)
- Python 3.9+
- Git

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bank-binary-classification
   ```

2. **Start local environment**
   ```bash
   docker-compose up -d
   ```

3. **Access services**
   - Application: http://localhost:8000
   - Jenkins: http://localhost:8080
   - MLflow: http://localhost:5000
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

### Jenkins Setup

1. **Install Jenkins plugins**
   - Pipeline
   - Git
   - Docker
   - HTML Publisher
   - Cobertura
   - Email Extension

2. **Configure Jenkins**
   - Set up Git credentials
   - Configure Docker access
   - Set up email notifications

3. **Create Jenkins job**
   - Create a new Pipeline job
   - Point to the `Jenkinsfile` in the repository
   - Configure webhook for automatic triggering

## 📊 Monitoring and Observability

### Metrics Collection
- **Prometheus**: Collects application metrics
- **Grafana**: Visualizes metrics and creates dashboards
- **MLflow**: Tracks ML experiments and model versions

### Logging
- Structured logging with JSON format
- Centralized log collection
- Log rotation and retention policies

### Health Checks
- Application health endpoints
- Database connectivity checks
- Model serving status

## 🔧 Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=development
DEBUG=True
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://user:password@db:5432/bank_classification

# Redis
REDIS_URL=redis://redis:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Logging
LOG_LEVEL=INFO
```

### Pipeline Configuration

The pipeline can be customized by modifying:
- `Jenkinsfile`: Pipeline stages and steps
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Service configuration
- `monitoring/prometheus.yml`: Metrics collection

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_validation.py -v
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v
```

### Performance Tests
```bash
# Run performance tests
python scripts/performance_test.py
```

## 📈 Model Performance

The pipeline tracks and reports on:
- **ROC AUC**: Primary evaluation metric
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced performance metric

## 🚀 Deployment

### Staging Deployment
- Automatic deployment on main branch
- Integration testing
- Performance validation

### Production Deployment
- Manual approval required
- Blue-green deployment strategy
- Rollback capabilities

## 🔒 Security

### Security Measures
- **Dependency scanning**: Automated vulnerability checks
- **Code analysis**: Static code analysis with Bandit
- **Container security**: Base image scanning
- **Access control**: Role-based access management

### Compliance
- Data privacy compliance
- Model explainability requirements
- Audit trail maintenance

## 📝 Documentation

- **API Documentation**: Auto-generated with FastAPI
- **Model Documentation**: MLflow experiment tracking
- **Pipeline Documentation**: Jenkins build logs and reports

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review Jenkins build logs

## 🔄 Pipeline Triggers

The pipeline is triggered by:
- **Push to main branch**: Full pipeline execution
- **Pull request**: Validation and testing only
- **Manual trigger**: On-demand execution
- **Scheduled**: Daily model retraining

## 📊 Pipeline Metrics

Track pipeline performance with:
- **Build success rate**: Percentage of successful builds
- **Build duration**: Time to complete pipeline
- **Test coverage**: Code coverage percentage
- **Security issues**: Number of security vulnerabilities
