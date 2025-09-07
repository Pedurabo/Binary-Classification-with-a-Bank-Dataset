pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.9'
        PROJECT_NAME = 'bank-binary-classification'
        DOCKER_IMAGE = 'bank-classification-model'
        DOCKER_TAG = 'latest'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code from repository...'
                checkout scm
            }
        }
        
        stage('Setup Environment') {
            steps {
                echo 'Setting up Python environment...'
                script {
                    // Create virtual environment
                    sh '''
                        python -m venv venv
                        source venv/bin/activate
                        pip install --upgrade pip
                        pip install -r requirements.txt
                    '''
                }
            }
        }
        
        stage('Data Validation') {
            steps {
                echo 'Validating dataset...'
                script {
                    sh '''
                        source venv/bin/activate
                        python scripts/data_validation.py
                    '''
                }
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports/data_validation',
                        reportFiles: 'validation_report.html',
                        reportName: 'Data Validation Report'
                    ])
                }
            }
        }
        
        stage('Data Preprocessing') {
            steps {
                echo 'Preprocessing data...'
                script {
                    sh '''
                        source venv/bin/activate
                        python scripts/data_preprocessing.py
                    '''
                }
            }
        }
        
        stage('Feature Engineering') {
            steps {
                echo 'Performing feature engineering...'
                script {
                    sh '''
                        source venv/bin/activate
                        python scripts/feature_engineering.py
                    '''
                }
            }
        }
        
        stage('Model Training') {
            parallel {
                stage('Train Baseline Model') {
                    steps {
                        echo 'Training baseline model...'
                        script {
                            sh '''
                                source venv/bin/activate
                                python scripts/train_baseline.py
                            '''
                        }
                    }
                }
                
                stage('Train Advanced Models') {
                    steps {
                        echo 'Training advanced models...'
                        script {
                            sh '''
                                source venv/bin/activate
                                python scripts/train_advanced_models.py
                            '''
                        }
                    }
                }
            }
        }
        
        stage('Model Evaluation') {
            steps {
                echo 'Evaluating models...'
                script {
                    sh '''
                        source venv/bin/activate
                        python scripts/evaluate_models.py
                    '''
                }
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports/model_evaluation',
                        reportFiles: 'evaluation_report.html',
                        reportName: 'Model Evaluation Report'
                    ])
                    
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports/plots',
                        reportFiles: '*.html',
                        reportName: 'Model Performance Plots'
                    ])
                }
            }
        }
        
        stage('Generate Predictions') {
            steps {
                echo 'Generating predictions for test set...'
                script {
                    sh '''
                        source venv/bin/activate
                        python scripts/generate_predictions.py
                    '''
                }
            }
        }
        
        stage('Build Docker Image') {
            when {
                branch 'main'
            }
            steps {
                echo 'Building Docker image...'
                script {
                    sh '''
                        docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_IMAGE}:${BUILD_NUMBER}
                    '''
                }
            }
        }
        
        stage('Run Tests') {
            steps {
                echo 'Running unit tests...'
                script {
                    sh '''
                        source venv/bin/activate
                        python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=xml
                    '''
                }
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Test Coverage Report'
                    ])
                    
                    publishCobertura([
                        coberturaReportFile: 'coverage.xml'
                    ])
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                echo 'Running security scan...'
                script {
                    sh '''
                        source venv/bin/activate
                        safety check
                        bandit -r src/ -f html -o reports/security/bandit_report.html
                    '''
                }
            }
            post {
                always {
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports/security',
                        reportFiles: 'bandit_report.html',
                        reportName: 'Security Scan Report'
                    ])
                }
            }
        }
        
        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                echo 'Deploying to staging environment...'
                script {
                    // Deploy to staging environment
                    sh '''
                        echo "Deploying to staging..."
                        # Add your staging deployment commands here
                    '''
                }
            }
        }
        
        stage('Performance Testing') {
            when {
                branch 'main'
            }
            steps {
                echo 'Running performance tests...'
                script {
                    sh '''
                        source venv/bin/activate
                        python scripts/performance_test.py
                    '''
                }
            }
        }
    }
    
    post {
        always {
            echo 'Cleaning up workspace...'
            cleanWs()
        }
        
        success {
            echo 'Pipeline completed successfully!'
            script {
                // Send success notification
                emailext (
                    subject: "Pipeline Success: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                    body: "Pipeline completed successfully. View details at: ${env.BUILD_URL}",
                    recipientProviders: [[$class: 'DevelopersRecipientProvider']]
                )
            }
        }
        
        failure {
            echo 'Pipeline failed!'
            script {
                // Send failure notification
                emailext (
                    subject: "Pipeline Failed: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                    body: "Pipeline failed. View details at: ${env.BUILD_URL}",
                    recipientProviders: [[$class: 'DevelopersRecipientProvider']]
                )
            }
        }
        
        unstable {
            echo 'Pipeline is unstable!'
        }
    }
}
