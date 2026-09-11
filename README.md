# Customer Churn Prediction & ML Deployment

An end-to-end Machine Learning project that predicts whether a customer is likely to churn based on customer, service, contract, and billing information.

## 🚀 Project Overview

This project uses Machine Learning to predict customer churn and provides a REST API and web interface for real-time predictions.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Random Forest
- FastAPI
- Streamlit
- Docker
- Git & GitHub
- Render
- Streamlit Community Cloud

## 🏗️ Architecture

Dataset
↓
Data Preprocessing
↓
Feature Engineering
↓
Train/Test Split
↓
Random Forest Model
↓
Model Evaluation
↓
Saved ML Pipeline
↓
FastAPI REST API
↓
Streamlit Web Interface
↓
Docker Deployment

## 📊 Machine Learning

Model: Random Forest Classifier

The preprocessing pipeline includes:

- Missing value handling
- Numerical feature processing
- Categorical feature encoding
- One-Hot Encoding
- Train/Test Split

Model evaluation is performed using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

## ⚡ Features

- Customer churn prediction
- Churn probability
- Customer information analysis
- Billing information analysis
- REST API using FastAPI
- Interactive Streamlit interface
- Docker containerization
- Cloud deployment

## 🌐 Deployment

### FastAPI

Deployed using Render.

API Documentation:
https://customer-churn-ml-qcdw.onrender.com/docs

### Streamlit

The Streamlit application provides an interactive interface for entering customer details and receiving churn predictions.

## 🐳 Docker

Build Docker image:

```bash
docker build -t customer-churn-api .
## 📊 Model Evaluation

- Accuracy: 78%
- Algorithm: Random Forest Classifier
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

## 📈 Exploratory Data Analysis
## 📊 EDA Visualizations

### Customer Churn Distribution

![Customer Churn Distribution](reports/churn_distribution.png)

### Monthly Charges Distribution

![Monthly Charges Distribution](reports/monthly_charges.png)

### Customer Tenure Distribution

![Customer Tenure Distribution](reports/tenure_distribution.png)

### Churn by Contract Type

![Churn by Contract](reports/churn_by_contract.png)

### Churn by Internet Service

![Churn by Internet Service](reports/churn_by_internet_service.png)

### Churn by Payment Method

![Churn by Payment Method](reports/churn_by_payment_method.png)

### Monthly Charges vs Churn

![Monthly Charges vs Churn](reports/monthly_charges_vs_churn.png)

### Tenure vs Churn

![Tenure vs Churn](reports/tenure_vs_churn.png)

### Correlation Heatmap

![Correlation Heatmap](reports/correlation_heatmap.png)

### Feature Importance

![Feature Importance](reports/feature_importance.png)

The project includes EDA for:

- Customer Churn Distribution
- Monthly Charges Distribution
- Tenure Distribution
- Categorical Feature Analysis

## ⭐ Feature Importance

Feature importance analysis is performed using the trained Random Forest model to identify the most influential factors affecting customer churn.

## 🐳 Docker

Build the Docker image:

```bash
docker build -t customer-churn-api .