# loan_approval_prediction
ML package and CLI files for the Loan Approval use case

A modular, installable, config-driven machine learning pipeline to train and predict loan approval decisions using Logistic Regression.

Built with proper src/ layout, CLI interface, YAML configuration, and clean separation of concerns.

# 🚀 Features

1. Installable Python package (pyproject.toml)
2. src/ layout (production-ready packaging structure)
3. CLI-based training (loan-train)
4. CLI-based inference (loan-predict)
5. YAML-driven configuration
6. Configurable probability threshold
7. Model artifacts persistence
8. Modular architecture (data, features, models, config, utils)


# Project Structure

loan-approval-prediction/
│
├── pyproject.toml
├── config.yaml
├── README.md
├── LICENSE
├── src/
│   └── loan_approval/
│       ├── cli.py
│       ├── config.py
│       ├── data/
│       ├── features/
│       ├── models/
│       └── utils/


# Installation

Step 1: Clone the repo and make the root as working directory. Use:
```
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
```
Step 2: Create a new virtual environment and activate it
Step 3: Install the package in editable mode using:
```
pip install -e .
```
Runtime behaviour controlled by config.yaml file in the root directory
Data Source: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=train_u6lujuX_CVtuZ9i.csv

## Training
Post installations, train using the command:
```
loan-train --data <relative path to train datset> --config config.yaml
```
Outputs:
1. Logs for model performance
2. Saved train model at artifacts/logistic_regression_model.joblib

## Inference

Use sample.json for inference, or create it from the test data available at the source. 
Run
```
loan-predict --model artifats/logistic_regression_model.joblib --config config.yaml --input sample.json
```

Output will be a dictionary

# Design Principles
1. Config-driven ML pipeline
2. Clean dependency injection
3. No hardcoded hyperparameters
4. Separation of model prediction and business decision logic
5. Production-style packaging

License: This project is licensed under the MIT License.

Author: Vanad Narayane


