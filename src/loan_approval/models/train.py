import json
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from loan_approval.data.loader import DataLoader
from loan_approval.data.schema import TARGET_COLUMN
from loan_approval.features.preprocessing import build_preprocessor
from loan_approval.utils.logger import get_logger
from loan_approval.models.model_io import save_model
from loan_approval.config import Config

logger = get_logger(__name__)

def train(data_path: str, config_path: str):
    
    loader = DataLoader(data_path)
    logger.info("Loading the dataset.....")
    df = loader.load()
    
    config = Config(config_path)
    model_config = config.model
    
    X = df.drop(TARGET_COLUMN, axis = 1)
    y = df[TARGET_COLUMN]
    
    logger.info("Splitting Dataset.....")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = config.data["test_size"], random_state = config.data["random_state"], stratify = y)
    
    logger.info("Preprocessing the data.....")
    preprocessor = build_preprocessor()
    classifier = LogisticRegression(**model_config)
    model = Pipeline(steps = [("preprocessor", preprocessor),("classifier", classifier)])
    
    logger.info("training the model.....")
    logger.info("TRAINING STARTED..... testing real time changes")

    model.fit(X_train, y_train)
    
    logger.info("Inference on validation data......")
    y_pred = model.predict(X_test)
    
    logger.info("Evaluating the data....")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    logger.info("Creating a classification report.....")
    report_dict = classification_report(y_test, y_pred, output_dict = True)
    logger.info("Classification report created successfully.....")
    
    with open(config.artifacts["report_path"], "w") as f:
        json.dump(report_dict, f, indent = 4)
    logger.info(f"Classification report saved successfully at {config.artifacts['report_path']}")
    
    logger.info("Saving the model.....")
    save_model(model, config.artifacts["model_path"])
    logger.info(f"Model saved successfully at {config.artifacts['model_path']}")
    