import pandas as pd 

from loan_approval.models.model_io import load_model
from loan_approval.utils.logger import get_logger
from loan_approval.config import Config

logger = get_logger(__name__)


def predict_prob(model_path:str, config_path:str, input_data: dict):
    
    logger.info("Loading the model.....")
    model = load_model(model_path)
    
    config = Config(config_path)
    threshold = config.inference.get("threshold", 0.5)
    
    logger.info("Converting input data to DataFrame.....")
    input_df = pd.DataFrame([input_data])
    
    logger.info("Computing probability of loan approval.....")
    proba = model.predict_proba(input_df)[0][1]
    proba = round(proba, 2)
    
    logger.info(f"Probability: {proba}")
    decision = "Approved" if proba >= threshold else "Not Approved"
    logger.info(f"Loan Approval Decision: {decision}")  
    return {"probability": proba, "decision": decision, "threshold": threshold}
    