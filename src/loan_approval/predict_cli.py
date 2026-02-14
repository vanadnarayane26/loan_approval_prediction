import argparse
import json
from loan_approval.models.predict import predict_prob
from loan_approval.utils.logger import get_logger
logger = get_logger(__name__)


def predict_cli():
    parser = argparse.ArgumentParser(description="Loan Approval Prediction")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data in JSON format",
    )
    args = parser.parse_args()
    with open(args.input, "r") as f:
        input_data = json.load(f)  
    result = predict_prob(args.model, args.config, input_data)
    logger.info(f"Prediction Result: {result}")