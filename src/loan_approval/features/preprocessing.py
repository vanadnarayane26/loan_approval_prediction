from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_approval.data.schema import (NUMERICAL_FEATURES,
                                       CATEGORICAL_FEATURES,
                                       DROP_COLUMNS)


def build_preprocessor():
    numerical_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy="median")),
                                        ("scaler", StandardScaler())])
    
    categorical_pipeline = Pipeline(steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                                          ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    
    preprocessor = ColumnTransformer(transformers = [("num", numerical_pipeline, NUMERICAL_FEATURES),
                                                   ("cat", categorical_pipeline, CATEGORICAL_FEATURES)],
                                   remainder="drop")
    return preprocessor