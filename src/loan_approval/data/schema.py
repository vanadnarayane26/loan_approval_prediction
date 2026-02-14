TARGET_COLUMN = "Loan_Status"
NUMERICAL_FEATURES = ["ApplicantIncome",
                      "CoapplicantIncome",
                      "LoanAmount",
                      "Loan_Amount_Term"]

CATEGORICAL_FEATURES = ["Gender",
                        "Married",
                        "Dependents",
                        "Education",
                        "Self_Employed",
                        "Credit_History",
                        "Property_Area"]

DROP_COLUMNS = ["Loan_ID"]