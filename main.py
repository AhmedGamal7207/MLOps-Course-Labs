from src.data_preprocessing import preprocess
from src.models_training import train_random_forest, train_logistic_regression, train_decision_tree
from src.mlflow_logging import log_model_with_mlflow, setup_mlflow_experiment
from pathlib import Path
import logging
from colorama import Fore, Style

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )

def main():
    setup_logging()
    logging.info("Starting Churn Prediction Experiment Multimodels...")
    experiment_id = setup_mlflow_experiment("Churn_Prediction_Experiment_Multimodels")
    BASE_DIR = Path(__file__).resolve().parent
    
    output_dir = BASE_DIR / "plots"
    
    X_train, X_test, y_train, y_test = preprocess()
    
    rf_model = train_random_forest(X_train, y_train)
    log_model_with_mlflow(rf_model, X_test, y_test, "RandomForestClassifier", experiment_id, output_dir)
    
    lr_model = train_logistic_regression(X_train, y_train)
    log_model_with_mlflow(lr_model, X_test, y_test, "LogisticRegression", experiment_id, output_dir)
    
    dt_model = train_decision_tree(X_train, y_train)
    log_model_with_mlflow(dt_model, X_test, y_test, "DecisionTreeClassifier", experiment_id, output_dir)
    
    logging.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()