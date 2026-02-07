import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
from pathlib import Path
from .evaluation import eval_metrics, plot_roc_curve, plot_confusion_matrix
import logging

import yaml
import os


def setup_mlflow_experiment(experiment_name, tracking_uri = "http://localhost:5000"):
    os.environ["LOGNAME"] = "Ahmed Gamal"
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id

def log_model_with_mlflow(model, X_test, y_test, model_name, exp_id, output_dir):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")

        mlflow.set_tag("model", model_name)

        pred = model.predict(X_test)
        accuracy, f1, auc = eval_metrics(y_test, pred, model_name)
        plot_roc_curve(y_test, pred, output_dir, model_name)
        plot_confusion_matrix(y_test, pred, output_dir, model_name, model.classes_)

        mlflow.log_params(model.best_params_)
        mlflow.log_metrics({
            "Mean CV score": model.best_score_,
            "Accuracy": accuracy,
            "f1-score": f1,
            "AUC": auc
        })

        mlflow.log_artifact(str(output_dir / f"ROC_curve_{model_name}.png"))

        # Logging Column Transformer as an artifact
        mlflow.log_artifact(get_col_transf_path())

        pd_dataset = mlflow.data.from_pandas(X_test, name="Churn Prediction Testing")
        mlflow.log_input(pd_dataset, context="Testing")

        signature = mlflow.models.infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_test[:2])


def get_col_transf_path():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Accessing values
    col_transf_path = config['col_transf_path']
    return col_transf_path