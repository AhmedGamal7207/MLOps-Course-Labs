"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### TODO Import MLflow
import mlflow

import logging
from colorama import Fore, Style
import joblib
import os

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # TODO Log the transformer as an artifact
    col_transf_path = 'models/column_transformer_model.joblib'
    joblib.dump(col_transf, col_transf_path)
    mlflow.log_artifact(col_transf_path)


    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train, max_iter):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=max_iter)
    log_reg.fit(X_train, y_train)

    ### TODO Log the model with the input and output schema
    # TODO Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, y_train)
    
    # TODO Log model
    mlflow.sklearn.log_model(log_reg, "Logistic Regression", signature=signature, input_example=X_train[:2])

    ### TODO Log the data
    dataset = mlflow.data.from_pandas(X_train, name="Churn Prediction Training")
    mlflow.log_input(dataset, context="training")
    
    return log_reg

def eval_metrics(actual, pred):
    '''
    Docstring for eval_metrics
    This function takes two params
    
    :param actual: array of the actual y data
    :param pred: array of the predicted data by the model (y_hat)

    returns: (accuracy, f1_score)
    '''
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, pos_label=1)

    return(accuracy, f1)

def main():
    max_iter = 1000
    ### TODO Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")


    ### TODO set the experiment name
    os.environ["LOGNAME"] = "Ahmed Gamal"
    mlflow.set_experiment("Churn Prediction Experiment")


    ### TODO Start a new run and leave all the main function code as part of the experiment
    with mlflow.start_run(run_name="LR_basic") as run:
            
        df = pd.read_csv("dataset/Churn_Modelling.csv")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)

        ### TODO Log the max_iter parameter
        mlflow.log_param("max_iter", max_iter)

        model = train(X_train, y_train, max_iter)

        
        y_pred = model.predict(X_test)

        ### TODO Log metrics after calculating them
        accuracy, f1 = eval_metrics(y_test, y_pred)
        mlflow.log_metrics({"Accuracy": accuracy, "F1 Score": f1})

        ### TODO Log tag
        mlflow.set_tag("run_id", run.info.run_id)

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()

        plot_path = "./plots/confusion_matrix_lr.png"
        plt.savefig(plot_path)

        # TODO Log the image as an artifact in MLflow
        mlflow.log_artifact(plot_path)
        
        #plt.show()


if __name__ == "__main__":
    main()
