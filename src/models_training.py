from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import logging


def train_classification_grid(model, param_grid, model_name, X_train, y_train):
    """
    Train a GridSearch using a classification model.

    Args:
        model (sklearn Model): Model that you want to train eg: LogisticRegression
        param_grid (dict): Dictionary with the grid search parameters
        model_name (str): String representing the name of the model used in logging
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        model: trained classifation model resulted from the GridSearch
    """
    logging.info(f"Starting Training with GridSearch for {model_name} model...")
    grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=0)
    model = grid.fit(X_train, y_train)
    logging.info(f"Training {model_name} model has been finished successfully...")
    return model

def train_logistic_regression(X_train, y_train):
    '''
    Docstring for train_logistic_regression
    
    :param X_train: The training data features
    :param y_train: The training data labels

    :returns trained grid search logstic regression model
    '''

    lr = LogisticRegression(random_state=6)
    param_grid = {
        'C': [10, 0.1, 0.01],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    return train_classification_grid(lr, param_grid, "Logistic Regression", X_train, y_train)

def train_random_forest(X_train, y_train):
    '''
    Docstring for train_random_forest
    
    :param X_train: The training data features
    :param y_train: The training data labels

    :returns trained grid search random forest model
    '''

    rf = RandomForestClassifier(random_state=6)
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [10, 20],
        'criterion': ["gini", "entropy"],
        'max_leaf_nodes': [50, 100]
    }
    return train_classification_grid(rf, param_grid, "Random Forest", X_train, y_train)

def train_decision_tree(X_train, y_train):
    '''
    Docstring for train_decision_tree
    
    :param X_train: The training data features
    :param y_train: The training data labels

    :returns trained grid search decision tree model
    '''

    dt = DecisionTreeClassifier(random_state=6)
    param_grid = {
        "max_depth": [3, 5, 7, 9],
        'criterion': ["gini", "entropy"]
    }
    return train_classification_grid(dt, param_grid, "Decision Tree", X_train, y_train)
