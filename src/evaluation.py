from sklearn import metrics
import matplotlib.pyplot as plt
from pathlib import Path
import logging

def eval_metrics(actual, pred, model_name):
    '''
    Docstring for eval_metrics
    
    :param actual: array with the true y data
    :param pred: array with the predicted y data
    :param model_name: string representing the name of the current model being used for calculating metrics
    '''
    logging.info(f"Calculating evaluation metrics for {model_name}...")
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    logging.info(f"Calculating evaluation metrics for {model_name} has been finished successfully")
    return accuracy, f1, auc

def plot_roc_curve(actual, pred, output_dir, model_name):
    '''
    Docstring for plot_roc_curve
    
    :param actual: array with the true y data
    :param pred: array with the predicted y data
    :output_dir: path of the ouput roc figure
    :param model_name: string representing the name of the current model being used for calculating metrics
    '''
    logging.info(f"Plotting ROC curve of {model_name}...")
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    plt.title(f"ROC Curve for {model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"ROC_curve_{model_name}.png")
    plt.close()
    logging.info(f"Plotting ROC curve of {model_name} has been finished successfully")


def plot_confusion_matrix(actual, pred, output_dir: Path, model_name, labels):
    '''
    Docstring for plot_roc_curve
    
    :param actual: array with the true y data
    :param pred: array with the predicted y data
    :output_dir: path of the ouput roc figure
    :param model_name: string representing the name of the current model being used for calculating metrics
    :param labels: model.classes_

    '''
    logging.info(f"Plotting Confusion Matrix for {model_name}...")
    conf_mat = metrics.confusion_matrix(actual, pred, labels=labels)
    conf_mat_disp = metrics.ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=labels
    )
    conf_mat_disp.plot()

    plt.title(f"Confusion Matrix for {model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"ROC_curve_{model_name}.png")
    plt.close()
    logging.info(f"Plotting Confusion Matrix of {model_name} has been finished successfully")