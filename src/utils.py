import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def save_metrics(report, filepath, y_test=None, y_pred=None):
    """
    Save model metrics to a file.
    """
    with open(filepath, 'w') as f:
        f.write("Classification Report:\n")
        f.write(f"F1-score (Class 1): {report['1']['f1-score']:.4f}\n")
        f.write(f"Precision (Class 1): {report['1']['precision']:.4f}\n")
        f.write(f"Recall (Class 1): {report['1']['recall']:.4f}\n")
        f.write(f"Accuracy: {report['accuracy']:.4f}\n")
        
        if y_test is not None and y_pred is not None:
            f.write("\nFull report:\n")
            f.write(classification_report(
                y_test, 
                y_pred, 
                target_names=['No Churn', 'Churn']
            ))

def plot_confusion_matrix(y_true, y_pred, filepath):
    """
    Plot and save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=['No Churn', 'Churn']
    )
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(filepath)
    plt.close()