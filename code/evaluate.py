import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def get_metrics(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }

def evaluate_all_models(y_test, predictions_dict):
    metrics_summary = []
    for name, pred in predictions_dict.items():
        metrics_summary.append(get_metrics(y_test, pred, name))
    
    summary_df = pd.DataFrame(metrics_summary)
    return summary_df

def plot_confusion_matrices(y_test, predictions_dict):
    fig, axes = plt.subplots(1, len(predictions_dict), figsize=(6 * len(predictions_dict), 5))
    if len(predictions_dict) == 1:
        axes = [axes]
    
    for i, (name, pred) in enumerate(predictions_dict.items()):
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Vacant', 'Occupied'],
                    yticklabels=['Vacant', 'Occupied'])
        axes[i].set_title(f"{name} Confusion Matrix")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    plt.tight_layout()
    plt.show()