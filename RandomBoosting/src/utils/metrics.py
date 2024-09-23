from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)


def get_metrics(y_true, y_proba, y_pred):
    return {"ROC AUC": round(roc_auc_score(y_true, y_proba[:, 1]), 3),
            "Accuracy": round(accuracy_score(y_true, y_pred), 3),
            "f1": round(f1_score(y_true, y_pred), 3),
            "Log loss": round(log_loss(y_true, y_proba), 3),
            "Balanced accuracy": round(balanced_accuracy_score(y_true, y_pred), 3),
           }