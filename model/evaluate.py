from sklearn.metrics import (
        accuracy_score,roc_auc_score,precision_score,
        recall_score,f1_score,matthews_corrcoef,
        confusion_matrix,classification_report
)

def fetch_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "AUC": round(roc_auc_score(y_test, y_prob), 2),
        "Precision": round(precision_score(y_test, y_pred), 2),
        "Recall": round(recall_score(y_test, y_pred), 2),
        "F1": round(f1_score(y_test, y_pred), 2),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 2)
    }

    return metrics