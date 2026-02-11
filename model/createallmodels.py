import joblib
from logistic_regression import get_logistic_regression_model
from datapreprocessing import fetch_processed_data
from decision_tree_classifier import get_dt_model
from random_forest import get_random_forest_model
from knn_classifier import get_knn_classifier_model
from naive_bayes_classifier import get_naive_bayes_classifier_model
from xgboost_model import get_xgboost_model
from evaluate import fetch_metrics
import csv
from pathlib import Path




create_model = {    
    "logistic_regression": get_logistic_regression_model,
    "knn": get_knn_classifier_model,
    "decision_tree": get_dt_model,
    "random_forest": get_random_forest_model,
    "naive_bayes": get_naive_bayes_classifier_model,
    "xgboost": get_xgboost_model
}
metrics = {}
#X_train = X_test = X_train_scaled = X_test_scaled = y_train = y_test = None

def fetch_data():
    script_dir = Path(__file__).parent.parent
    print(script_dir)
    csv_path = script_dir / "data" /"initial"/ "bank-full.csv"
    #X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,scaler = fetch_processed_data(path=csv_path)  #(path="../data/initial/bank-full.csv")
    return fetch_processed_data(path=csv_path)
    

def createmodels(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,scaler):
    
    
    joblib.dump(scaler,f"model/pkl/scaler.pkl")
    for model_name, model_func in create_model.items():
        model = model_func(X_train_scaled=X_train_scaled, y_train=y_train)
        joblib.dump(model, f"model/pkl/{model_name}.pkl")
        metrics[model_name] = fetch_metrics(model, X_test_scaled, y_test)



def save_metrics_to_csv(metrics_dict, filename="model_metrics.csv"):
    model_names = [
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "naive_bayes",
        "xgboost"
    ]
    display_names = {
        "logistic_regression": "Logistic Regression",
        "knn": "KNN",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "naive_bayes": "Naive Bayes",
        "xgboost": "XGBoost"
    }
    metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["Model"] + metric_order)
        for model_key in model_names:
            model_metrics = metrics_dict.get(model_key, {})
            row = [display_names[model_key]]
            for metric in metric_order:
                value = model_metrics.get(metric, "--")
                row.append(str(value))
            writer.writerow(row)


def save_metrics_to_txt(metrics_dict, filename="model_metrics.txt"):
    model_names = [
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "naive_bayes",
        "xgboost"
    ]
    display_names = {
        "logistic_regression": "Logistic Regression",
        "knn": "KNN",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "naive_bayes": "Naive Bayes",
        "xgboost": "XGBoost"
    }
    metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

    with open(filename, "w") as txtfile:
        header = "|\t".join(["Model"] + metric_order ) 
        txtfile.write(header + "\n")
        for model_key in model_names:
            model_metrics = metrics_dict.get(model_key, {})
            row = [display_names[model_key]]
            row.append("|")
            for metric in metric_order:
                value = model_metrics.get(metric, "--")
                row.append(str(value))
                row.append("|")
            txtfile.write("\t".join(row) + "\n")


X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,scaler=fetch_data()
createmodels(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test,scaler)
#save_metrics_to_csv(metrics)
save_metrics_to_txt(metrics)

