from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from datapreprocessing  import fetch_processed_data
from evaluate import fetch_metrics

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = fetch_processed_data()

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

print(fetch_metrics(model,X_test_scaled,y_test))


# y_pred = model.predict(X_test_scaled)
# y_prob = model.predict_proba(X_test_scaled)[:, 1]

# print("Logistic Regression Metrics")
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("AUC:", roc_auc_score(y_test, y_prob))
# print("Precision:", precision_score(y_test, y_pred))
# print("Recall:", recall_score(y_test, y_pred))
# print("F1:", f1_score(y_test, y_pred))
# print("MCC:", matthews_corrcoef(y_test, y_pred))