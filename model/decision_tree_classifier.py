from sklearn.tree import DecisionTreeClassifier
from datapreprocessing import fetch_processed_data
from evaluate import fetch_metrics

X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = fetch_processed_data()

model= DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

print(fetch_metrics(model,X_test,y_test))