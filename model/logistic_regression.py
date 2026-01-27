
from sklearn.linear_model import LogisticRegression


lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_metrics = evaluate_model(lr, X_test_scaled, y_test)