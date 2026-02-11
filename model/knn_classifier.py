from sklearn.neighbors import KNeighborsClassifier 
from datapreprocessing import fetch_processed_data
from evaluate import fetch_metrics


def get_knn_classifier_model(X_train_scaled, y_train):
    #X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = fetch_processed_data()
    model=KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    return model


#X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = fetch_processed_data()


#model.fit(X_train_scaled,y_train)

#print(fetch_metrics(model,X_test_scaled,y_test))