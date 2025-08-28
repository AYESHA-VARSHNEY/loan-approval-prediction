from sklearn import metrics

def evaluate(model, X_train, y_train, X_test, y_test):
    """Evaluate accuracy on train and test data"""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = 100 * metrics.accuracy_score(y_train, y_pred_train)
    test_acc = 100 * metrics.accuracy_score(y_test, y_pred_test)

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    return train_acc, test_acc
