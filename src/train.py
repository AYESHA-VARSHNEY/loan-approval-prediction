from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def split_data(df, target="Loan_Status", test_size=0.4, random_state=1):
    """Train-test split"""
    X = df.drop([target], axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_svm(X_train, y_train):
    """Train Support Vector Machine model"""
    model = SVC()
    model.fit(X_train, y_train)
    return model
