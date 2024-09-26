import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_predictive_model(data_path):
    data = pd.read_csv(data_path)
    X = data.drop('failure', axis=1)
    y = data['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'predictive_model.pkl')
    return model
