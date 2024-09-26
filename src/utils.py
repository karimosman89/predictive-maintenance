import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the sensor data CSV."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the sensor data for predictive modeling."""
    X = data.drop('failure', axis=1)
    y = data['failure']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

