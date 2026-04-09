import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data():
    df = pd.read_csv('data/breast_cancer.csv')
    print(f"Dataset loaded: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    return df

def preprocess(df):
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']

    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"  0 = Malignant (Cancer), 1 = Benign (No Cancer)")

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for later use in app.py
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(scaler, 'outputs/scaler.pkl')
    print("Scaler saved to outputs/scaler.pkl")

    print(f"Train size: {X_train_scaled.shape}")
    print(f"Test size: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    print("Preprocessing complete!")