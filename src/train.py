import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from preprocess import load_data, preprocess

def train_model(X_train, y_train):
    # Initialize Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    # Cross validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Train the model
    model.fit(X_train, y_train)
    print("Model training complete!")

    # Save model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.pkl')
    print("Model saved to outputs/model.pkl")

    return model

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train_model(X_train, y_train)
    print("Training pipeline complete!")