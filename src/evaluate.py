import joblib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from preprocess import load_data, preprocess

def evaluate_model(model, X_test, y_test):

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Core Metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print("=" * 40)
    print("        MODEL EVALUATION RESULTS")
    print("=" * 40)
    print(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall    : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print("=" * 40)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Malignant', 'Benign']))

    # ─── Save metrics to JSON for app.py ───────────────────────
    os.makedirs('outputs', exist_ok=True)
    metrics = {
        "accuracy":  round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall * 100, 2),
        "f1":        round(f1 * 100, 2)
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\nMetrics saved to outputs/metrics.json ✅")
    print(f"  → {metrics}")

    return y_pred, y_prob, accuracy, precision, recall, f1


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Benign'],
                yticklabels=['Malignant', 'Benign'])
    plt.title('Confusion Matrix - CanCure AI', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    print("Saved: outputs/confusion_matrix.png")
    plt.show()
    plt.close()


def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - CanCure AI', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.savefig('outputs/roc_curve.png', dpi=150)
    print(f"Saved: outputs/roc_curve.png")
    print(f"AUC Score: {roc_auc:.4f}")
    plt.show()
    plt.close()


def plot_feature_importance(model):
    from sklearn.datasets import load_breast_cancer
    feature_names = load_breast_cancer().feature_names

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 7))
    sns.barplot(x=importances[indices],
                y=[feature_names[i] for i in indices],
                palette='viridis')
    plt.title('Top 15 Feature Importances - CanCure AI',
              fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()

    plt.savefig('outputs/feature_importance.png', dpi=150)
    print("Saved: outputs/feature_importance.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # Load model and data
    model = joblib.load('outputs/model.pkl')
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)

    # Evaluate + save metrics.json
    y_pred, y_prob, acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)

    # Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_feature_importance(model)

    print("\n✅ All evaluation outputs saved to outputs/ folder!")
    print("✅ metrics.json ready for app.py!")
