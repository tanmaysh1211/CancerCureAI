from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
cancer = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target  # 0 = malignant, 1 = benign

# Save as CSV
df.to_csv('data/breast_cancer.csv', index=False)
print("Dataset saved successfully!")
print(f"Shape: {df.shape}")
print(df.head())