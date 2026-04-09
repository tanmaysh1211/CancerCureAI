import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import os

os.makedirs('outputs', exist_ok=True)

def plot_target_distribution():
    cancer = load_breast_cancer()
    labels = ['Malignant (0)', 'Benign (1)']
    counts = [np.sum(cancer.target == 0), np.sum(cancer.target == 1)]
    colors = ['#e74c3c', '#2ecc71']

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, counts, color=colors, edgecolor='black', width=0.5)
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5,
                 str(count), ha='center', fontsize=13, fontweight='bold')

    plt.title('Target Distribution - Malignant vs Benign',
              fontsize=15, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.ylim(0, max(counts) + 50)
    plt.tight_layout()
    plt.savefig('outputs/target_distribution.png', dpi=150)
    print("Saved: outputs/target_distribution.png")
    plt.show()

def plot_correlation_heatmap():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    plt.figure(figsize=(16, 12))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
                linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap - CanCure AI',
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=150)
    print("Saved: outputs/correlation_heatmap.png")
    plt.show()

def plot_feature_distributions():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target

    # Plot top 6 most important features
    top_features = [
        'mean radius', 'mean texture', 'mean perimeter',
        'mean area', 'mean smoothness', 'mean concavity'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        malignant = df[df['target'] == 0][feature]
        benign    = df[df['target'] == 1][feature]

        axes[i].hist(malignant, bins=30, alpha=0.6,
                     color='#e74c3c', label='Malignant', edgecolor='black')
        axes[i].hist(benign, bins=30, alpha=0.6,
                     color='#2ecc71', label='Benign', edgecolor='black')
        axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Value', fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].legend()

    plt.suptitle('Feature Distributions - Malignant vs Benign',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/feature_distributions.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/feature_distributions.png")
    plt.show()

def plot_boxplots():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})

    top_features = ['mean radius', 'mean area', 'mean concavity', 'mean texture']

    fig, axes = plt.subplots(1, 4, figsize=(16, 6))

    for i, feature in enumerate(top_features):
        sns.boxplot(x='diagnosis', y=feature, data=df,
                    palette={'Malignant': '#e74c3c', 'Benign': '#2ecc71'},
                    ax=axes[i])
        axes[i].set_title(f'{feature}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')

    plt.suptitle('Boxplots - Key Features by Diagnosis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/boxplots.png', dpi=150)
    print("Saved: outputs/boxplots.png")
    plt.show()

if __name__ == "__main__":
    print("Generating all visualizations...\n")
    plot_target_distribution()
    plot_correlation_heatmap()
    plot_feature_distributions()
    plot_boxplots()
    print("\nAll visualizations saved to outputs/ folder!")