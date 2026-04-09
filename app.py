import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CanCure AI",
    page_icon="🎗️",
    layout="wide"
)

# ─── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('outputs/model.pkl')
    scaler = joblib.load('outputs/scaler.pkl')
    return model, scaler

model, scaler = load_model()
cancer        = load_breast_cancer()
feature_names = cancer.feature_names

# ─── Header ────────────────────────────────────────────────────
st.title("🎗️ CanCure AI")
st.subheader("Breast Cancer Detection using Random Forest")
st.markdown("---")

# ─── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🔬 Predict",
    "📊 Model Performance",
    "📈 Visualizations",
    "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🔬 Predict":
    st.header("🔬 Cancer Prediction")
    st.write("Adjust the sliders below to input patient diagnostic data:")

    col1, col2, col3 = st.columns(3)
    input_data = {}

    for i, feature in enumerate(feature_names):
        min_val  = float(cancer.data[:, i].min())
        max_val  = float(cancer.data[:, i].max())
        mean_val = float(cancer.data[:, i].mean())

        if i % 3 == 0:
            with col1:
                input_data[feature] = st.slider(
                    feature, min_val, max_val, mean_val, key=feature)
        elif i % 3 == 1:
            with col2:
                input_data[feature] = st.slider(
                    feature, min_val, max_val, mean_val, key=feature)
        else:
            with col3:
                input_data[feature] = st.slider(
                    feature, min_val, max_val, mean_val, key=feature)

    st.markdown("---")

    if st.button("🔍 Predict", use_container_width=True):
        input_array  = np.array(list(input_data.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            if prediction == 1:
                st.success("✅ Prediction: **BENIGN**")
                st.info(f"Confidence: **{probability[1]*100:.2f}%**")
            else:
                st.error("⚠️ Prediction: **MALIGNANT**")
                st.info(f"Confidence: **{probability[0]*100:.2f}%**")

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(['Malignant', 'Benign'],
                    [probability[0], probability[1]],
                    color=['#e74c3c', '#2ecc71'])
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.set_title('Prediction Confidence')
            plt.tight_layout()
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.header("📊 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "96%",  "↑ High")
    col2.metric("Precision", "93%",  "↑ High")
    col3.metric("Recall",    "98%",  "↑ High")
    col4.metric("F1 Score",  "95%",  "↑ High")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        try:
            st.image('outputs/confusion_matrix.png')
        except:
            st.warning("Run evaluate.py first to generate this plot")

    with col_b:
        st.subheader("ROC Curve")
        try:
            st.image('outputs/roc_curve.png')
        except:
            st.warning("Run evaluate.py first to generate this plot")

    st.subheader("Feature Importance")
    try:
        st.image('outputs/feature_importance.png')
    except:
        st.warning("Run evaluate.py first to generate this plot")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.header("📈 Data Visualizations")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Target Distribution")
        try:
            st.image('outputs/target_distribution.png')
        except:
            st.warning("Run visualize.py first")

    with col_b:
        st.subheader("Feature Distributions")
        try:
            st.image('outputs/feature_distributions.png')
        except:
            st.warning("Run visualize.py first")

    st.subheader("Correlation Heatmap")
    try:
        st.image('outputs/correlation_heatmap.png')
    except:
        st.warning("Run visualize.py first")

    st.subheader("Boxplots")
    try:
        st.image('outputs/boxplots.png')
    except:
        st.warning("Run visualize.py first")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.header("ℹ️ About CanCure AI")

    st.markdown("""
    ## 🎗️ CanCure AI — Breast Cancer Detection

    **CanCure AI** is a machine learning web application that predicts
    whether a breast tumor is **Malignant** or **Benign** based on
    clinical diagnostic data.

    ---

    ### 🧠 Model Details
    - **Algorithm**: Random Forest Classifier
    - **Dataset**: Wisconsin Breast Cancer Dataset (569 samples, 30 features)
    - **Train/Test Split**: 80% / 20%
    - **Cross Validation**: 5-Fold

    ### 📊 Performance
    | Metric    | Score |
    |-----------|-------|
    | Accuracy  | 96%   |
    | Precision | 93%   |
    | Recall    | 98%   |
    | F1 Score  | 95%   |

    ### 🛠️ Tech Stack
    - Python, Scikit-learn, Pandas, NumPy
    - Matplotlib, Seaborn
    - Streamlit

    ---
    ### ⚠️ Disclaimer
    This tool is for **educational purposes only**.
    Always consult a medical professional for diagnosis.
    """)