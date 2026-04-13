# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import load_breast_cancer

# # ─── Page Config ───────────────────────────────────────────────
# st.set_page_config(
#     page_title="CanCure AI",
#     page_icon="🎗️",
#     layout="wide"
# )

# # ─── Load Model & Scaler ───────────────────────────────────────
# @st.cache_resource
# def load_model():
#     model  = joblib.load('outputs/model.pkl')
#     scaler = joblib.load('outputs/scaler.pkl')
#     return model, scaler

# model, scaler = load_model()
# cancer        = load_breast_cancer()
# feature_names = cancer.feature_names

# # ─── Header ────────────────────────────────────────────────────
# st.title("🎗️ CanCure AI")
# st.subheader("Breast Cancer Detection using Random Forest")
# st.markdown("---")

# # ─── Sidebar ───────────────────────────────────────────────────
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", [
#     "🔬 Predict",
#     "📊 Model Performance",
#     "📈 Visualizations",
#     "ℹ️ About"
# ])

# # ══════════════════════════════════════════════════════════════
# # PAGE 1 — PREDICT
# # ══════════════════════════════════════════════════════════════
# if page == "🔬 Predict":
#     st.header("🔬 Cancer Prediction")
#     st.write("Adjust the sliders below to input patient diagnostic data:")

#     col1, col2, col3 = st.columns(3)
#     input_data = {}

#     for i, feature in enumerate(feature_names):
#         min_val  = float(cancer.data[:, i].min())
#         max_val  = float(cancer.data[:, i].max())
#         mean_val = float(cancer.data[:, i].mean())

#         if i % 3 == 0:
#             with col1:
#                 input_data[feature] = st.slider(
#                     feature, min_val, max_val, mean_val, key=feature)
#         elif i % 3 == 1:
#             with col2:
#                 input_data[feature] = st.slider(
#                     feature, min_val, max_val, mean_val, key=feature)
#         else:
#             with col3:
#                 input_data[feature] = st.slider(
#                     feature, min_val, max_val, mean_val, key=feature)

#     st.markdown("---")

#     if st.button("🔍 Predict", width='stretch'):
#         input_array  = np.array(list(input_data.values())).reshape(1, -1)
#         input_scaled = scaler.transform(input_array)
#         prediction   = model.predict(input_scaled)[0]
#         probability  = model.predict_proba(input_scaled)[0]

#         st.markdown("---")
#         col_a, col_b = st.columns(2)

#         with col_a:
#             if prediction == 1:
#                 st.success("✅ Prediction: **BENIGN**")
#                 st.info(f"Confidence: **{probability[1]*100:.2f}%**")
#             else:
#                 st.error("⚠️ Prediction: **MALIGNANT**")
#                 st.info(f"Confidence: **{probability[0]*100:.2f}%**")

#         with col_b:
#             fig, ax = plt.subplots(figsize=(5, 3))
#             ax.barh(['Malignant', 'Benign'],
#                     [probability[0], probability[1]],
#                     color=['#e74c3c', '#2ecc71'])
#             ax.set_xlim(0, 1)
#             ax.set_xlabel('Probability')
#             ax.set_title('Prediction Confidence')
#             plt.tight_layout()
#             st.pyplot(fig)

# # ══════════════════════════════════════════════════════════════
# # PAGE 2 — MODEL PERFORMANCE
# # ══════════════════════════════════════════════════════════════
# elif page == "📊 Model Performance":
#     st.header("📊 Model Performance Metrics")

#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Accuracy",  "96%",  "↑ High")
#     col2.metric("Precision", "93%",  "↑ High")
#     col3.metric("Recall",    "98%",  "↑ High")
#     col4.metric("F1 Score",  "95%",  "↑ High")

#     st.markdown("---")
#     col_a, col_b = st.columns(2)

#     with col_a:
#         st.subheader("Confusion Matrix")
#         try:
#             st.image('outputs/confusion_matrix.png')
#         except:
#             st.warning("Run evaluate.py first to generate this plot")

#     with col_b:
#         st.subheader("ROC Curve")
#         try:
#             st.image('outputs/roc_curve.png')
#         except:
#             st.warning("Run evaluate.py first to generate this plot")

#     st.subheader("Feature Importance")
#     try:
#         st.image('outputs/feature_importance.png')
#     except:
#         st.warning("Run evaluate.py first to generate this plot")

# # ══════════════════════════════════════════════════════════════
# # PAGE 3 — VISUALIZATIONS
# # ══════════════════════════════════════════════════════════════
# elif page == "📈 Visualizations":
#     st.header("📈 Data Visualizations")

#     col_a, col_b = st.columns(2)

#     with col_a:
#         st.subheader("Target Distribution")
#         try:
#             st.image('outputs/target_distribution.png')
#         except:
#             st.warning("Run visualize.py first")

#     with col_b:
#         st.subheader("Feature Distributions")
#         try:
#             st.image('outputs/feature_distributions.png')
#         except:
#             st.warning("Run visualize.py first")

#     st.subheader("Correlation Heatmap")
#     try:
#         st.image('outputs/correlation_heatmap.png')
#     except:
#         st.warning("Run visualize.py first")

#     st.subheader("Boxplots")
#     try:
#         st.image('outputs/boxplots.png')
#     except:
#         st.warning("Run visualize.py first")

# # ══════════════════════════════════════════════════════════════
# # PAGE 4 — ABOUT
# # ══════════════════════════════════════════════════════════════
# elif page == "ℹ️ About":
#     st.header("ℹ️ About CanCure AI")

#     st.markdown("""
#     ## 🎗️ CanCure AI — Breast Cancer Detection

#     **CanCure AI** is a machine learning web application that predicts
#     whether a breast tumor is **Malignant** or **Benign** based on
#     clinical diagnostic data.

#     ---

#     ### 🧠 Model Details
#     - **Algorithm**: Random Forest Classifier
#     - **Dataset**: Wisconsin Breast Cancer Dataset (569 samples, 30 features)
#     - **Train/Test Split**: 80% / 20%
#     - **Cross Validation**: 5-Fold

#     ### 📊 Performance
#     | Metric    | Score |
#     |-----------|-------|
#     | Accuracy  | 96%   |
#     | Precision | 93%   |
#     | Recall    | 98%   |
#     | F1 Score  | 95%   |

#     ### 🛠️ Tech Stack
#     - Python, Scikit-learn, Pandas, NumPy
#     - Matplotlib, Seaborn
#     - Streamlit

#     ---
#     ### ⚠️ Disclaimer
#     This tool is for **educational purposes only**.
#     Always consult a medical professional for diagnosis.
#     """)




















# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
# from sklearn.datasets import load_breast_cancer

# st.set_page_config(
#     page_title="CanCure AI",
#     page_icon="🎗️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ─── CUSTOM CSS ────────────────────────────────────────────────
# st.markdown("""
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

# html, body, [class*="css"] {
#     font-family: 'DM Sans', sans-serif;
# }

# /* Hide default streamlit header */
# #MainMenu, footer, header { visibility: hidden; }

# /* Page background */
# .stApp {
#     background: #0a0a0a;
# }

# /* Sidebar */
# [data-testid="stSidebar"] {
#     background: #111111 !important;
#     border-right: 1px solid #222;
# }
# [data-testid="stSidebar"] * {
#     color: #fff !important;
# }

# /* Hero banner */
# .hero {
#     background: linear-gradient(135deg, #FF3CAC, #784BA0, #2B86C5);
#     border-radius: 20px;
#     padding: 48px 40px;
#     margin-bottom: 32px;
# }
# .hero h1 {
#     font-family: 'Syne', sans-serif;
#     font-size: 56px;
#     font-weight: 800;
#     color: #fff;
#     margin: 0 0 8px 0;
#     line-height: 1.1;
# }
# .hero p {
#     font-size: 18px;
#     color: rgba(255,255,255,0.8);
#     margin: 0;
# }

# /* Stat cards */
# .stat-row {
#     display: grid;
#     grid-template-columns: repeat(4, 1fr);
#     gap: 16px;
#     margin-bottom: 32px;
# }
# .stat-card {
#     background: #111;
#     border: 1px solid #222;
#     border-radius: 16px;
#     padding: 24px;
#     text-align: center;
# }
# .stat-card .val {
#     font-family: 'Syne', sans-serif;
#     font-size: 40px;
#     font-weight: 800;
#     background: linear-gradient(135deg, #FF3CAC, #2B86C5);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     line-height: 1;
#     margin-bottom: 6px;
# }
# .stat-card .lbl {
#     font-size: 13px;
#     color: #888;
#     text-transform: uppercase;
#     letter-spacing: 1px;
# }

# /* Section heading */
# .section-heading {
#     font-family: 'Syne', sans-serif;
#     font-size: 28px;
#     font-weight: 700;
#     color: #fff;
#     margin-bottom: 20px;
# }

# /* Predict button */
# .stButton > button {
#     background: linear-gradient(135deg, #FF3CAC, #784BA0, #2B86C5) !important;
#     color: white !important;
#     font-family: 'Syne', sans-serif !important;
#     font-size: 18px !important;
#     font-weight: 700 !important;
#     border: none !important;
#     border-radius: 12px !important;
#     padding: 16px 40px !important;
#     width: 100% !important;
#     letter-spacing: 1px;
#     cursor: pointer;
# }
# .stButton > button:hover {
#     opacity: 0.9 !important;
#     transform: scale(1.01);
# }

# /* Result box */
# .result-benign {
#     background: #0d2b1a;
#     border: 2px solid #00e676;
#     border-radius: 16px;
#     padding: 28px;
#     text-align: center;
# }
# .result-malignant {
#     background: #2b0d0d;
#     border: 2px solid #ff1744;
#     border-radius: 16px;
#     padding: 28px;
#     text-align: center;
# }
# .result-title {
#     font-family: 'Syne', sans-serif;
#     font-size: 32px;
#     font-weight: 800;
#     margin-bottom: 8px;
# }
# .result-subtitle {
#     font-size: 15px;
#     color: #aaa;
# }

# /* Slider labels */
# .stSlider label {
#     color: #ccc !important;
#     font-size: 13px !important;
# }

# /* Tabs / radio */
# .stRadio label {
#     color: #fff !important;
# }

# /* Dark cards for charts */
# .chart-card {
#     background: #111;
#     border: 1px solid #222;
#     border-radius: 16px;
#     padding: 24px;
#     margin-bottom: 20px;
# }
# .chart-title {
#     font-family: 'Syne', sans-serif;
#     font-size: 18px;
#     font-weight: 700;
#     color: #fff;
#     margin-bottom: 16px;
# }

# /* About section */
# .about-card {
#     background: #111;
#     border: 1px solid #222;
#     border-radius: 16px;
#     padding: 28px;
#     margin-bottom: 16px;
# }
# .about-card h3 {
#     font-family: 'Syne', sans-serif;
#     font-size: 20px;
#     font-weight: 700;
#     color: #fff;
#     margin-bottom: 12px;
# }
# .about-card p, .about-card li {
#     color: #aaa;
#     font-size: 15px;
#     line-height: 1.7;
# }
# .tag {
#     display: inline-block;
#     background: #1e1e2e;
#     border: 1px solid #333;
#     color: #ccc;
#     border-radius: 8px;
#     padding: 4px 12px;
#     font-size: 13px;
#     margin: 4px 4px 4px 0;
# }
# .disclaimer {
#     background: #1a1200;
#     border: 1px solid #ffab00;
#     border-radius: 12px;
#     padding: 16px 20px;
#     color: #ffab00;
#     font-size: 14px;
#     margin-top: 16px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ─── Load Model ────────────────────────────────────────────────
# @st.cache_resource
# def load_model():
#     model  = joblib.load('outputs/model.pkl')
#     scaler = joblib.load('outputs/scaler.pkl')
#     return model, scaler

# model, scaler = load_model()
# cancer        = load_breast_cancer()
# feature_names = cancer.feature_names

# # ─── Sidebar ───────────────────────────────────────────────────
# with st.sidebar:
#     st.markdown("""
#     <div style='padding: 8px 0 24px 0;'>
#         <div style='font-family: Syne, sans-serif; font-size: 24px; font-weight: 800;
#                     background: linear-gradient(135deg, #FF3CAC, #2B86C5);
#                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
#             CanCure AI
#         </div>
#         <div style='font-size: 12px; color: #666; margin-top: 4px;'>
#             Breast Cancer Detection
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     page = st.radio("Navigation", [
#         "🔬  Predict",
#         "📊  Performance",
#         "📈  Visualizations",
#         "ℹ️  About"
#     ], label_visibility="collapsed")

#     st.markdown("""
#     <div style='margin-top: 40px; padding: 16px; background: #1a1a1a;
#                 border-radius: 12px; border: 1px solid #222;'>
#         <div style='font-size: 12px; color: #555; margin-bottom: 8px;
#                     text-transform: uppercase; letter-spacing: 1px;'>Model Info</div>
#         <div style='font-size: 13px; color: #888; line-height: 2;'>
#             Algorithm: Random Forest<br>
#             Dataset: 569 samples<br>
#             Features: 30<br>
#             Split: 80 / 20
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# # ─── Hero ──────────────────────────────────────────────────────
# st.markdown("""
# <div class="hero">
#     <h1>CanCure AI 🎗️</h1>
#     <p>Breast Cancer Detection powered by Random Forest — 96% Accuracy</p>
# </div>
# """, unsafe_allow_html=True)

# # ─── Stat Cards ────────────────────────────────────────────────
# st.markdown("""
# <div class="stat-row">
#     <div class="stat-card"><div class="val">96%</div><div class="lbl">Accuracy</div></div>
#     <div class="stat-card"><div class="val">93%</div><div class="lbl">Precision</div></div>
#     <div class="stat-card"><div class="val">98%</div><div class="lbl">Recall</div></div>
#     <div class="stat-card"><div class="val">95%</div><div class="lbl">F1 Score</div></div>
# </div>
# """, unsafe_allow_html=True)

# # ══════════════════════════════════════════════════════════════
# # PAGE 1 — PREDICT
# # ══════════════════════════════════════════════════════════════
# if page == "🔬  Predict":
#     st.markdown('<div class="section-heading">Patient Diagnostic Input</div>', unsafe_allow_html=True)
#     st.markdown('<p style="color:#666; margin-top:-12px; margin-bottom:24px;">Adjust sliders to match patient data, then click Analyze.</p>', unsafe_allow_html=True)

#     col1, col2, col3 = st.columns(3)
#     input_data = {}

#     for i, feature in enumerate(feature_names):
#         min_val  = float(cancer.data[:, i].min())
#         max_val  = float(cancer.data[:, i].max())
#         mean_val = float(cancer.data[:, i].mean())
#         if i % 3 == 0:
#             with col1:
#                 input_data[feature] = st.slider(feature, min_val, max_val, mean_val, key=feature)
#         elif i % 3 == 1:
#             with col2:
#                 input_data[feature] = st.slider(feature, min_val, max_val, mean_val, key=feature)
#         else:
#             with col3:
#                 input_data[feature] = st.slider(feature, min_val, max_val, mean_val, key=feature)

#     st.markdown("<br>", unsafe_allow_html=True)

#     if st.button("🔍  ANALYZE PATIENT DATA"):
#         input_array  = np.array(list(input_data.values())).reshape(1, -1)
#         input_scaled = scaler.transform(input_array)
#         prediction   = model.predict(input_scaled)[0]
#         probability  = model.predict_proba(input_scaled)[0]

#         st.markdown("<br>", unsafe_allow_html=True)
#         col_a, col_b = st.columns([1, 1])

#         with col_a:
#             if prediction == 1:
#                 st.markdown(f"""
#                 <div class="result-benign">
#                     <div class="result-title" style="color: #00e676;">✓ BENIGN</div>
#                     <div class="result-subtitle">Non-cancerous tumor detected</div>
#                     <div style="margin-top: 16px; font-family: Syne; font-size: 36px;
#                                 font-weight: 800; color: #00e676;">
#                         {probability[1]*100:.1f}%
#                     </div>
#                     <div style="color: #555; font-size: 13px;">Confidence Score</div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                 <div class="result-malignant">
#                     <div class="result-title" style="color: #ff1744;">⚠ MALIGNANT</div>
#                     <div class="result-subtitle">Cancerous tumor detected</div>
#                     <div style="margin-top: 16px; font-family: Syne; font-size: 36px;
#                                 font-weight: 800; color: #ff1744;">
#                         {probability[0]*100:.1f}%
#                     </div>
#                     <div style="color: #555; font-size: 13px;">Confidence Score</div>
#                 </div>
#                 """, unsafe_allow_html=True)

#         with col_b:
#             fig, ax = plt.subplots(figsize=(5, 3.5))
#             fig.patch.set_facecolor('#111111')
#             ax.set_facecolor('#111111')
#             colors = ['#ff1744', '#00e676']
#             bars = ax.barh(['Malignant', 'Benign'],
#                            [probability[0], probability[1]],
#                            color=colors, height=0.5, edgecolor='none')
#             ax.set_xlim(0, 1)
#             ax.set_xlabel('Probability', color='#888', fontsize=11)
#             ax.tick_params(colors='#aaa')
#             ax.spines['top'].set_visible(False)
#             ax.spines['right'].set_visible(False)
#             ax.spines['bottom'].set_color('#333')
#             ax.spines['left'].set_color('#333')
#             for bar, prob in zip(bars, [probability[0], probability[1]]):
#                 ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
#                         f'{prob*100:.1f}%', va='center', color='#fff', fontsize=12, fontweight='bold')
#             ax.set_title('Prediction Confidence', color='#fff', fontsize=13, fontweight='bold', pad=12)
#             plt.tight_layout()
#             st.pyplot(fig)
#             plt.close()

#         st.markdown("""
#         <div class="disclaimer">
#             ⚠️ For educational purposes only. Always consult a qualified medical professional.
#         </div>
#         """, unsafe_allow_html=True)

# # ══════════════════════════════════════════════════════════════
# # PAGE 2 — PERFORMANCE
# # ══════════════════════════════════════════════════════════════
# elif page == "📊  Performance":
#     st.markdown('<div class="section-heading">Model Performance</div>', unsafe_allow_html=True)

#     col_a, col_b = st.columns(2)
#     with col_a:
#         st.markdown('<div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
#         try:
#             st.image('outputs/confusion_matrix.png', width='stretch')
#         except:
#             st.warning("Run evaluate.py first")

#     with col_b:
#         st.markdown('<div class="chart-title">ROC Curve</div>', unsafe_allow_html=True)
#         try:
#             st.image('outputs/roc_curve.png', width='stretch')
#         except:
#             st.warning("Run evaluate.py first")

#     st.markdown('<div class="chart-title" style="margin-top:8px;">Feature Importance — Top 15</div>', unsafe_allow_html=True)
#     try:
#         st.image('outputs/feature_importance.png', width='stretch')
#     except:
#         st.warning("Run evaluate.py first")

# # ══════════════════════════════════════════════════════════════
# # PAGE 3 — VISUALIZATIONS
# # ══════════════════════════════════════════════════════════════
# elif page == "📈  Visualizations":
#     st.markdown('<div class="section-heading">Data Visualizations</div>', unsafe_allow_html=True)

#     col_a, col_b = st.columns(2)
#     with col_a:
#         st.markdown('<div class="chart-title">Target Distribution</div>', unsafe_allow_html=True)
#         try:
#             st.image('outputs/target_distribution.png', width='stretch')
#         except:
#             st.warning("Run visualize.py first")

#     with col_b:
#         st.markdown('<div class="chart-title">Feature Distributions</div>', unsafe_allow_html=True)
#         try:
#             st.image('outputs/feature_distributions.png', width='stretch')
#         except:
#             st.warning("Run visualize.py first")

#     st.markdown('<div class="chart-title" style="margin-top:8px;">Correlation Heatmap</div>', unsafe_allow_html=True)
#     try:
#         st.image('outputs/correlation_heatmap.png', width='stretch')
#     except:
#         st.warning("Run visualize.py first")

#     st.markdown('<div class="chart-title" style="margin-top:8px;">Boxplots — Key Features</div>', unsafe_allow_html=True)
#     try:
#         st.image('outputs/boxplots.png', width='stretch')
#     except:
#         st.warning("Run visualize.py first")

# # ══════════════════════════════════════════════════════════════
# # PAGE 4 — ABOUT
# # ══════════════════════════════════════════════════════════════
# elif page == "ℹ️  About":
#     st.markdown('<div class="section-heading">About CanCure AI</div>', unsafe_allow_html=True)

#     col_a, col_b = st.columns(2)

#     with col_a:
#         st.markdown("""
#         <div class="about-card">
#             <h3>What is CanCure AI?</h3>
#             <p>CanCure AI is a machine learning web application that predicts
#             whether a breast tumor is <strong style="color:#fff">Malignant</strong> or
#             <strong style="color:#fff">Benign</strong> using clinical diagnostic data
#             from the Wisconsin Breast Cancer Dataset.</p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#         <div class="about-card">
#             <h3>Tech Stack</h3>
#             <span class="tag">Python 3.13</span>
#             <span class="tag">Scikit-learn</span>
#             <span class="tag">Pandas</span>
#             <span class="tag">NumPy</span>
#             <span class="tag">Matplotlib</span>
#             <span class="tag">Seaborn</span>
#             <span class="tag">Streamlit</span>
#             <span class="tag">Joblib</span>
#         </div>
#         """, unsafe_allow_html=True)

#     with col_b:
#         st.markdown("""
#         <div class="about-card">
#             <h3>Model Details</h3>
#             <p>
#             <strong style="color:#fff">Algorithm:</strong> Random Forest Classifier<br><br>
#             <strong style="color:#fff">Dataset:</strong> Wisconsin Breast Cancer (UCI)<br><br>
#             <strong style="color:#fff">Samples:</strong> 569 (212 Malignant, 357 Benign)<br><br>
#             <strong style="color:#fff">Features:</strong> 30 numeric diagnostic features<br><br>
#             <strong style="color:#fff">Train / Test Split:</strong> 80% / 20%<br><br>
#             <strong style="color:#fff">Cross Validation:</strong> 5-Fold
#             </p>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("""
#         <div class="about-card">
#             <h3>Performance Summary</h3>
#             <p>
#             Accuracy &nbsp;&nbsp; → &nbsp; <strong style="color:#FF3CAC">96%</strong><br>
#             Precision &nbsp; → &nbsp; <strong style="color:#FF3CAC">93%</strong><br>
#             Recall &nbsp;&nbsp;&nbsp;&nbsp; → &nbsp; <strong style="color:#FF3CAC">98%</strong><br>
#             F1 Score &nbsp;&nbsp; → &nbsp; <strong style="color:#FF3CAC">95%</strong>
#             </p>
#         </div>
#         """, unsafe_allow_html=True)

#     # st.markdown("""
#     # <div class="disclaimer">
#     #     ⚠️ Disclaimer: This application is for educational purposes only.
#     #     It is not a substitute for professional medical advice, diagnosis, or treatment.
#     #     Always consult a qualified healthcare professional.
#     # </div>
#     # """, unsafe_allow_html=True)












import streamlit as st
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import load_breast_cancer

st.set_page_config(
    page_title="CanCure AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: #0a0a0a;
}

[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * {
    color: #fff !important;
}

.hero {
    background: #151515;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 800;
    color: #fff;
    margin: 0 0 8px 0;
    line-height: 1.1;
}
.hero p {
    font-size: 18px;
    color: rgba(255,255,255,0.8);
    margin: 0;
}

.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.stat-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.stat-card .val {
    font-family: 'Syne', sans-serif;
    font-size: 40px;
    font-weight: 800;
    background: linear-gradient(135deg, #FF3CAC, #2B86C5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin-bottom: 6px;
}

            
.stat-card.accuracy { border-left: 4px solid #4caf50; }
.stat-card.precision { border-left: 4px solid #2196f3; }
.stat-card.recall { border-left: 4px solid #ff9800; }
.stat-card.f1 { border-left: 4px solid #9c27b0; }
            
.stat-card .lbl {
    font-size: 13px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 20px;
}

.stButton > button {
    background: linear-gradient(135deg, #FF3CAC, #784BA0, #2B86C5) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 40px !important;
    width: 100% !important;
    letter-spacing: 1px;
    cursor: pointer;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: scale(1.01);
}

.result-benign {
    background: #0d2b1a;
    border: 2px solid #00e676;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-malignant {
    background: #2b0d0d;
    border: 2px solid #ff1744;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    margin-bottom: 8px;
}
.result-subtitle {
    font-size: 15px;
    color: #aaa;
}

.stSlider label {
    color: #ccc !important;
    font-size: 13px !important;
}

.stRadio label {
    color: #fff !important;
}

.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 16px;
}

.about-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
}
.about-card h3 {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 12px;
}
.about-card p, .about-card li {
    color: #aaa;
    font-size: 15px;
    line-height: 1.7;
}
.tag {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #333;
    color: #ccc;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 13px;
    margin: 4px 4px 4px 0;
}
.disclaimer {
    background: #1a1200;
    border: 1px solid #ffab00;
    border-radius: 12px;
    padding: 16px 20px;
    color: #ffab00;
    font-size: 14px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('outputs/model.pkl')
    scaler = joblib.load('outputs/scaler.pkl')
    return model, scaler

model, scaler = load_model()
cancer        = load_breast_cancer()
feature_names = cancer.feature_names

# ─── Load Real Metrics from evaluate.py output ────────────────
try:
    with open("outputs/metrics.json", "r") as f:
        metrics = json.load(f)
    ACC  = f"{metrics['accuracy']}%"
    PREC = f"{metrics['precision']}%"
    REC  = f"{metrics['recall']}%"
    F1   = f"{metrics['f1']}%"
except:
    st.warning("⚠️ metrics.json not found. Run: python src/evaluate.py")
    ACC, PREC, REC, F1 = "N/A", "N/A", "N/A", "N/A"

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    # st.markdown("""
    # <div style='padding: 8px 0 24px 0;'>
    #     <div style='font-family: Syne, sans-serif; font-size: 24px; font-weight: 800;
    #                 background: linear-gradient(135deg, #FF3CAC, #2B86C5);
    #                 -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
    #         CanCure AI
    #     </div>
    #     <div style='font-size: 12px; color: #666; margin-top: 4px;'>
    #         Breast Cancer Detection
    #     </div>
    # </div>
    # """, unsafe_allow_html=True)

    st.markdown(f"""
<div class="hero">
    <h1 style="font-size:42px;">CanCure AI 🎗️</h1>
    <p style="font-size:16px; color:#aaa;">
        AI-assisted breast cancer screening tool
    </p>
    <p style="margin-top:8px; font-size:14px; color:#666;">
        Model accuracy: {ACC}
    </p>
</div>
""", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🔬  Predict",
        "📊  Performance",
        "📈  Visualizations",
        "ℹ️  About"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='margin-top: 40px; padding: 16px; background: #1a1a1a;
                border-radius: 12px; border: 1px solid #222;'>
        <div style='font-size: 12px; color: #555; margin-bottom: 8px;
                    text-transform: uppercase; letter-spacing: 1px;'>Model Info</div>
        <div style='font-size: 13px; color: #888; line-height: 2;'>
            Algorithm: Random Forest<br>
            Dataset: 569 samples<br>
            Features: 30<br>
            Split: 80 / 20
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Hero ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <h1>CanCure AI 🎗️</h1>
    <p>Breast Cancer Detection powered by Random Forest — {ACC} Accuracy</p>
</div>
""", unsafe_allow_html=True)

# ─── Stat Cards (Dynamic) ──────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card"><div class="val">{ACC}</div><div class="lbl">Accuracy</div></div>
    <div class="stat-card"><div class="val">{PREC}</div><div class="lbl">Precision</div></div>
    <div class="stat-card"><div class="val">{REC}</div><div class="lbl">Recall</div></div>
    <div class="stat-card"><div class="val">{F1}</div><div class="lbl">F1 Score</div></div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════
if page == "🔬  Predict":
    st.markdown('<div class="section-heading">Patient Diagnostic Input</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#666; margin-top:-12px; margin-bottom:24px;">Adjust sliders to match patient data, then click Analyze.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    # input_data = {}

    # for i, feature in enumerate(feature_names):
    #     min_val  = float(cancer.data[:, i].min())
    #     max_val  = float(cancer.data[:, i].max())
    #     mean_val = float(cancer.data[:, i].mean())
    #     if i % 3 == 0:
    #         with col1:
    #             input_data[feature] = st.slider(feature, min_val, max_val, mean_val, key=feature)
    #     elif i % 3 == 1:
    #         with col2:
    #             input_data[feature] = st.slider(feature, min_val, max_val, mean_val, key=feature)
    #     else:
    #         with col3:
    #             input_data[feature] = st.slider(feature, min_val, max_val, mean_val, key=feature)

    input_data = {}

    with st.expander("📊 Basic Features"):
        col1, col2 = st.columns(2)
        for i, feature in enumerate(feature_names[:10]):
            with col1 if i % 2 == 0 else col2:
                min_val = float(cancer.data[:, i].min())
                max_val = float(cancer.data[:, i].max())
                mean_val = float(cancer.data[:, i].mean())
                input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

    with st.expander("🧬 Advanced Features"):
        col1, col2 = st.columns(2)
        for i, feature in enumerate(feature_names[10:]):
            with col1 if i % 2 == 0 else col2:
                min_val = float(cancer.data[:, i].min())
                max_val = float(cancer.data[:, i].max())
                mean_val = float(cancer.data[:, i].mean())
                input_data[feature] = st.slider(feature, min_val, max_val, mean_val)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  ANALYZE PATIENT DATA"):
        input_array  = np.array(list(input_data.values())).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1, 1])

        with col_a:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-benign">
                    <div class="result-title" style="color: #00e676;">Result: ✓Benign</div>
                    <div class="result-subtitle">Non-cancerous tumor detected</div>
                    <div style="margin-top: 16px; font-family: Syne; font-size: 36px;
                                font-weight: 800; color: #00e676;">
                        {probability[1]*100:.1f}%
                    </div>
                    <div style="color: #555; font-size: 13px;">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
                st.warning("This tool is not a medical diagnosis. Consult a doctor.")
            else:
                st.markdown(f"""
                <div class="result-malignant">
                    <div class="result-title" style="color: #ff1744;">Result: ⚠Malignant</div>
                    <div class="result-subtitle">Cancerous tumor detected</div>
                    <div style="margin-top: 16px; font-family: Syne; font-size: 36px;
                                font-weight: 800; color: #ff1744;">
                        {probability[0]*100:.1f}%
                    </div>
                    <div style="color: #555; font-size: 13px;">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)
                st.warning("This tool is not a medical diagnosis. Consult a doctor.")

        with col_b:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor('#111111')
            ax.set_facecolor('#111111')
            colors = ['#ff1744', '#00e676']
            bars = ax.barh(['Malignant', 'Benign'],
                           [probability[0], probability[1]],
                           color=colors, height=0.5, edgecolor='none')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability', color='#888', fontsize=11)
            ax.tick_params(colors='#aaa')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#333')
            ax.spines['left'].set_color('#333')
            for bar, prob in zip(bars, [probability[0], probability[1]]):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{prob*100:.1f}%', va='center', color='#fff',
                        fontsize=12, fontweight='bold')
            ax.set_title('Prediction Confidence', color='#fff',
                         fontsize=13, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # st.markdown("""
        # <div class="disclaimer">
        #     ⚠️ For educational purposes only. Always consult a qualified medical professional.
        # </div>
        # """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊  Performance":
    st.markdown('<div class="section-heading">Model Performance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  ACC,  "↑ High")
    col2.metric("Precision", PREC, "↑ High")
    col3.metric("Recall",    REC,  "↑ High")
    col4.metric("F1 Score",  F1,   "↑ High")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-title">Confusion Matrix</div>', unsafe_allow_html=True)
        try:
            st.image('outputs/confusion_matrix.png', width='stretch')
        except:
            st.warning("Run evaluate.py first")

    with col_b:
        st.markdown('<div class="chart-title">ROC Curve</div>', unsafe_allow_html=True)
        try:
            st.image('outputs/roc_curve.png', width='stretch')
        except:
            st.warning("Run evaluate.py first")

    st.markdown('<div class="chart-title" style="margin-top:8px;">Feature Importance — Top 15</div>', unsafe_allow_html=True)
    try:
        st.image('outputs/feature_importance.png', width='stretch')
    except:
        st.warning("Run evaluate.py first")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "📈  Visualizations":
    st.markdown('<div class="section-heading">Data Visualizations</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="chart-title">Target Distribution</div>', unsafe_allow_html=True)
        try:
            st.image('outputs/target_distribution.png', width='stretch')
        except:
            st.warning("Run visualize.py first")

    with col_b:
        st.markdown('<div class="chart-title">Feature Distributions</div>', unsafe_allow_html=True)
        try:
            st.image('outputs/feature_distributions.png', width='stretch')
        except:
            st.warning("Run visualize.py first")

    st.markdown('<div class="chart-title" style="margin-top:8px;">Correlation Heatmap</div>', unsafe_allow_html=True)
    try:
        st.image('outputs/correlation_heatmap.png', width='stretch')
    except:
        st.warning("Run visualize.py first")

    st.markdown('<div class="chart-title" style="margin-top:8px;">Boxplots — Key Features</div>', unsafe_allow_html=True)
    try:
        st.image('outputs/boxplots.png', width='stretch')
    except:
        st.warning("Run visualize.py first")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️  About":
    st.markdown('<div class="section-heading">About CanCure AI</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="about-card">
            <h3>What is CanCure AI?</h3>
            <p>CanCure AI is a machine learning web application that predicts
            whether a breast tumor is <strong style="color:#fff">Malignant</strong> or
            <strong style="color:#fff">Benign</strong> using clinical diagnostic data
            from the Wisconsin Breast Cancer Dataset.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>Tech Stack</h3>
            <span class="tag">Python 3.13</span>
            <span class="tag">Scikit-learn</span>
            <span class="tag">Pandas</span>
            <span class="tag">NumPy</span>
            <span class="tag">Matplotlib</span>
            <span class="tag">Seaborn</span>
            <span class="tag">Streamlit</span>
            <span class="tag">Joblib</span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-card">
            <h3>Model Details</h3>
            <p>
            <strong style="color:#fff">Algorithm:</strong> Random Forest Classifier<br><br>
            <strong style="color:#fff">Dataset:</strong> Wisconsin Breast Cancer (UCI)<br><br>
            <strong style="color:#fff">Samples:</strong> 569 (212 Malignant, 357 Benign)<br><br>
            <strong style="color:#fff">Features:</strong> 30 numeric diagnostic features<br><br>
            <strong style="color:#fff">Train / Test Split:</strong> 80% / 20%<br><br>
            <strong style="color:#fff">Cross Validation:</strong> 5-Fold
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="about-card">
            <h3>Performance Summary</h3>
            <p>
            Accuracy &nbsp;&nbsp; → &nbsp; <strong style="color:#FF3CAC">{ACC}</strong><br>
            Precision &nbsp; → &nbsp; <strong style="color:#FF3CAC">{PREC}</strong><br>
            Recall &nbsp;&nbsp;&nbsp;&nbsp; → &nbsp; <strong style="color:#FF3CAC">{REC}</strong><br>
            F1 Score &nbsp;&nbsp; → &nbsp; <strong style="color:#FF3CAC">{F1}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ Disclaimer: This application is for educational purposes only.
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)