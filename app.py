import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="HLH Diagnostic System",
    page_icon="🧬",
    layout="centered"
)

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("saved_model/random_forest_HLH_model.pkl")
    return model

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.markdown("""
<h1 style="text-align:center; color:#1b3b5f; font-weight:700;">
HLH Diagnostic System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#333333; font-size:18px; margin-bottom:25px;">
Machine learning model with SHAP interpretability for laboratory-based assessment of HLH.
<br><br>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# User Inputs
# -----------------------------
st.subheader("Enter laboratory indicators", divider="gray")

PLT = st.number_input("Platelet count (PLT, ×10⁹/L)", min_value=0.0, value=200.0, step=1.0)
ALP = st.number_input("Alkaline phosphatase (ALP, U/L)", min_value=0.0, value=100.0, step=0.1)
IFN = st.number_input("Interferon-γ (IFN-γ, pg/mL)", min_value=0.0, value=5.0, step=0.1)
IL10_IL6 = st.number_input("IL-10 / IL-6 ratio", min_value=0.0, value=0.5, step=0.01)
Hb = st.number_input("Hemoglobin (Hb, g/L)", min_value=0.0, value=100.0, step=0.1)

# assemble features
feature_names = ["PLT", "ALP", "IFN-γ", "IL-10/IL-6", "Hb"]
X_input = np.array([[PLT, ALP, IFN, IL10_IL6, Hb]])


# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Run Prediction", use_container_width=True):

    prob = model.predict_proba(X_input)[:, 1][0]
    pred = int(prob >= 0.5)

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(
            f"⚠️ High HLH likelihood (Probability = {prob:.3f})",
            icon="⚠️"
        )
    else:
        st.success(
            f"✅ Low HLH likelihood (Probability = {prob:.3f})",
            icon="✔️"
        )

    st.progress(min(max(prob, 0), 1))


    # -----------------------------
    # SHAP Explainability
    # -----------------------------
    st.markdown("### 🔍 SHAP Explanation")

    # build explainer dynamically
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_input)

    # get SHAP values for positive class
    if isinstance(shap_values, list):
        shap_sample = shap_values[1][0]
    else:
        shap_sample = shap_values[0]

    # -----------------------------
    # 🔥 Force baseline to 0.65
    # -----------------------------
    FIXED_BASE_VALUE = 0.65

    shap_exp = shap.Explanation(
        values=shap_sample,
        base_values=FIXED_BASE_VALUE,
        data=X_input[0],
        feature_names=feature_names
    )

    # plot waterfall
    fig = plt.figure(figsize=(7, 6))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------
# Footer info
# -----------------------------
st.markdown("""
---
<div style="color:#666666; font-size:14px; line-height:1.6;">
<strong>Model:</strong> Random Forest (trained on matched adult febrile cohort)<br>
<strong>Features:</strong> PLT, ALP, IFN-γ, IL-10/IL-6, Hb<br>
<strong>Version:</strong> 1.1.0 (with SHAP)
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("""
<div style="color:#888888; font-size:13px; line-height:1.6; margin-top:10px;">
⚠️ <strong>Disclaimer:</strong><br>
This tool is intended for research and educational purposes only.  
It does not provide medical advice, diagnosis, or treatment.  
Clinical decision-making should rely on comprehensive evaluation by qualified physicians.
</div>
""", unsafe_allow_html=True)
