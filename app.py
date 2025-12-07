import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="HLH Diagnostic System",
    page_icon="🧬",
    layout="centered"
)

# ------------------------------------
# Load model and explainer
# ------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("saved_model/random_forest_HLH_model.pkl")
    explainer = joblib.load("saved_model/shap_explainer.pkl")
    feature_names = joblib.load("saved_model/feature_names.pkl")
    return model, explainer, feature_names

model, explainer, feature_names = load_assets()


# ------------------------------------
# UI
# ------------------------------------
st.title("🩸 HLH Diagnostic System")

st.markdown("""
This web application uses a **Random Forest model** to assist in the
**identification of hemophagocytic lymphohistiocytosis (HLH)** based on laboratory data.
""")


# ------------------------------------
# User input
# ------------------------------------
st.subheader("Enter laboratory indicators")

inputs = []

for feat in feature_names:
    val = st.number_input(f"{feat}", min_value=0.0, value=1.0)
    inputs.append(val)

X_input = np.array([inputs])


# ------------------------------------
# Prediction
# ------------------------------------
if st.button("🚀 Run Prediction"):

    prob = model.predict_proba(X_input)[:, 1][0]
    pred = int(prob >= 0.5)

    st.markdown("---")

    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"⚠️ High HLH likelihood (Probability = {prob:.3f})")
    else:
        st.success(f"✅ Low HLH likelihood (Probability = {prob:.3f})")

    st.progress(prob)


    # ------------------------------------
    # SHAP explain single input
    # ------------------------------------
    shap_vals = explainer.shap_values(X_input)

    if isinstance(shap_vals, list):
        shap_vals_sample = shap_vals[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_vals_sample = shap_vals[0]
        base_value = explainer.expected_value

    shap_exp = shap.Explanation(
        values=shap_vals_sample,
        base_values=base_value,
        data=X_input[0],
        feature_names=feature_names
    )

    st.markdown("### 🔍 SHAP Explanation")

    fig = plt.figure(figsize=(7, 6))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------
# Disclaimer
# ------------------------------------
st.markdown("""
---
⚠️ **Disclaimer:**  
This tool is intended for research and educational purposes only.  
It does not provide medical advice, diagnosis, or treatment.
""")
