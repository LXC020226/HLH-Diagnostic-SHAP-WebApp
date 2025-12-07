import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ================================================
# 页面设置
# ================================================
st.set_page_config(
    page_title="HLH Diagnostic System",
    page_icon="🧬",
    layout="centered"
)

st.markdown(
    "<h2 style='text-align:center;'>HLH Diagnostic System</h2>",
    unsafe_allow_html=True
)

st.markdown("""
This application uses a machine learning model trained on laboratory data  
to assist in the **identification of hemophagocytic lymphohistiocytosis (HLH)**.  
""")

# ================================================
# 加载模型与标准化器
# ================================================
@st.cache_resource
def load_model():
    rf = joblib.load("saved_model/random_forest_HLH_model.pkl")
    scaler = joblib.load("saved_model/feature_scaler.pkl")
    return rf, scaler

rf_model, scaler = load_model()

# SHAP explainer
explainer = shap.TreeExplainer(rf_model)
feature_names = ["PLT", "ALP", "IFN-γ", "IL-10/IL-6", "Hb"]


# ================================================
# 用户输入区域
# ================================================
st.subheader("Input laboratory indicators")

PLT = st.number_input("Platelet count (PLT, ×10⁹/L)", min_value=0.0, value=200.0)
ALP = st.number_input("Alkaline phosphatase (ALP, U/L)", min_value=0.0, value=100.0)
IFN = st.number_input("Interferon-γ (IFN-γ, pg/mL)", min_value=0.0, value=5.0)
IL10_IL6 = st.number_input("IL-10 / IL-6 ratio", min_value=0.0, value=0.5)
Hb = st.number_input("Hemoglobin (Hb, g/L)", min_value=0.0, value=100.0)


# ================================================
# 预测
# ================================================
if st.button("Run Prediction"):
    
    X_input = np.array([[PLT, ALP, IFN, IL10_IL6, Hb]])
    X_scaled = scaler.transform(X_input)

    prob = rf_model.predict_proba(X_scaled)[:, 1][0]
    pred = int(prob >= 0.5)

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"High HLH likelihood (probability = {prob:.3f})")
        st.markdown("*Suggest further hematologic evaluation when clinically indicated.*")
    else:
        st.success(f"Low HLH likelihood (probability = {prob:.3f})")
        st.markdown("*Continue routine clinical monitoring.*")

    st.progress(float(prob))

    # ============================================
    # SHAP waterfall 图
    # ============================================

    st.markdown("---")
    st.subheader("SHAP Explanation (Waterfall Plot)")

    shap_values = explainer.shap_values(X_scaled)

    # 取正类 shap 值
    if isinstance(shap_values, list):
        shap_vals_sample = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_vals_sample = shap_values[0]
        base_value = explainer.expected_value

    # 构建 explanation
    shap_exp = shap.Explanation(
        values=shap_vals_sample,
        base_values=base_value,
        data=X_scaled[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(7, 7))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)


# ================================================
# 免责声明
# ================================================
st.markdown("---")
st.markdown("""
### Disclaimer

This tool is designed **for research and educational purposes only**.  
It is **not intended for clinical diagnosis or medical decision-making**.  
Decisions regarding patient care should rely on comprehensive clinical assessment  
and standard diagnostic criteria.
""")

st.markdown("""
**Model:** Random Forest  
**Features:** PLT, ALP, IFN-γ, IL-10/IL-6, Hb  
**Version:** 1.0.0  
""")
