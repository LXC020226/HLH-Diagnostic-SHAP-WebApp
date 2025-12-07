import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import streamlit.components.v1 as components

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="HLH Diagnostic System",
    page_icon="🧬",
    layout="wide"
)

# ===================== 全局样式 =====================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: "Arial", sans-serif !important;
    font-size: 20px !important;
}

.top-banner {
    background-color: #0A3D62;
    padding: 30px 40px;
    border-radius: 12px;
    margin-bottom: 25px;
}
.top-banner h1 {
    color: white;
    font-size: 40px;
    margin: 0;
}
.top-banner p {
    color: #dfe6e9;
    font-size: 22px;
    margin: 0;
}

label, div[data-baseweb="input"] label {
    font-size: 24px !important;
    font-weight: 600 !important;
    color: #0A3D62 !important;
}

.card {
    padding: 25px;
    border-radius: 12px;
    background-color: #F8F9FA;
    border: 1px solid #dcdde1;
    margin-bottom: 20px;
    font-size: 22px;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    color: #0A3D62;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-banner">
    <h1>HLH Diagnostic System</h1>
    <p>Machine-learning–assisted laboratory interpretation with SHAP explanation</p>
</div>
""", unsafe_allow_html=True)

# ===================== 加载模型与 scaler =====================
@st.cache_resource
def load_model():
    rf_model = joblib.load("saved_model/random_forest_HLH_model.pkl")
    scaler = joblib.load("saved_model/feature_scaler.pkl")
    return rf_model, scaler

rf_model, scaler = load_model()

# ===================== 构建 SHAP explainer =====================
@st.cache_resource
def load_explainer(model):
    explainer = shap.TreeExplainer(model)
    shap.initjs()
    return explainer

explainer = load_explainer(rf_model)
feature_names = ["PLT", "ALP", "IFN-γ", "IL-10/IL-6", "Hb"]

# ===================== 输入区 =====================
st.markdown('<div class="section-title">Patient Laboratory Inputs</div>', unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        PLT = st.number_input("Platelet Count (PLT, ×10⁹/L)", min_value=0.0, value=200.0)
        ALP = st.number_input("Alkaline Phosphatase (ALP, U/L)", min_value=0.0, value=100.0)

    with col2:
        IFN = st.number_input("Interferon-γ (IFN-γ, pg/mL)", min_value=0.0, value=5.0)
        IL10_IL6 = st.number_input("IL-10/IL-6 Ratio", min_value=0.0, value=1.0, format="%.3f")

    with col3:
        Hb = st.number_input("Hemoglobin (Hb, g/L)", min_value=0.0, value=100.0)

    submitted = st.form_submit_button("Run Prediction 🚀")

# ===================== 预测与显示结果 =====================
if submitted:
    X_input = np.array([[PLT, ALP, IFN, IL10_IL6, Hb]])
    X_scaled = scaler.transform(X_input)

    prob = rf_model.predict_proba(X_scaled)[:, 1][0]
    pred = int(prob >= 0.5)

    st.markdown('<div class="section-title">Diagnostic Result</div>', unsafe_allow_html=True)

    if pred == 1:
        color = "#c0392b"
        msg = "High Likelihood of HLH"
        Sug = "Further hematologic evaluation and HScore assessment recommended."
    else:
        color = "#27ae60"
        msg = "Low Likelihood of HLH"
        Sug = "Continue monitoring and rule out alternative causes."

    st.markdown(
        f"""
        <div class="card">
            <h3 style="color:{color}; margin-top:0; font-size:30px;">{msg}</h3>
            <p><strong>Predicted Probability: {prob:.3f}</strong></p>
            <p>{Sug}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(float(min(max(prob, 0), 1)))

    # ===================== 单个样本 SHAP force 图 =====================
    st.markdown('<div class="section-title">SHAP Force Plot (Individual Explanation)</div>', unsafe_allow_html=True)

    # 对当前样本计算 SHAP 值（针对阳性类别）
    shap_values_all = explainer.shap_values(X_scaled)
    # TreeExplainer + 二分类：返回 [class0, class1]
    shap_values_sample = shap_values_all[1][0]     # 当前样本、阳性类
    base_value = explainer.expected_value[1]       # 基本值

    # 生成 force plot（返回 JS + HTML）
    force_plot = shap.force_plot(
        base_value,
        shap_values_sample,
        X_scaled[0, :],
        feature_names=feature_names,
        matplotlib=False,
        show=False
    )

    # 将 JS + HTML 一起嵌入 Streamlit
    shap_html = f"""
    <head>{shap.getjs()}</head>
    <body>{force_plot.html()}</body>
    """

    components.html(shap_html, height=300)

    # 可选：再给一个简单的条形图表示各特征贡献大小
    st.subheader("Feature Contributions (bar plot)")
    plt.figure(figsize=(6, 4))
    order = np.argsort(np.abs(shap_values_sample))[::-1]
    ordered_names = [feature_names[i] for i in order]
    ordered_vals = shap_values_sample[order]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in ordered_vals]
    plt.bar(ordered_names, ordered_vals, color=colors)
    plt.axhline(0, color="grey", linestyle="--")
    plt.ylabel("SHAP value")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

# ===================== 免责声明 =====================
st.markdown("---")
st.markdown("""
### ⚠️ Disclaimer
This tool is intended solely for research and educational purposes.  
It is not a clinical diagnostic device and must not replace professional medical judgment.  
All predictions must be interpreted by qualified clinicians.
""")

st.markdown("""
---
Developed for research use • Version 2.1.0 (SHAP force plot enabled)
""")
