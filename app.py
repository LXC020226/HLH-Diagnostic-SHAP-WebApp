iimport streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# =============================================================
# 🌐 页面设置
# =============================================================
st.set_page_config(
    page_title="HLH Diagnostic System",
    page_icon="🧬",
    layout="centered"
)

# 全局字体 & 风格设定
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 14,
})


# =============================================================
# 🧬 顶部标题
# =============================================================
st.markdown("""
<div style="text-align:center; font-size:32px; font-weight:600; color:#1b3b5f;">
HLH Diagnostic System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; font-size:16px; color:#4b545c; margin-bottom:10px;">
Machine learning–assisted interpretation of laboratory indicators
</div>
""", unsafe_allow_html=True)


# =============================================================
# 📦 加载模型
# =============================================================
@st.cache_resource
def load_model():
    model = joblib.load("saved_model/random_forest_HLH_model.pkl")
    scaler = joblib.load("saved_model/feature_scaler.pkl")
    return model, scaler

rf_model, scaler = load_model()
explainer = shap.TreeExplainer(rf_model)

feature_names = ["PLT", "ALP", "IFN-γ", "IL-10/IL-6", "Hb"]


# =============================================================
# 🧪 输入界面
# =============================================================
st.markdown("""
<div style="font-size:20px; font-weight:500; color:#1b3b5f; margin-top:15px;">
Patient Laboratory Indicators
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)

    PLT = col1.number_input("Platelet count (PLT, ×10⁹/L)", min_value=0.0, value=200.0)
    ALP = col2.number_input("Alkaline phosphatase (ALP, U/L)", min_value=0.0, value=100.0)

    IFN = col1.number_input("Interferon-γ (IFN-γ, pg/mL)", min_value=0.0, value=5.0)
    IL10_IL6 = col2.number_input("IL-10 / IL-6 ratio", min_value=0.0, value=0.5)

    Hb = st.number_input("Hemoglobin (Hb, g/L)", min_value=0.0, value=100.0)


# =============================================================
# 🚀 预测与解释
# =============================================================
if st.button("Run Prediction", use_container_width=True):
    st.markdown("<hr>", unsafe_allow_html=True)

    # --------------- 预测 --------------------
    X_input = np.array([[PLT, ALP, IFN, IL10_IL6, Hb]])
    X_scaled = scaler.transform(X_input)
    prob = rf_model.predict_proba(X_scaled)[:, 1][0]
    pred = int(prob >= 0.5)

    # --------------- 结果展示 ---------------------
    st.markdown("""
    <div style="font-size:20px; font-weight:600; color:#1b3b5f;">
    Prediction Result
    </div>
    """, unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div style="background-color:#fdecef; border-left:6px solid #d93455; padding:12px;
                    border-radius:6px; margin:8px 0; font-size:16px;">
        <strong>High HLH likelihood</strong><br>
        Probability: {prob:.3f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color:#e7f6f1; border-left:6px solid #00a27d; padding:12px;
                    border-radius:6px; margin:8px 0; font-size:16px;">
        <strong>Low HLH likelihood</strong><br>
        Probability: {prob:.3f}
        </div>
        """, unsafe_allow_html=True)

    st.progress(float(prob))


    # =============================================================
    # 📊 SHAP 单样本 waterfall 图
    # =============================================================
    st.markdown("""
    <div style="font-size:20px; font-weight:600; color:#1b3b5f; margin-top:15px;">
    SHAP Explanation
    </div>
    """, unsafe_allow_html=True)

    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        shap_vals = shap_values[0]
        base_value = explainer.expected_value

    shap_exp = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=X_scaled[0],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(7, 6))
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    st.pyplot(fig)
    plt.close(fig)


# =============================================================
# 📌 免责声明
# =============================================================
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:14px; color:#616a70; text-align:justify;">
<b>Disclaimer:</b><br>
This tool is provided for research and educational purposes only. 
It is not intended for clinical diagnosis, treatment, or medical decision-making.  
Clinical judgment and standardized diagnostic criteria should guide patient care.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-size:12px; color:#8a9198; margin-top:10px;">
Model: Random Forest | Version 1.0.0
</div>
""", unsafe_allow_html=True)

