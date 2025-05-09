import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit.components.v1 as components
import matplotlib
# Title
# 使用 Markdown 和 HTML 调整字体大小
st.markdown('<h2 style="font-size:20px;">XGBoost-Based HCC Recurrence Prediction Model Post-Liver Transplant</h2>', unsafe_allow_html=True)

AFP = st.number_input("AFP (µg/L):")
Ki67 = st.number_input("Ki67:")
Maximumdiameter = st.number_input("Maximumdiameter (cm):")
TSC2 = st.selectbox('TSC2', ['Low', 'High'])
TSC2 = 1 if TSC2 == 'High' else 0
MVI = st.selectbox('MVI', ['No', 'Yes'])
MVI = 1 if MVI == 'Yes' else 0
Satellite = st.selectbox('Satellite', ['No', 'Yes'])
Satellite = 1 if Satellite == 'Yes' else 0


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    XGB = joblib.load("XGB.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Store inputs into dataframe
    input_numerical = np.array([AFP, Ki67, MVI,Maximumdiameter, Satellite,TSC2]).reshape(1, -1)
    feature_names  = ['AFP', 'Ki67', 'MVI', 'Maximumdiameter', 'Satellite', 'TSC2']
    input_numericalyuan = pd.DataFrame(input_numerical, columns=feature_names)
    input_numerical = pd.DataFrame(input_numerical, columns=feature_names)

    input_numerical[['AFP','Maximumdiameter','Ki67']] = scaler.transform(input_numerical[['AFP','Maximumdiameter','Ki67']])

        # 使用模型进行预测概率
    prediction_proba = XGB.predict_proba(input_numerical)
    target_class_proba = prediction_proba[:, 1]
    target_class_proba_percent = (target_class_proba * 100).round(2)
    # 在 Streamlit 中显示结果，调整标题样式和内容
    st.markdown("## **Prediction Probabilities (%)**")
    for prob in target_class_proba_percent:
        st.markdown(f"**{prob:.2f}%**")

  
    explainer = shap.TreeExplainer(XGB)
    shap_values = explainer.shap_values(input_numerical)
    
    # SHAP Force Plot
    st.write("### SHAP Value Force Plot")
    shap.initjs()
    force_plot_visualizer = shap.plots.force(
        explainer.expected_value, shap_values, input_numericalyuan)
    # 将 force_plot 保存为一个临时 HTML 文件
    shap.save_html("force_plot.html", force_plot_visualizer)

# 读取 HTML 文件内容
    with open("force_plot.html", "r", encoding="utf-8") as html_file:
        html_content = html_file.read()

# 将 HTML 嵌入到 Streamlit 中
    components.html(html_content, height=400)
