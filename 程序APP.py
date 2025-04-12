import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # 导入SHAP库
import matplotlib.pyplot as plt

# 加载保存的XGBoost模型
model = joblib.load('XGB.pkl')

# 更新后的特征范围定义，假设所有蛋白质值已归一化到[-10, 10]范围
feature_ranges = {
    "Plasma GDF15": {"type": "numerical"},
    "Age": {"type": "numerical"},
    "Systolic Blood Pressure": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"},
    "Plasma NTproBNP": {"type": "numerical"},
    "Non Cancer Illness Count": {"type": "numerical"},
    "Sex": {"type": "categorical"},
    "Plasma AGER": {"type": "numerical"},
    "Plasma PRSS8": {"type": "numerical"},
    "Plasma PSPN": {"type": "numerical"},
    "CHOL RATIO": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma LPA": {"type": "numerical"},
    "Plasma CXCL17": {"type": "numerical"},
    "Long Standing Illness Disability": {"type": "categorical"},
    "Number of treatments medications taken": {"type": "numerical"},
    "Plasma GAST": {"type": "numerical"},
    "Plasma RGMA": {"type": "numerical"},
    "Plasma EPHA4": {"type": "numerical"},
}

# Streamlit 界面标题
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        # 数值型输入框，没有范围限制
        value = st.number_input(
            label=f"{feature}",
            value=0.0,  # 默认值为0
        )
    elif properties["type"] == "categorical":
        # 类别型选择框，假设类别为0和1
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=[0, 1],  # 这里假设类别为0和1，实际情况可以修改
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与SHAP可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用Matplotlib渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',  # 使用Times New Roman字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 打印SHAP值的形状，确保它是二维的
    print(f"SHAP values shape: {np.shape(shap_values)}")
    
    # 获取Class 1（正类）的SHAP值，二分类问题时，shap_values会包含两个元素，分别对应两个类别
    shap_values_class_1 = shap_values # 获取正类（Class 1）的SHAP值
    expected_value_class_1 = explainer.expected_value  # 获取Class 1的期望值

    # 生成Class 1（正类）的SHAP力图
    shap_fig = shap.force_plot(
        expected_value_class_1,  # 使用Class 1的期望值
        shap_values_class_1,  # Class 1的SHAP值
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),  # 输入特征数据
        matplotlib=True,
    )

    # 保存并显示SHAP力图
    plt.savefig("shap_force_plot_class_1.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_class_1.png")
