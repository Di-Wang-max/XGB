import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import streamlit as st
data = pd.read_csv("D:/博士/李灿明综述/临床数据/datac.csv", encoding='utf-8')
X = data.drop(columns=['VTE'])
y = data['VTE']
# 假设 X 为特征矩阵，y 为标签
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=123
)
scaler = StandardScaler()
X_train[['AFP','Maximumdiameter','Ki67']] = scaler.fit_transform(X_train[['AFP','Maximumdiameter','Ki67']])
X_test[['AFP','Maximumdiameter','Ki67']] = scaler.transform(X_test[['AFP','Maximumdiameter','Ki67']])#用训练集的参数转换验证集
#进行排序
X_train = X_train.reindex(sorted(X_train.columns), axis=1)
X_test = X_test.reindex(sorted(X_test.columns), axis=1)
XGB = xgb.XGBClassifier(booster='dart',
                            learning_rate=0.191726487719208,
                            max_depth=14,
                            min_child_weight= 0.024777350617292,
                            subsample=0.6412914945062665,
                            colsample_bytree= 0.6724114076325344, 
                            reg_lambda=7.64101174891213,
                            n_estimators=395,
                            scale_pos_weight= 1.395887616884788,
                   random_state=42)
XGB.fit(X_train, y_train)

import joblib

joblib.dump(XGB, "XGB.pkl")
joblib.dump(scaler, "scaler.pkl")