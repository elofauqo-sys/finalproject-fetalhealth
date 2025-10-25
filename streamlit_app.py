# ===============================================
# Fetal Health Classification Dashboard
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

# ===============================================
# 1. Load Dataset
# ===============================================
st.title("🩺 Fetal Health Classification Dashboard")

st.write("Aplikasi ini menampilkan EDA, training model XGBoost, dan prediksi kesehatan janin berdasarkan data CTG (Cardiotocography).")

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload Dataset (fetal_health.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.stop()

st.subheader("📊 Preview Data")
st.dataframe(df.head())

# ===============================================
# 2. EDA
# ===============================================
st.header("🔍 Exploratory Data Analysis")

# Distribusi kelas
st.subheader("Distribusi Kelas Fetal Health")
fig, ax = plt.subplots()
sns.countplot(data=df, x="fetal_health", palette="Set2", ax=ax)
st.pyplot(fig)

# Korelasi antar fitur
st.subheader("Heatmap Korelasi Fitur")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, annot=False)
st.pyplot(fig)

# Outlier
st.subheader("Deteksi Outlier (contoh: baseline value)")
fig, ax = plt.subplots()
sns.boxplot(data=df, x="baseline value", ax=ax, color="lightblue")
st.pyplot(fig)

# ===============================================
# 3. Training Model XGBoost
# ===============================================
st.header("🤖 Training Model XGBoost")

# Split data
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"].astype(int) - 1  # ubah jadi 0,1,2 untuk XGBoost

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
acc = accuracy_score(y_test, y_pred)
st.metric(label="🎯 Akurasi Model", value=f"{acc*100:.2f}%")

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Suspect", "Pathological"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
st.pyplot(fig)

# ===============================================
# 4. Prediksi dari Input User
# ===============================================
st.header("🧮 Prediksi Fetal Health Berdasarkan Input")

st.write("Isi nilai parameter di bawah ini untuk memprediksi kondisi janin:")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]
    label_map = {0: "Normal", 1: "Suspect", 2: "Pathological"}
    st.success(f"Hasil Prediksi: **{label_map[pred]}**")

# ===============================================
st.caption("Made with ❤️ by Elok Fauqo Himmah")
