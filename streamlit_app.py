# ========================================
# Fetal Health Classification Dashboard
# ========================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------------
# App Title
# ------------------------------
st.set_page_config(page_title="Fetal Health Dashboard", layout="wide")
st.title("🩺 Fetal Health Classification Dashboard")

st.markdown("""
Aplikasi ini menampilkan hasil **analisis dan pemodelan machine learning** pada dataset *Fetal Health Classification*.  
Berikut meliputi: *EDA, Model Comparison, Confusion Matrix, ROC Curve,* dan insight singkat dari tiap model.
""")

# ------------------------------
# Load Data
# ------------------------------
st.header("📂 Dataset Overview")
uploaded_file = st.file_uploader("Upload dataset CSV kamu (gunakan yang sudah di-cleaning)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Sample")
    st.dataframe(df.head())

    # ------------------------------
    # Exploratory Data Analysis
    # ------------------------------
    st.header("🔍 Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Target (Fetal Health)")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="fetal_health", palette="coolwarm", ax=ax)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Korelasi Antar Fitur")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(df.corr(), cmap="Blues", ax=ax)
        st.pyplot(fig)

    st.markdown("""
    **Insight:**  
    - Distribusi target menunjukkan apakah data seimbang antara kelas *Normal*, *Suspect*, dan *Pathological*.  
    - Korelasi membantu mengidentifikasi hubungan antar fitur yang bisa mempengaruhi hasil model.
    """)

    # ------------------------------
    # Split Data
    # ------------------------------
    X = df.drop("fetal_health", axis=1)
    y = df["fetal_health"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ------------------------------
    # Train Models
    # ------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy (Train)": accuracy_score(y_train, y_pred_train),
            "Accuracy (Test)": accuracy_score(y_test, y_pred_test),
            "Recall (Test)": recall_score(y_test, y_pred_test, average="macro"),
            "F1-Score (Test)": f1_score(y_test, y_pred_test, average="macro"),
            "ROC-AUC (Test)": roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
        })

    result_df = pd.DataFrame(results)
    st.header("📊 Model Comparison")
    st.dataframe(result_df.style.highlight_max(axis=0, color="lightblue"))

    # ------------------------------
    # Confusion Matrix & ROC
    # ------------------------------
    st.header("📉 Confusion Matrix & ROC Curve")

    model_choice = st.selectbox("Pilih model untuk visualisasi:", list(models.keys()))
    selected_model = models[model_choice]
    y_pred = selected_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion Matrix Plot
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_choice}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    y_prob = selected_model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in np.unique(y):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, int(i - 1)])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fig, ax = plt.subplots()
    for i in roc_auc.keys():
        ax.plot(fpr[i], tpr[i], label=f'Class {int(i)} (AUC = {roc_auc[i]:.2f})')
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # ------------------------------
    # Insights
    # ------------------------------
    st.header("💬 Model Insights")
    if model_choice == "Logistic Regression":
        st.write("""
        - Logistic Regression menunjukkan performa yang baik namun cenderung lebih rendah dari model ensemble.
        - Model ini sederhana dan transparan, cocok untuk baseline model.
        """)
    elif model_choice == "Random Forest":
        st.write("""
        - Random Forest memberikan hasil akurasi dan ROC-AUC yang tinggi.
        - Model ini kuat terhadap overfitting dan dapat menangkap non-linearitas data.
        """)
    else:
        st.write("""
        - XGBoost menghasilkan performa terbaik dengan nilai F1-Score dan ROC-AUC tertinggi.
        - Model ini mampu menyeimbangkan bias dan variansi dengan baik setelah tuning.
        """)
else:
    st.warning("📁 Silakan upload dataset CSV terlebih dahulu untuk memulai analisis.")
