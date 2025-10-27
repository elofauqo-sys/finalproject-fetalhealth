# =========================
# 📂 LOAD DATASET
# =========================
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Fetal Health Classification Dashboard", layout="wide")

st.title("👶 Fetal Health Classification Dashboard")
st.markdown("""
Kesehatan janin selama masa kehamilan merupakan faktor penting untuk menjamin keselamatan ibu dan bayi.
Dalam praktik medis, dokter kandungan biasanya menggunakan alat Cardiotocography (CTG) untuk memantau kondisi janin.
CTG merekam detak jantung janin (fetal heart rate) dan aktivitas rahim (uterine contractions) untuk mendeteksi tanda-tanda stres atau gangguan pada janin.

menggunakan dataset **fetal_health.csv**.
""")

# Upload dataset
uploaded_file = st.file_uploader("📤 Upload Dataset (CSV)", type=["csv"], key="file_uploader_dataset")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset berhasil diunggah!")
else:
    df = pd.read_csv("fetal_health.csv")
    st.info("ℹ️ Tidak ada file yang diunggah. Menggunakan dataset default: `fetal_health.csv`")

# Tampilkan preview dataset
st.subheader("🔍 Preview Dataset")
st.dataframe(df.head())

# Informasi dasar
st.subheader("📊 Informasi Dataset")
st.write(f"Jumlah baris: {df.shape[0]}")
st.write(f"Jumlah kolom: {df.shape[1]}")

st.write("Kolom dataset:")
st.write(list(df.columns))


# =========================
# 📈 EDA: Distribusi Kelas Target
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("📊 Distribusi Kelas Target (Fetal Health)")

if "fetal_health" in df.columns:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="fetal_health", data=df, palette="viridis", ax=ax)
    ax.set_title("Distribusi Kelas Target", fontsize=14)
    ax.set_xlabel("Kategori Kesehatan Janin")
    ax.set_ylabel("Jumlah Data")

    # Tampilkan jumlah tiap kelas di atas bar
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + 0.3, p.get_height() + 5))

    st.pyplot(fig)

    # Tampilkan distribusi numerik
    st.write("### 📋 Proporsi Kelas:")
    st.dataframe(df["fetal_health"].value_counts(normalize=True).rename("Proporsi (%)") * 100)
else:
    st.warning("Kolom `fetal_health` tidak ditemukan di dataset. Pastikan nama kolom benar.")
st.markdown("""
**🧩 Insight:**
Berdasarkan distribusi kategori kesehatan janin, mayoritas data berada pada kategori Normal, sedangkan kategori Suspect dan Pathological memiliki jumlah sampel yang jauh lebih sedikit.
Kondisi ini menunjukkan bahwa dataset tidak seimbang (imbalanced), yang perlu diperhatikan pada tahap modeling agar model tidak bias terhadap kelas mayoritas (Normal).
""")

# =========================
# 📊 EDA: Distribusi Fitur Numerik
# =========================
st.subheader("📈 Distribusi Fitur Numerik")

# Ambil hanya kolom numerik
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Jika ada kolom target di dalamnya, kita keluarkan
if "fetal_health" in num_cols:
    num_cols.remove("fetal_health")

# Plot distribusi setiap fitur numerik
for col in num_cols:
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.histplot(df[col], bins=30, kde=True, color='teal', ax=ax)
    ax.set_title(f"Distribusi {col}", fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

# Tampilkan statistik ringkas
st.write("### 📋 Statistik Ringkas Fitur Numerik:")
st.dataframe(df[num_cols].describe().T)
st.markdown("""
**🧩 Insight:**
Beberapa fitur seperti accelerations, fetal movement, light decelerations, dan prolongued decelerations, semuanya menunjukkan distribusi yang skewed ke kanan. Artinya, sebagian besar data memiliki nilai kecil, dan hanya sedikit yang besar. Secara medis ini wajar, karena dalam kondisi normal, janin tidak terlalu sering mengalami percepatan atau penurunan detak jantung ekstrem. Hanya sedikit kasus yang menunjukkan nilai tinggi dan itu bisa mengindikasikan janin yang sangat aktif atau justru mengalami stres.
""")


# =========================
# 🔗 EDA: Korelasi antar Fitur
# =========================
st.subheader("🔗 Korelasi antar Fitur")

# Hitung matriks korelasi
corr_matrix = df.corr(numeric_only=True)

# Tampilkan heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=False,        # ubah ke True kalau mau nilai korelasi muncul di setiap kotak
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
ax.set_title("Heatmap Korelasi antar Fitur", fontsize=14)
st.pyplot(fig)

# Tampilkan fitur yang paling berkorelasi dengan target (fetal_health)
st.write("### 📊 Korelasi terhadap Fetal Health:")
if "fetal_health" in corr_matrix.columns:
    corr_target = corr_matrix["fetal_health"].sort_values(ascending=False)
    st.dataframe(corr_target)
else:
    st.write("Kolom target 'fetal_health' tidak ditemukan dalam data numerik.")
st.markdown("""
**🧩 Insight:**
Korelasi yang sangat tinggi antara histogram mean, histogram median, dan histogram mode (0.89–0.95) menunjukkan bahwa ketiganya menyimpan informasi medis yang hampir identik, yaitu pusat distribusi detak jantung janin (FHR). Kondisi ini menggambarkan kestabilan pola detak jantung janin, di mana rata-rata, nilai tengah, dan nilai yang paling sering muncul hampir sama. Namun, secara analisis statistik, hubungan yang terlalu tinggi juga menandakan potensi multikolinearitas.
Korelasi positif yang kuat antara baseline value dengan histogram mean dan median menunjukkan konsistensi fisiologis dari detak jantung janin. Secara medis, ini wajar karena ketiganya menggambarkan aspek yang sama yaitu tingkat rata-rata aktivitas jantung janin.

""")


from sklearn.model_selection import train_test_split

# Pisahkan fitur dan target
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================================
# 📦 MODELING - LOGISTIC REGRESSION
# ================================
st.subheader("🧠 Modeling: Logistic Regression")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

# === 1️⃣ Pisahkan fitur dan target ===
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"]

# === 2️⃣ Split data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === 3️⃣ Standarisasi data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 4️⃣ Inisialisasi & Training Model ===
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

# === 5️⃣ Prediksi ===
y_train_pred = log_model.predict(X_train_scaled)
y_test_pred = log_model.predict(X_test_scaled)

# === 6️⃣ Evaluasi Metrik ===
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
roc_auc = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled), multi_class='ovr')

# === 7️⃣ Tampilkan Hasil Metrik ===
metrics_data = {
    "Set": ["Train", "Test"],
    "Accuracy": [train_acc, test_acc],
    "F1-Score": [train_f1, test_f1],
    "ROC-AUC": [roc_auc, roc_auc]
}
metrics_df = pd.DataFrame(metrics_data)

st.write("### 📊 Hasil Evaluasi Model")
# Format hanya kolom numerik
# Format hanya kolom numerik
numeric_cols = ["Accuracy", "F1-Score", "ROC-AUC"]
st.dataframe(metrics_df.style.format({col: "{:.4f}" for col in numeric_cols}))

# === 8️⃣ Confusion Matrix ===
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
ax.set_title("Confusion Matrix - Logistic Regression")
st.pyplot(fig)
st.markdown("""
**🧩 Insight:**
Model Logistic Regression menunjukkan performa yang baik dengan F1-score 0.885 dan ROC-AUC 0.961, menandakan model cukup stabil tanpa overfitting.
Model sangat akurat dalam mengenali kelas Normal, namun masih sering membingungkan kelas Suspect dan Pathological, yang memiliki karakteristik fitur serupa.
""")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 🧠 TRAIN RANDOM FOREST MODEL
# --------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=None, 
    random_state=42
)
rf_model.fit(X_train, y_train)

# Prediksi
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# --------------------------------------------
# 📊 EVALUASI MODEL
# --------------------------------------------
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_test), rf_model.predict_proba(X_test), multi_class='ovr')

# Buat dataframe metrik
rf_metrics_df = pd.DataFrame({
    "Set": ["Train", "Test"],
    "Accuracy": [train_acc, test_acc],
    "F1-Score": [f1, f1],
    "ROC-AUC": [roc_auc, roc_auc]
})

st.subheader("🌲 Hasil Evaluasi Model: Random Forest")
numeric_cols = ["Accuracy", "F1-Score", "ROC-AUC"]
st.dataframe(rf_metrics_df.style.format({col: "{:.4f}" for col in numeric_cols}))

# --------------------------------------------
# 🔍 CONFUSION MATRIX
# --------------------------------------------
st.subheader("🔹 Confusion Matrix Random Forest")
cm = confusion_matrix(y_test, y_test_pred)
fig_cm, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix - Random Forest")
st.pyplot(fig_cm)
st.markdown("""
**🧩 Insight:**
Akurasi Tinggi:
Model menunjukkan performa sangat baik dengan akurasi 92.49% pada data test, menandakan kemampuan prediksi yang kuat terhadap data baru.

F1-Score & ROC-AUC:
Nilai F1-Score (0.922) dan ROC-AUC (0.9787) menunjukkan keseimbangan antara presisi dan recall yang sangat baik serta kemampuan model dalam membedakan tiap kelas (Normal, Suspect, Pathological).

Confusion Matrix:

Kelas Normal (0) terprediksi sangat akurat (325 benar dari 332).

Kelas Suspect (1) dan Pathological (2) memiliki sedikit kesalahan klasifikasi, tapi performanya tetap konsisten.

Sebagian kecil kasus Suspect kadang tertukar dengan Normal, menunjukkan model sedikit bias terhadap kelas mayoritas.
""")


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# XGBOOST MODELING
# =========================
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

st.subheader("🚀 XGBoost Model")

# Buat model
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    objective='multi:softprob',
    num_class=3
)

# Ubah label dari [1,2,3] → [0,1,2]
y_train_adj = y_train - 1
y_test_adj = y_test - 1

# Latih model
xgb_model.fit(X_train, y_train_adj)

# Prediksi
y_train_pred = xgb_model.predict(X_train) + 1
y_test_pred = xgb_model.predict(X_test) + 1

# Evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_test), xgb_model.predict_proba(X_test), multi_class='ovr')

# Tampilkan metrik
st.write("### 📊 Hasil Evaluasi Model")
metrics_df = pd.DataFrame({
    "Metrik": ["Akurasi (Train)", "Akurasi (Test)", "F1-score (Weighted)", "ROC-AUC (OvR)"],
    "Nilai": [train_acc, test_acc, f1, roc_auc]
})
st.dataframe(metrics_df)

# Confusion matrix
st.write("### 🔹 Confusion Matrix")
cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
st.markdown("""
**🧩 Insight Model

Akurasi Train: 0.9994

Akurasi Test: 0.9413

F1-score (Weighted): 0.9397

ROC-AUC (OvR): 0.9816

👉 Model memiliki performa sangat baik, dengan akurasi dan AUC yang tinggi di data uji.
Perbedaan kecil antara akurasi train dan test menunjukkan model tidak mengalami overfitting signifikan.
F1-score yang tinggi menandakan keseimbangan antara presisi dan recall.

🧩 Insight Confusion Matrix

Kelas 0 (Normal): Mempunyai prediksi yang paling akurat (325 benar dari total data kelas ini).

Kelas 1 (Suspect): Sebagian kecil masih tertukar dengan kelas 0.

Kelas 2 (Pathological): Hampir semua terdeteksi dengan benar (32 benar, hanya 3 salah klasifikasi).

👉 Secara keseluruhan, model mampu membedakan kondisi janin dengan baik, terutama dalam mengidentifikasi kasus Normal dan Pathological, meskipun masih ada sedikit kesalahan pada kelas Suspect.
""")


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
import numpy as np

st.subheader("🚀 XGBoost Classifier dengan Hyperparameter Tuning")

# Pastikan label mulai dari 0 untuk XGBoost
y_train_adj = y_train - 1
y_test_adj = y_test - 1

# ---- PARAMETER GRID ----
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_lambda': [0.1, 1, 5, 10]
}

# ---- MODEL DASAR ----
xgb_base = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=42,
    n_jobs=-1
)

# ---- RANDOMIZED SEARCH ----
st.write("🔍 Sedang melakukan tuning parameter... Tunggu sebentar ya.")
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=25,
    scoring='f1_weighted',
    cv=3,
    verbose=0,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train_adj)

best_model = random_search.best_estimator_

st.success("✅ Hyperparameter tuning selesai!")
st.write("### 🔹 Parameter Terbaik Ditemukan:")
st.json(random_search.best_params_)

# ---- PREDIKSI ----
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)

# ---- EVALUASI ----
acc_best = accuracy_score(y_test_adj, y_pred_best)
f1_best = f1_score(y_test_adj, y_pred_best, average='weighted')
roc_auc_best = roc_auc_score(
    label_binarize(y_test_adj, classes=[0,1,2]),
    y_proba_best,
    average='weighted',
    multi_class='ovr'
)

st.write("### 📊 Hasil Evaluasi Model Terbaik")
st.metric("Accuracy", f"{acc_best:.4f}")
st.metric("F1-Score (weighted)", f"{f1_best:.4f}")
st.metric("ROC AUC (weighted)", f"{roc_auc_best:.4f}")

# ---- CONFUSION MATRIX ----
cm = confusion_matrix(y_test_adj, y_pred_best)
st.write("### 🔸 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Suspect', 'Pathological'],
            yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - XGBoost")
st.pyplot(fig)
st.markdown("""
**🧩 Insight:**
Insight:
Model XGBoost yang telah di-tuning menunjukkan performa yang sangat baik dengan akurasi tinggi (93,66%) dan AUC yang mendekati sempurna (0,98).
Nilai F1-score yang tinggi juga menunjukkan bahwa model mampu menjaga keseimbangan antara presisi dan recall di ketiga kelas (Normal, Suspect, Pathological).
            Confusion Matrix (XGBoost - Setelah Tuning)

Kelas Normal terprediksi dengan sangat baik (325 benar dari total 332), menandakan kemampuan model dalam mengenali kondisi normal dengan sangat tinggi.

Kelas Suspect masih memiliki beberapa kesalahan prediksi (15 salah klasifikasi menjadi Normal), menunjukkan bahwa kondisi borderline lebih sulit dipisahkan.

Kelas Pathological juga menunjukkan performa bagus (32 benar dari 35), menandakan model cukup sensitif terhadap kasus patologis.
""")


# === ROC CURVE MULTICLASS ===
st.write("### 🧭 ROC Curve per Class (1, 2, 3)")
y_test_bin = label_binarize(y_test, classes=[1, 2, 3])

fpr, tpr, roc_auc = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_best[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fig2, ax2 = plt.subplots()
colors = ['blue', 'orange', 'green']
labels = [
    'Normal (Class 1)',
    'Suspect (Class 2)',
    'Pathological (Class 3)'
]

for i, color, label in zip(range(3), colors, labels):
    ax2.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f"{label} (AUC = {roc_auc[i]:.3f})")

ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve - XGBoost (Multi-Class)')
ax2.legend(loc="lower right")
st.pyplot(fig2)
st.markdown("""
**🧩 Insight:**
Semua nilai AUC di atas 0.96, artinya model sangat baik dalam membedakan ketiga kelas.

Model paling presisi untuk kelas Pathological, diikuti oleh Normal, sementara Suspect sedikit lebih sulit dipisahkan — hal ini wajar karena karakteristik fisiologis kelas Suspect sering berada di tengah antara kondisi normal dan patologis.

Secara keseluruhan, kurva ROC ini menunjukkan bahwa model XGBoost hasil tuning memiliki kinerja klasifikasi yang sangat kuat dan andal.
""")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- Pastikan model sudah ada ---
st.subheader("🔍 Feature Importance - XGBoost (Hasil Tuning)")

importance = best_model.feature_importances_
features = X_train.columns

imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plot Feature Importance
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=imp_df,
    palette='viridis'
)
plt.title("XGBoost Feature Importance (Model Tuned)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

st.pyplot(fig)
st.dataframe(imp_df)
st.markdown("""
**🧩 Insight:**
Fitur-fitur yang berkaitan dengan variabilitas detak jantung janin (short & long term variability) dan decelerations paling menentukan dalam prediksi kondisi janin.

Fitur-fitur seperti uterine_contractions dan baseline_value juga memberikan kontribusi tambahan, tetapi tidak sebesar faktor variabilitas.

Secara keseluruhan, model XGBoost menilai bahwa pola perubahan detak jantung janin dari waktu ke waktu adalah sinyal utama untuk membedakan janin Normal, Suspect, dan Pathological.
""")
