# Fetal Health Classification using Machine Learning

<img src = "https://sl.bing.net/gzJZvfHCxfE">

Kesehatan janin selama masa kehamilan merupakan faktor penting untuk menjamin keselamatan ibu dan bayi.
Dalam praktik medis, dokter kandungan biasanya menggunakan alat Cardiotocography (CTG) untuk memantau kondisi janin.
CTG merekam detak jantung janin (fetal heart rate) dan aktivitas rahim (uterine contractions) untuk mendeteksi tanda-tanda stres atau gangguan pada janin.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpro-elofauq0.streamlit.app/)

📘 Project Overview

Proyek ini bertujuan untuk memprediksi kondisi kesehatan janin berdasarkan data Cardiotocography (CTG) menggunakan beberapa model machine learning seperti:
1. Decision Tree
2. Random Forest
3. XGBoost

Setelah proses evaluasi dan tuning, XGBoost terbukti menjadi model terbaik dalam mengklasifikasikan kondisi janin menjadi:
1 = Normal
2 = Suspect
3= Pathological

🧠 Dataset

Dataset yang digunakan berasal dari UCI Machine Learning Repository – Fetal Health Classification Dataset, yang berisi hasil pengukuran CTG (Cardiotocography).

- Jumlah data: 2126 observasi
- Jumlah fitur: 21 fitur numerik
- Target: fetal_health (3 kelas: Normal, Suspect, Pathological)

⚙️ Model Development

Model dievaluasi menggunakan beberapa algoritma dengan metrik:

- Accuracy
- F1-Score (Weighted)
- ROC-AUC

📊 Confusion Matrix

Model mampu mengklasifikasikan dengan baik pada semua kelas, terutama pada kelas Normal dan Pathological.
Kebingungan kecil terjadi pada kelas Suspect, yang sering tumpang tindih dengan kelas lain (karena kondisi borderline).

🧩 ROC Curve Insight

ROC Curve menunjukkan bahwa:

- Normal (AUC = 0.983)
- Suspect (AUC = 0.962)
- Pathological (AUC = 0.990)
Semua nilai AUC mendekati 1, menandakan model memiliki kemampuan klasifikasi yang sangat baik pada ketiga kelas.

🔍 Feature Importance

Fitur paling berpengaruh dalam prediksi kesehatan janin:

- mean_value_of_short_term_variability
- histogram_mean
- prolongued_decelerations
- percentage_of_time_with_abnormal_long_term_variability
- abnormal_short_term_variability

➡️ Artinya, variabilitas detak jantung janin dan perlambatan detak jantung menjadi faktor utama dalam menentukan kondisi janin.

🧾 Conclusion

- Model XGBoost (tuned) memberikan performa terbaik dengan akurasi tinggi dan kemampuan generalisasi yang kuat.
- Fitur-fitur terkait variabilitas detak jantung merupakan penentu utama kesehatan janin.
- Proyek ini dapat membantu tenaga medis dalam mendeteksi dini risiko janin tidak normal berdasarkan hasil CTG.
