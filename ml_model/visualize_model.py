import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================
# 1. MENENTUKAN LOKASI FILE
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR / "prediction" / "datasets" /
    "indonesia_kesehatan_mental_gen_z_akademik_enriched_2026.csv"
)

MODEL_PATH = (
    BASE_DIR / "ml_model" / "saved_models" /
    "decision_tree_model.pkl"
)

OUTPUT_DIR = BASE_DIR / "ml_model" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# 2. MEMBACA DATASET DAN MODEL
# =====================================

df = pd.read_csv(DATASET_PATH)
model = joblib.load(MODEL_PATH)

target_column = "kategori_keparahan_depresi_kecemasan"

features = [
    "jenjang_dan_rumpun_studi",
    "status_perguruan_tinggi",
    "provinsi_domisili_kampus",
    "semester_studi",
    "golongan_ukt",
    "indeks_prestasi_kumulatif_ipk",
    "beban_sks_semester_ini",
    "durasi_layar_harian_jam",
    "durasi_tidur_harian_jam",
    "kondisi_tempat_tinggal",
    "pengeluaran_bulanan_total_idr",
    "aktivitas_pekerjaan_sampingan",
    "skor_tekanan_finansial",
    "skor_kecemasan_akademik",
    "skor_tekanan_masa_depan_karir",
    "skor_kesepian_dan_isolasi_sosial",
    "frekuensi_keluhan_somatik_mingguan",
    "akses_layanan_konseling"
]

X = df[features]
y = df[target_column]

# Membuat pembagian data yang sama dengan pelatihan
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

# Urutan kategori dibuat konsisten
labels = [
    "Berat / Kritis",
    "Minimal",
    "Ringan",
    "Sedang"
]

print("Data dan model berhasil dimuat!")

# =====================================
# 3. VISUALISASI DISTRIBUSI KATEGORI
# =====================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x=target_column,
    order=labels
)

plt.title("Distribusi Kategori Kesehatan Mental")
plt.xlabel("Kategori")
plt.ylabel("Jumlah Data")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "distribusi_kategori.png",
    dpi=300
)
plt.close()

# =====================================
# 4. VISUALISASI CONFUSION MATRIX
# =====================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix Decision Tree")
plt.xlabel("Prediksi Model")
plt.ylabel("Kategori Sebenarnya")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)
plt.close()

# =====================================
# 5. VISUALISASI METRIK EVALUASI
# =====================================

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

metrics = pd.DataFrame({
    "Precision": [
        report[label]["precision"] for label in labels
    ],
    "Recall": [
        report[label]["recall"] for label in labels
    ],
    "F1-Score": [
        report[label]["f1-score"] for label in labels
    ]
}, index=labels)

plt.figure(figsize=(10, 6))

metrics.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=15
)

plt.title("Evaluasi Performa Model Decision Tree")
plt.xlabel("Kategori")
plt.ylabel("Nilai Evaluasi")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "evaluasi_model.png",
    dpi=300
)
plt.close()

# =====================================
# 6. VISUALISASI FEATURE IMPORTANCE
# =====================================

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Fitur": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Fitur"
)

plt.title("15 Fitur Paling Berpengaruh pada Model")
plt.xlabel("Nilai Feature Importance")
plt.ylabel("Fitur")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)
plt.close()

# =====================================
# 7. SELESAI
# =====================================

print("\nSemua visualisasi berhasil dibuat!")
print("Lokasi penyimpanan:", OUTPUT_DIR)

print("\nDaftar file:")
for file in OUTPUT_DIR.glob("*.png"):
    print("-", file.name)
