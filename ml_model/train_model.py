import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. MENENTUKAN LOKASI DATASET
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    BASE_DIR
    / "prediction"
    / "datasets"
    / "indonesia_kesehatan_mental_gen_z_akademik_enriched_2026.csv"
)

# ==========================================
# 2. MEMBACA DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("Dataset berhasil dibaca!")
print("Jumlah data:", len(df))
print("Jumlah kolom:", len(df.columns))

# ==========================================
# 3. MENENTUKAN FITUR DAN TARGET
# ==========================================

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

print("\nJumlah fitur:", len(features))
print("Distribusi target:")
print(y.value_counts())

# ==========================================
# 4. MEMISAHKAN FITUR NUMERIK DAN KATEGORIKAL
# ==========================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nFitur kategorikal:", categorical_features)
print("Fitur numerik:", numeric_features)

# ==========================================
# 5. PREPROCESSING DATA
# ==========================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 6. MEMBAGI DATA TRAINING DAN TESTING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

# ==========================================
# 7. MEMBUAT PIPELINE MODEL
# ==========================================

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=42,
        max_depth=8,
        class_weight="balanced"
    ))
])

# ==========================================
# 8. MELATIH MODEL
# ==========================================

print("\nSedang melatih model...")
model.fit(X_train, y_train)

print("Pelatihan model selesai!")

# ==========================================
# 9. MELAKUKAN PREDIKSI DATA TESTING
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 10. EVALUASI MODEL
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("HASIL EVALUASI MODEL")
print("===================================")

print(f"Akurasi: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# 11. MENYIMPAN MODEL
# ==========================================

MODEL_DIR = BASE_DIR / "ml_model" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "decision_tree_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel berhasil disimpan!")
print("Lokasi:", MODEL_PATH)