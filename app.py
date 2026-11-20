import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

# ==========================================
# 1. KONFIGURASI APLIKASI
# ==========================================

st.set_page_config(
    page_title="PsycoPredict",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR
    / "prediction"
    / "datasets"
    / "indonesia_kesehatan_mental_gen_z_akademik_enriched_2026.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "ml_model"
    / "saved_models"
    / "decision_tree_model.pkl"
)

# ==========================================
# 2. MEMUAT DATASET DAN MODEL
# ==========================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_dataset()
model = load_model()

# ==========================================
# 3. MENENTUKAN FITUR
# ==========================================

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

# ==========================================
# 4. JUDUL DAN DESKRIPSI
# ==========================================

st.title("🧠 PsycoPredict")

st.subheader(
    "Sistem Prediksi Tingkat Keparahan "
    "Depresi dan Kecemasan Mahasiswa"
)

st.write(
    """
    PsycoPredict merupakan aplikasi berbasis Machine Learning
    yang menggunakan algoritma Decision Tree untuk memperkirakan
    kategori keparahan depresi dan kecemasan berdasarkan data
    akademik, aktivitas, dan kesejahteraan mahasiswa.
    """
)

st.info(
    "Hasil prediksi merupakan estimasi berdasarkan model dan dataset, "
    "bukan diagnosis medis atau psikologis."
)

st.divider()

# ==========================================
# 5. FORM INPUT PREDIKSI
# ==========================================

st.header("📝 Form Prediksi")

with st.form("prediction_form"):

    st.subheader("A. Data Akademik dan Perguruan Tinggi")

    col1, col2 = st.columns(2)

    with col1:
        jenjang = st.selectbox(
            "Jenjang dan Rumpun Studi",
            sorted(df["jenjang_dan_rumpun_studi"].dropna().unique())
        )

        status_kampus = st.selectbox(
            "Status Perguruan Tinggi",
            sorted(df["status_perguruan_tinggi"].dropna().unique())
        )

        provinsi = st.selectbox(
            "Provinsi Domisili Kampus",
            sorted(df["provinsi_domisili_kampus"].dropna().unique())
        )

        semester = st.number_input(
            "Semester Studi",
            min_value=1,
            max_value=20,
            value=2
        )

    with col2:
        ukt = st.selectbox(
            "Golongan UKT",
            sorted(df["golongan_ukt"].dropna().unique())
        )

        ipk = st.number_input(
            "Indeks Prestasi Kumulatif (IPK)",
            min_value=0.0,
            max_value=4.0,
            value=3.0,
            step=0.01
        )

        sks = st.number_input(
            "Beban SKS Semester Ini",
            min_value=0,
            max_value=30,
            value=20
        )

    st.divider()

    st.subheader("B. Aktivitas dan Kebiasaan Harian")

    col1, col2 = st.columns(2)

    with col1:
        durasi_layar = st.number_input(
            "Durasi Layar Harian (Jam)",
            min_value=0.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )

        durasi_tidur = st.number_input(
            "Durasi Tidur Harian (Jam)",
            min_value=0.0,
            max_value=24.0,
            value=7.0,
            step=0.5
        )

        tempat_tinggal = st.selectbox(
            "Kondisi Tempat Tinggal",
            sorted(df["kondisi_tempat_tinggal"].dropna().unique())
        )

    with col2:
        pengeluaran = st.number_input(
            "Pengeluaran Bulanan Total (IDR)",
            min_value=0,
            value=1000000,
            step=100000
        )

        pekerjaan = st.selectbox(
            "Aktivitas Pekerjaan Sampingan",
            sorted(df["aktivitas_pekerjaan_sampingan"].dropna().unique())
        )

    st.divider()

    st.subheader("C. Kondisi dan Kesejahteraan Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        tekanan_finansial = st.number_input(
            "Skor Tekanan Finansial",
            value=0.0,
            step=0.1
        )

        kecemasan_akademik = st.number_input(
            "Skor Kecemasan Akademik",
            value=0.0,
            step=0.1
        )

        tekanan_karir = st.number_input(
            "Skor Tekanan Masa Depan Karir",
            value=0.0,
            step=0.1
        )

    with col2:
        kesepian = st.number_input(
            "Skor Kesepian dan Isolasi Sosial",
            value=0.0,
            step=0.1
        )

        keluhan_somatik = st.number_input(
            "Frekuensi Keluhan Somatik Mingguan",
            value=0.0,
            step=0.1
        )

        konseling = st.selectbox(
            "Akses Layanan Konseling",
            sorted(df["akses_layanan_konseling"].dropna().unique())
        )

    submit = st.form_submit_button(
        "🔮 Proses Prediksi",
        use_container_width=True
    )

# ==========================================
# 6. PROSES PREDIKSI
# ==========================================

if submit:

    input_data = {
        "jenjang_dan_rumpun_studi": jenjang,
        "status_perguruan_tinggi": status_kampus,
        "provinsi_domisili_kampus": provinsi,
        "semester_studi": semester,
        "golongan_ukt": ukt,
        "indeks_prestasi_kumulatif_ipk": ipk,
        "beban_sks_semester_ini": sks,
        "durasi_layar_harian_jam": durasi_layar,
        "durasi_tidur_harian_jam": durasi_tidur,
        "kondisi_tempat_tinggal": tempat_tinggal,
        "pengeluaran_bulanan_total_idr": pengeluaran,
        "aktivitas_pekerjaan_sampingan": pekerjaan,
        "skor_tekanan_finansial": tekanan_finansial,
        "skor_kecemasan_akademik": kecemasan_akademik,
        "skor_tekanan_masa_depan_karir": tekanan_karir,
        "skor_kesepian_dan_isolasi_sosial": kesepian,
        "frekuensi_keluhan_somatik_mingguan": keluhan_somatik,
        "akses_layanan_konseling": konseling
    }

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]

    probability_df = pd.DataFrame({
        "Kategori": model.classes_,
        "Probabilitas (%)": probabilities * 100
    })

    probability_df = probability_df.sort_values(
        by="Probabilitas (%)",
        ascending=False
    )

    # ======================================
    # 7. MENAMPILKAN HASIL
    # ======================================

    st.divider()
    st.header("📊 Hasil Prediksi")

    st.success(
        f"Kategori prediksi: {prediction}"
    )

    st.subheader("Distribusi Probabilitas Model")

    chart_data = probability_df.set_index("Kategori")

    st.bar_chart(chart_data)

    st.dataframe(
        probability_df.style.format({
            "Probabilitas (%)": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Probabilitas merupakan keluaran model untuk masing-masing "
        "kategori, bukan jaminan kebenaran prediksi."
    )

    st.warning(
        "Hasil ini hanya estimasi berbasis Machine Learning "
        "dan tidak dapat digunakan sebagai diagnosis klinis. "
        "Untuk penilaian kondisi kesehatan mental, konsultasikan "
        "dengan tenaga profesional."
    )

# ==========================================
# 8. VISUALISASI DATASET
# ==========================================

st.divider()

st.header("📈 Visualisasi Dataset")

st.write(
    "Distribusi kategori keparahan depresi dan kecemasan "
    "pada dataset yang digunakan."
)

distribution = (
    df["kategori_keparahan_depresi_kecemasan"]
    .value_counts()
    .rename_axis("Kategori")
    .reset_index(name="Jumlah Data")
)

st.bar_chart(
    distribution.set_index("Kategori")
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", f"{len(df):,}")

with col2:
    st.metric("Jumlah Fitur", len(features))

with col3:
    st.metric(
        "Jumlah Kategori",
        df["kategori_keparahan_depresi_kecemasan"].nunique()
    )

st.divider()

st.caption(
    "PsycoPredict | Aplikasi Prediksi Kesehatan Mental Mahasiswa "
    "Berbasis Machine Learning"
)
