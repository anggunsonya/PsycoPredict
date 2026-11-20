from django.db.models import Count
from django.db.models.functions import TruncDate
from .models import PredictionHistory
import json
import pandas as pd
import joblib

from pathlib import Path
from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from .ai_service import generate_ai_analysis

# =====================================
# 1. MENENTUKAN LOKASI FILE
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

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

TARGET_COLUMN = "kategori_keparahan_depresi_kecemasan"

FEATURES = [
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

LABELS = [
    "Berat / Kritis",
    "Minimal",
    "Ringan",
    "Sedang"
]

# =====================================
# 2. HALAMAN HOME
# =====================================

def home(request):
    return render(request, "prediction/home.html")

# =====================================
# 3. HALAMAN DASHBOARD
# =====================================



def dashboard(request):

    # Membaca dataset
    df = pd.read_csv(DATASET_PATH)

    total_data = len(df)
    category_counts = df[TARGET_COLUMN].value_counts()

    # =====================================
    # 1. DISTRIBUSI KATEGORI DATASET
    # =====================================

    categories = [
        "Minimal",
        "Ringan",
        "Sedang",
        "Berat / Kritis"
    ]

    category_data = []

    for category in categories:
        count = int(category_counts.get(category, 0))

        percentage = (
            count / total_data * 100
            if total_data else 0
        )

        category_data.append({
            "name": category,
            "count": count,
            "percentage": round(percentage, 2),
        })

    # =====================================
    # 2. DATA SCATTER PLOT
    # =====================================

    scatter_columns = [
        "durasi_tidur_harian_jam",
        "skor_kecemasan_akademik",
        TARGET_COLUMN
    ]

    scatter_df = df[scatter_columns].dropna()

    # Menyiapkan ringkasan berdasarkan seluruh data
    scatter_summary = []

    for category in categories:
        category_df = scatter_df[
            scatter_df[TARGET_COLUMN] == category
        ]

        scatter_summary.append({
            "kategori": category,
            "jumlah_data": len(category_df),
            "rata_rata_durasi_tidur": (
                round(
                    category_df[
                        "durasi_tidur_harian_jam"
                    ].mean(), 2
                )
                if len(category_df) else 0
            ),
            "rata_rata_skor_kecemasan": (
                round(
                    category_df[
                        "skor_kecemasan_akademik"
                    ].mean(), 2
                )
                if len(category_df) else 0
            )
        })

    correlation = (
        scatter_df[
            [
                "durasi_tidur_harian_jam",
                "skor_kecemasan_akademik"
            ]
        ].corr().iloc[0, 1]
        if len(scatter_df) > 1 else None
    )

    scatter_analysis_data = {
        "total_data_dianalisis": len(scatter_df),
        "korelasi_durasi_tidur_dan_kecemasan": (
            round(correlation, 3)
            if pd.notna(correlation) else None
        ),
        "ringkasan_per_kategori": scatter_summary
    }

    # Mengambil maksimal 500 sampel untuk visualisasi
    if len(scatter_df) > 500:
        scatter_plot_df = scatter_df.sample(
            n=500,
            random_state=42
        )
    else:
        scatter_plot_df = scatter_df

    category_colors = {
        "Minimal": "#22c55e",
        "Ringan": "#60a5fa",
        "Sedang": "#f59e0b",
        "Berat / Kritis": "#ef4444"
    }

    scatter_datasets = []

    for category in categories:

        category_df = scatter_plot_df[
            scatter_plot_df[TARGET_COLUMN] == category
        ]

        points = [
            {
                "x": float(row["durasi_tidur_harian_jam"]),
                "y": float(row["skor_kecemasan_akademik"])
            }
            for _, row in category_df.iterrows()
        ]

        scatter_datasets.append({
            "label": category,
            "data": points,
            "backgroundColor": category_colors[category]
        })

    # =====================================
    # 3. DISTRIBUSI HASIL PREDIKSI PENGGUNA
    # =====================================

    prediction_categories = [
        "Minimal",
        "Ringan",
        "Sedang",
        "Berat / Kritis"
    ]

    distribution_query = (
        PredictionHistory.objects
        .values("result")
        .annotate(total=Count("id"))
    )

    distribution_counts = {
        item["result"]: item["total"]
        for item in distribution_query
    }

    prediction_distribution = [
        {
            "name": category,
            "count": distribution_counts.get(category, 0)
        }
        for category in prediction_categories
    ]

    # =====================================
    # 4. TREN JUMLAH PREDIKSI
    # =====================================

    trend_query = (
        PredictionHistory.objects
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(total=Count("id"))
        .order_by("date")
    )

    prediction_trend = [
        {
            "date": item["date"].strftime("%d-%m-%Y"),
            "count": item["total"]
        }
        for item in trend_query
    ]

    # =====================================
    # 5. RINGKASAN AKTIVITAS PREDIKSI
    # =====================================

    total_prediksi = PredictionHistory.objects.count()

    kategori_terbanyak = (
        PredictionHistory.objects
        .values("result")
        .annotate(total=Count("id"))
        .order_by("-total")
        .first()
    )

    prediksi_terbaru = PredictionHistory.objects.first()

    context_summary = {
        "total_prediksi": total_prediksi,
        "kategori_terbanyak": (
            kategori_terbanyak["result"]
            if kategori_terbanyak else "-"
        ),
        "prediksi_terbaru": prediksi_terbaru,
    }

    # =====================================
    # 6. DATA UNTUK ANALISIS AI
    # =====================================

    prediction_distribution_data = {
        "total_prediksi": total_prediksi,
        "distribusi_hasil_prediksi": prediction_distribution
    }

    if prediction_trend:
        peak_date = max(
            prediction_trend,
            key=lambda item: item["count"]
        )

        trend_summary = {
            "total_prediksi": total_prediksi,
            "jumlah_hari_tercatat": len(prediction_trend),
            "tanggal_awal": prediction_trend[0]["date"],
            "tanggal_terakhir": prediction_trend[-1]["date"],
            "tanggal_prediksi_terbanyak": peak_date["date"],
            "jumlah_prediksi_terbanyak": peak_date["count"],
            "data_tren": prediction_trend
        }
    else:
        trend_summary = {
            "total_prediksi": 0,
            "jumlah_hari_tercatat": 0,
            "data_tren": []
        }

    # =====================================
    # 7. GENERATE AI INSIGHT
    # =====================================

    ai_session_keys = {
        "distribusi": "ai_distribusi",
        "scatter": "ai_scatter",
        "prediksi": "ai_prediksi",
        "tren": "ai_tren"
    }

    ai_results = {
        key: request.session.get(session_key, "")
        for key, session_key in ai_session_keys.items()
    }

    if request.method == "POST":

        if "generate_distribusi" in request.POST:

            insight_data = {
                "total_data": total_data,
                "distribusi_kategori": category_data
            }

            ai_results["distribusi"] = generate_ai_analysis(
                insight_data,
                "Analisis Distribusi Kategori Dataset"
            )

        elif "generate_scatter" in request.POST:

            ai_results["scatter"] = generate_ai_analysis(
                scatter_analysis_data,
                "Analisis Hubungan Durasi Tidur dan Kecemasan Akademik"
            )

        elif "generate_prediksi" in request.POST:

            ai_results["prediksi"] = generate_ai_analysis(
                prediction_distribution_data,
                "Analisis Distribusi Hasil Prediksi Pengguna"
            )

        elif "generate_tren" in request.POST:

            ai_results["tren"] = generate_ai_analysis(
                trend_summary,
                "Analisis Tren Jumlah Prediksi Pengguna"
            )

        # Menyimpan setiap hasil secara terpisah
        for key, session_key in ai_session_keys.items():
            request.session[session_key] = ai_results[key]

    # =====================================
    # 8. DATA DASHBOARD
    # =====================================

    context = {
        "total_data": total_data,
        "total_kategori": len(category_counts),
        "total_kolom": len(df.columns),
        "total_fitur": len(FEATURES),

        "categories": category_data,

        "scatter_datasets": json.dumps(scatter_datasets),

        "prediction_distribution": json.dumps(
            prediction_distribution
        ),

        "prediction_trend": json.dumps(
            prediction_trend
        ),

        "total_prediksi": context_summary["total_prediksi"],
        "kategori_terbanyak": context_summary["kategori_terbanyak"],
        "prediksi_terbaru": context_summary["prediksi_terbaru"],

        "ai_results": ai_results
    }

    return render(
        request,
        "prediction/dashboard.html",
        context
    )


def prediction_page(request):
    df = pd.read_csv(DATASET_PATH)

    categorical_features = [
        "jenjang_dan_rumpun_studi",
        "status_perguruan_tinggi",
        "provinsi_domisili_kampus",
        "golongan_ukt",
        "kondisi_tempat_tinggal",
        "aktivitas_pekerjaan_sampingan",
        "akses_layanan_konseling",
    ]

    categorical_options = {}

    for feature in categorical_features:
        categorical_options[feature] = sorted(
            df[feature].dropna().astype(str).unique().tolist()
        )

    context = {
        "categorical_options": categorical_options,
    }

    if request.method == "POST":
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
            "akses_layanan_konseling",
        ]

        numeric_features = [
            "semester_studi",
            "indeks_prestasi_kumulatif_ipk",
            "beban_sks_semester_ini",
            "durasi_layar_harian_jam",
            "durasi_tidur_harian_jam",
            "pengeluaran_bulanan_total_idr",
            "skor_tekanan_finansial",
            "skor_kecemasan_akademik",
            "skor_tekanan_masa_depan_karir",
            "skor_kesepian_dan_isolasi_sosial",
            "frekuensi_keluhan_somatik_mingguan",
        ]

        try:
            input_data = {}

            for feature in features:
                value = request.POST.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(f"Kolom {feature} wajib diisi.")

                if feature in numeric_features:
                    input_data[feature] = float(value)
                else:
                    input_data[feature] = value

            input_df = pd.DataFrame([input_data], columns=features)

            model = joblib.load(MODEL_PATH)

            prediction = model.predict(input_df)[0]

            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            probability_data = []

            for category, probability in zip(classes, probabilities):
                probability_data.append({
                    "category": category,
                    "percentage": round(probability * 100, 2)
                })

            probability_data.sort(
                key=lambda item: item["percentage"],
                reverse=True
            )

            # Menyimpan hasil prediksi ke database
            PredictionHistory.objects.create(
                result=str(prediction),
                input_data=input_data
            )

            context["result"] = prediction
            context["probabilities"] = probability_data

        except (ValueError, TypeError) as e:
            context["error"] = str(e)

        except Exception as e:
            print("Prediction error:", e)
            context["error"] = (
                "Terjadi kesalahan saat memproses prediksi. "
                "Silakan periksa kembali data yang dimasukkan."
            )

    return render(
        request,
        "prediction/prediction.html",
        context
    )

# =====================================
# 4. HALAMAN RIWAYAT PREDIKSI
# =====================================

def history_page(request):

    histories = PredictionHistory.objects.all()

    context = {
        "histories": histories,
        "total_riwayat": histories.count(),
    }

    return render(
        request,
        "prediction/history.html",
        context
    )


# =====================================
# 5. HALAMAN EVALUASI MODEL
# =====================================

def evaluation_page(request):

    # Membaca dataset
    df = pd.read_csv(DATASET_PATH)

    # Memisahkan fitur dan target
    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    # Membagi data dengan pengaturan yang sama
    # seperti proses pelatihan model
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Memuat model Decision Tree
    model = joblib.load(MODEL_PATH)

    # Melakukan prediksi pada data testing
    y_pred = model.predict(X_test)

    # Menghitung Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Menghitung Classification Report
    report = classification_report(
        y_test,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    # Mengambil nilai Precision, Recall, dan F1-Score
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    # Menghitung Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=LABELS
    )

    # Menyiapkan data evaluasi setiap kategori
    class_metrics = []

    for label in LABELS:
        class_metrics.append({
            "name": label,
            "precision": round(
                report[label]["precision"] * 100, 2
            ),
            "recall": round(
                report[label]["recall"] * 100, 2
            ),
            "f1_score": round(
                report[label]["f1-score"] * 100, 2
            ),
            "support": report[label]["support"]
        })

    # Menyiapkan data Confusion Matrix
    confusion_data = []

    for i, label in enumerate(LABELS):
        confusion_data.append({
            "actual": label,
            "values": cm[i].tolist()
        })

    # Data yang dikirim ke halaman HTML
    context = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(macro_precision * 100, 2),
        "recall": round(macro_recall * 100, 2),
        "f1_score": round(macro_f1 * 100, 2),

        "class_metrics": class_metrics,
        "labels": LABELS,
        "confusion_data": confusion_data,

        "total_testing": len(y_test),
        "total_training": len(X) - len(X_test),
        "total_data": len(df),
    }

    return render(
        request,
        "prediction/evaluation.html",
        context
    )


# =====================================
# 6. HALAMAN TENTANG
# =====================================

def about_page(request):

    return render(
        request,
        "prediction/about.html"
    )




    