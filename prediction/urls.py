from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("prediksi/", views.prediction_page, name="prediction"),
    path("riwayat/", views.history_page, name="history"),

    # Halaman Evaluasi Model
    path("evaluasi-model/", views.evaluation_page, name="evaluation"),

    # Halaman Tentang
    path("tentang/", views.about_page, name="about"),
]