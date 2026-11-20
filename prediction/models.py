from django.db import models


class PredictionHistory(models.Model):

    # Menyimpan kategori hasil prediksi
    result = models.CharField(max_length=50)

    # Menyimpan data input yang digunakan saat prediksi
    input_data = models.JSONField(default=dict)

    # Menyimpan tanggal dan waktu prediksi dibuat
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Riwayat Prediksi"
        verbose_name_plural = "Riwayat Prediksi"

    def __str__(self):
        return f"{self.result} - {self.created_at}"
