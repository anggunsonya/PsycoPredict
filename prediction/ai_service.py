
from groq import Groq
from django.conf import settings


def generate_ai_analysis(analysis_data, section_title):
    """
    Menghasilkan analisis AI untuk setiap grafik
    pada dashboard PsycoPredict.
    """

    if not settings.GROQ_API_KEY:
        return "Groq API Key belum dikonfigurasi."

    try:
        client = Groq(
            api_key=settings.GROQ_API_KEY
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            
messages=[
    {
        "role": "system",
        "content": """
Kamu adalah AI Data Analyst pada aplikasi PsycoPredict,
sebuah sistem analisis dan prediksi tingkat kecemasan akademik.

Tugasmu adalah menjelaskan isi grafik dashboard
berdasarkan data yang diberikan.

ATURAN PENJELASAN:
1. Gunakan bahasa Indonesia yang sederhana,
   natural, jelas, dan mudah dipahami orang awam.
2. Jelaskan informasi utama yang ditampilkan grafik.
3. Gunakan angka dan persentase yang tersedia
   untuk memperkuat penjelasan.
4. Jelaskan pola atau perbedaan yang terlihat pada grafik.
5. Hindari istilah teknis yang sulit dipahami.
6. Jika terpaksa menggunakan istilah teknis,
   jelaskan dengan bahasa sederhana.
7. Jangan mengarang atau menambahkan informasi
   yang tidak terdapat dalam data.
8. Jangan menyatakan hubungan sebab-akibat.
9. Jangan memberikan diagnosis atau saran medis.
10. Tegaskan bahwa hasil prediksi bukan diagnosis klinis
    jika pembahasannya berkaitan dengan hasil prediksi.

FORMAT JAWABAN:
1. Buat hanya satu paragraf utuh.
2. Panjang jawaban sekitar 3-5 kalimat.
3. Mulailah dengan menjelaskan informasi utama grafik.
4. Sertakan angka penting yang mendukung penjelasan.
5. Akhiri dengan kesimpulan sederhana tentang
   informasi yang dapat dipahami dari grafik.
6. Jangan membuat subjudul seperti Temuan Utama,
   Interpretasi, atau Kesimpulan.
7. Jangan gunakan daftar atau poin-poin.
8. Jangan gunakan Markdown, tanda #, **, atau simbol dekoratif.
9. Hindari pengulangan kalimat.
10. Pastikan penjelasan ringkas dan nyaman dibaca
    langsung pada dashboard.
"""
    },
    {
        "role": "user",
        "content": f"""
Judul grafik: {section_title}

Data grafik:
{analysis_data}

Jelaskan grafik tersebut dalam satu paragraf singkat
sesuai seluruh ketentuan yang diberikan.
Gunakan bahasa sederhana agar mudah dipahami
oleh masyarakat umum.
"""
    }
],

            temperature=0.2,
            max_completion_tokens=700
        )

        result = response.choices[0].message.content

        return result.strip() if result else (
            "Analisis AI belum menghasilkan jawaban."
        )

    except Exception as e:
        print(f"Groq API Error: {e}")

        return (
            "Maaf, analisis AI belum dapat ditampilkan. "
            "Silakan coba kembali."
        )
