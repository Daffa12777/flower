# Week 3 ML — Model Deployment 🌸

Deployment model klasifikasi gambar bunga (MobileNetV2 dari Week 2) ke aplikasi
web sederhana menggunakan **Streamlit**.

## Alur (Arsitektur)
```
User  ->  Interface (Streamlit)  ->  Model (.keras)  ->  Output (kelas + probabilitas)
upload      resize + preprocess       predict            daisy/dandelion/roses/...
gambar                                                    + confidence
```

## Cara Menjalankan
1. Letakkan file model `mobilenetv2_flowers.keras` (hasil Week 2) di folder ini.
2. Install dependency:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
4. Browser terbuka otomatis. Upload gambar bunga, lihat hasil prediksinya.

## Struktur
- `app.py`            : aplikasi Streamlit (interface + inferensi model)
- `requirements.txt`  : daftar dependency
- `sample_images/`    : contoh gambar untuk pengujian
- `mobilenetv2_flowers.keras` : model terlatih (letakkan di sini)

## Kelas
daisy, dandelion, roses, sunflowers, tulips
