# Klasifikasi Gaya Belajar Mahasiswa

Perbandingan algoritma **Decision Tree**, **Random Forest**, dan **Support Vector Machine (SVM)** untuk klasifikasi preferensi gaya belajar mahasiswa berdasarkan model **FSLSM (Felder-Silverman Learning Style Model)**, pada 4 dimensi:

- Memproses Informasi (Active vs Reflective)
- Menerima Informasi (Sensing vs Intuitive)
- Menyerap Informasi (Visual vs Verbal)
- Memahami Informasi (Sequential vs Global)

## Abstrak

Memahami gaya belajar mahasiswa dapat membantu penyesuaian metode dan media pembelajaran agar lebih efektif. Penelitian ini membangun model klasifikasi gaya belajar berdasarkan model FSLSM menggunakan data kuesioner dari mahasiswa, dengan fitur berupa data demografis (jenis kelamin, pendapatan orang tua, kabupaten asal, jurusan sekolah), akademik (IPK), dan kebiasaan belajar (durasi belajar, media pembelajaran yang disukai). Data diproses melalui tahap cleansing dan transformation, kemudian dilakukan penyeimbangan kelas dengan SMOTE pada data latih sebelum tiga algoritma klasifikasi — Decision Tree, Random Forest, dan SVM — dilatih dan diuji dengan skema split 80:20 serta divalidasi menggunakan cross-validation 5-fold. Hasil pengujian menunjukkan Random Forest secara konsisten memberikan performa terbaik di antara ketiga algoritma pada keempat dimensi FSLSM, dengan akurasi berkisar 64%–79%, diikuti oleh Decision Tree dan SVM. Hasil lengkap evaluasi (accuracy, precision, recall, F1-score) dan visualisasi confusion matrix tersedia pada folder `hasil/`.

## Struktur Folder

```
klasifikasi-gaya-belajar-mahasiswa/
│
├── notebook/
│   ├── klasifikasi_gaya_belajar.ipynb   # Notebook utama (preprocessing + pemodelan)
│   └── preprocessing.py                 # Modul tahap data cleansing & transformation
│
├── dataset/
│   └── dataset_clean.csv                # Dataset hasil preprocessing (identitas responden dihapus)
│
├── hasil/
│   ├── confusion_matrix/                # Confusion matrix per model
│   ├── decision_tree/                   # Visualisasi pohon keputusan
│   ├── grafik_lainnya/                  # Grafik performa & perbandingan model
│   └── evaluasi_model.csv               # Ringkasan akurasi, precision, recall, F1-score
│
├── requirements.txt
└── README.md
```

## Cara Menjalankan

1. Clone repository ini
   ```bash
   git clone https://github.com/zalana0/klasifikasi-gaya-belajar-mahasiswa.git
   cd klasifikasi-gaya-belajar-mahasiswa
   ```
2. Install dependency
   ```bash
   pip install -r requirements.txt
   ```
3. Buka `notebook/klasifikasi_gaya_belajar.ipynb` di Jupyter/Google Colab, lalu jalankan sel secara berurutan.

## Metodologi Singkat

- **Data cleansing**: case folding nama, penghapusan data duplikat
- **Data transformation**: rename kolom, mapping label FSLSM, standarisasi jurusan & kabupaten, label encoding, ordinal encoding pendapatan & durasi belajar, encoding IPK, multi-label binarization media pembelajaran, normalisasi (StandardScaler)
- **Split data**: 80:20 stratified
- **Balancing kelas**: SMOTE (hanya pada data latih)
- **Validasi**: Cross-validation 5-fold
- **Algoritma**: Random Forest (200 pohon), Decision Tree (CART, max_depth=5), SVM (RBF, C=10)

## Catatan Privasi

Dataset yang dipublikasikan (`dataset_clean.csv`) sudah **tidak menyertakan** nama lengkap, email, NIM, tempat lahir, dan tanggal lahir responden untuk menjaga privasi mahasiswa yang berpartisipasi.

## Kontributor

- Salzabila Triana

