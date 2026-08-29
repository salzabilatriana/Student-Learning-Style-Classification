# Klasifikasi Gaya Belajar Mahasiswa

Perbandingan algoritma **Decision Tree**, **Random Forest**, dan **Support Vector Machine (SVM)** untuk klasifikasi preferensi gaya belajar mahasiswa berdasarkan model **FSLSM (Felder-Silverman Learning Style Model)**, pada 4 dimensi:

- Memproses Informasi (Active vs Reflective)
- Menerima Informasi (Sensing vs Intuitive)
- Menyerap Informasi (Visual vs Verbal)
- Memahami Informasi (Sequential vs Global)

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
3. Buka `notebook/klasifikasi_gaya_belajar.ipynb` di Google Colab, lalu jalankan sel secara berurutan.

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

