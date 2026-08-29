"""
preprocessing.py
=================
Tahap PREPROCESSING (Data Cleansing + Data Transformation) yang dipisah dari
notebook `terakhir_.ipynb`.

Mencakup:
    - Data Cleansing
        1. Case folding nama
        2. Penghapusan data duplikat
    - Data Transformation
        1. Rename kolom
        2. Mapping label gaya belajar (FSLSM)
        3. Standarisasi jurusan sekolah
        4. Standarisasi kabupaten
        5. Label encoding (jenis kelamin, jurusan, kabupaten)
        6. Ordinal encoding pendapatan
        7. Ordinal encoding durasi belajar
        8. Encoding IPK
        9. Encoding media pembelajaran (multi-label binarization)
        10. Normalisasi (StandardScaler)

Cara pakai di notebook:

    from preprocessing import run_preprocessing
    hasil = run_preprocessing('dataset.xlsx')

    df           = hasil['df']
    X            = hasil['X']
    X_scaled     = hasil['X_scaled']
    scaler       = hasil['scaler']
    TARGET_COLS  = hasil['TARGET_COLS']
    ... dst (lihat dict yang dikembalikan run_preprocessing)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from IPython.display import display
except ImportError:  # dijalankan di luar Jupyter/Colab
    def display(obj):
        print(obj)


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------

def bandingkan(before, after, col_before='Sebelum', col_after='Sesudah', n=10):
    """Menampilkan perbandingan sebelum/sesudah suatu tahap preprocessing
    dalam bentuk dataframe dua kolom berdampingan (bukan print teks biasa),
    mengikuti format tabel perbandingan pada contoh gambar."""
    df_banding = pd.DataFrame({
        col_before: pd.Series(before).reset_index(drop=True),
        col_after: pd.Series(after).reset_index(drop=True),
    })
    display(df_banding.head(n))
    return df_banding


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_data(filename='dataset.xlsx', verbose=True):
    df_raw = pd.read_excel(filename)
    if verbose:
        print("✅ Dataset berhasil dimuat!")
        print(f"   Jumlah baris  : {df_raw.shape[0]}")
        print(f"   Jumlah kolom  : {df_raw.shape[1]}")
        print("\nPreview 5 baris pertama:")
        display(df_raw.head(10))
    return df_raw


# --------------------------------------------------------------------------
# DATA CLEANSING
# --------------------------------------------------------------------------

def case_folding(df_raw, verbose=True):
    """TAHAP 1: Case Folding nama."""
    if verbose:
        print("=" * 65)
        print("  DATA CLEANSING — TAHAP 1: CASE FOLDING")
        print("=" * 65)

    df_raw['nama_normal'] = (
        df_raw['Nama Lengkap']
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if verbose:
        print("\nTabel Perbandingan Hasil Case Folding:")
        bandingkan(
            df_raw['Nama Lengkap'], df_raw['nama_normal'],
            col_before='nama_asli', col_after='case_folding', n=10
        )
        print(f"\n✅ Case folding selesai. {len(df_raw)} nama diproses.")

    return df_raw


def remove_duplicates(df_raw, verbose=True):
    """TAHAP 2: Penghapusan data duplikat (berdasarkan nama yang sudah
    di-case-fold), mempertahankan entri dengan timestamp terbaru."""
    if verbose:
        print("=" * 65)
        print("  DATA CLEANSING — TAHAP 2: PENGHAPUSAN DATA DUPLIKAT")
        print("=" * 65)

    target_cols_raw = [
        'Berdasarkan Cara Memproses Informasi:',
        'Berdasarkan Cara Menerima Informasi:',
        'Berdasarkan Cara Menyerap Informasi:',
        'Berdasarkan Cara Memahami Informasi:',
    ]

    duplikat_mask = df_raw.duplicated(subset='nama_normal', keep=False)
    df_duplikat = df_raw[duplikat_mask].copy()
    nama_duplikat = df_duplikat['nama_normal'].unique()
    jumlah_duplikat = len(nama_duplikat)

    if verbose:
        print(f"\nSEBELUM penghapusan: {len(df_raw)} responden, "
              f"{jumlah_duplikat} nama muncul lebih dari sekali.")
        if jumlah_duplikat > 0:
            baris_dup = []
            for nama in nama_duplikat:
                sub = df_raw[df_raw['nama_normal'] == nama]
                status = "Berbeda" if len(sub[target_cols_raw].drop_duplicates()) > 1 else "Sama"
                baris_dup.append({'nama_responden': nama.title(),
                                  'jumlah_entri': len(sub),
                                  'status_jawaban': status})
            display(pd.DataFrame(baris_dup))

    # Proses hapus duplikat (pertahankan entri timestamp terbaru)
    if 'Timestamp' in df_raw.columns:
        df_raw = df_raw.sort_values('Timestamp', ascending=False)
    df_clean = df_raw.drop_duplicates(subset='nama_normal', keep='first').copy()
    df_clean = df_clean.drop(columns=['nama_normal'])
    df_raw = df_raw.drop(columns=['nama_normal'])

    if verbose:
        print("\nSESUDAH penghapusan:")
        display(pd.DataFrame({
            'keterangan': ['Total data sebelum penghapusan duplikat',
                          'Nama yang muncul lebih dari satu kali',
                          'Total entri yang dihapus',
                          'Total data sesudah penghapusan duplikat'],
            'jumlah_data': [len(df_raw), jumlah_duplikat,
                            len(df_raw) - len(df_clean), len(df_clean)],
        }))
        print(f"\n✅ Data bersih: {len(df_clean)} responden")

    return df_clean


# --------------------------------------------------------------------------
# DATA TRANSFORMATION
# --------------------------------------------------------------------------

def rename_columns(df_clean, verbose=True):
    """TAHAP 1: Rename kolom."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 1: RENAME KOLOM")
        print("=" * 65)

    df = df_clean.copy()

    rename_map = {
        'Timestamp': 'timestamp',
        'Email Address': 'email',
        'Program Studi': 'prodi',
        'Nomor Stambuk': 'stambuk',
        'Nama Lengkap': 'nama',
        'Jenis Kelamin': 'jenis_kelamin',
        'Tempat Lahir': 'tempat_lahir',
        'Tanggal Lahir': 'tanggal_lahir',
        'Umur': 'umur',
        'Suku': 'suku',
        'Kabupaten': 'kabupaten',
        'Provinsi': 'provinsi',
        'Asal Sekolah (SMA Sederajat)': 'asal_sekolah',
        'Jurusan Sekolah': 'jurusan_sekolah',
        'Indeks Prestasi Kumulatif (IPK)': 'ipk',
        'Bagaimana Anda menggambarkan kondisi ekonomi keluarga Anda ?': 'kondisi_ekonomi',
        'Pendidikan terakhir Ayah:': 'pendidikan_ayah',
        'Pendidikan terakhir Ibu:': 'pendidikan_ibu',
        'Pekerjaan Ayah:': 'pekerjaan_ayah',
        'Pekerjaan Ibu:': 'pekerjaan_ibu',
        'Pendapatan Ayah:': 'pendapatan_ayah',
        'Pendapatan Ibu:': 'pendapatan_ibu',
        'Topik atau mata pelajaran yang paling diminati:': 'topik_minat',
        'Media pembelajaran yang paling disukai:\n\n(pilihan boleh lebih dari satu atau maksimal dua pilihan yang paling disukai)': 'media_belajar',
        'Berdasarkan Cara Memproses Informasi:': 'target_memproses',
        'Berdasarkan Cara Menerima Informasi:': 'target_menerima',
        'Berdasarkan Cara Menyerap Informasi:': 'target_menyerap',
        'Berdasarkan Cara Memahami Informasi:': 'target_memahami',
        'Durasi sesi belajar yang Anda anggap ideal? (dalam sehari)': 'durasi_belajar',
        'Apa yang memotivasi Anda untuk terus belajar dan menyelesaikan pembelajaran / pelatihan / kursus?': 'motivasi',
        "Jika pada pertanyaan sebelumnya anda menjawab 'lainnya', jelaskan jawaban anda disini:": 'motivasi_lainnya',
    }
    df = df.rename(columns=rename_map).drop(
        columns=['Column 31', 'Column 32'], errors='ignore')

    if verbose:
        pasangan = [
            ('Jenis Kelamin', 'jenis_kelamin'),
            ('Pendapatan Ayah:', 'pendapatan_ayah'),
            ('Pendapatan Ibu:', 'pendapatan_ibu'),
            ('Kabupaten', 'kabupaten'),
            ('Jurusan Sekolah', 'jurusan_sekolah'),
            ('Indeks Prestasi Kumulatif (IPK)', 'ipk'),
            ('Media pembelajaran yang paling disukai: ...', 'media_belajar'),
            ('Durasi sesi belajar yang Anda anggap ideal? ...', 'durasi_belajar'),
            ('Berdasarkan Cara Memproses Informasi:', 'target_memproses'),
            ('Berdasarkan Cara Menerima Informasi:', 'target_menerima'),
            ('Berdasarkan Cara Menyerap Informasi:', 'target_menyerap'),
            ('Berdasarkan Cara Memahami Informasi:', 'target_memahami'),
        ]
        print("\nTabel Perbandingan Nama Kolom (kolom yang dipakai):")
        display(pd.DataFrame(pasangan, columns=['nama_kolom_asli', 'nama_kolom_baru']))
        print("\n✅ Rename kolom selesai.")

    return df


def map_learning_labels(df, verbose=True):
    """TAHAP 2: Mapping label gaya belajar (FSLSM)."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 2: MAPPING LABEL GAYA BELAJAR")
        print("=" * 65)

    label_map = {
        'Saya belajar lebih baik dengan berdiskusi dan beraktivitas secara langsung (Active Learner)': 'Active',
        'Saya belajar lebih baik dengan berpikir sendiri dan merenungkan materi (Reflective Learner)': 'Reflective',
        'Saya lebih suka fakta, prosedur, dan aplikasi nyata (Sensing Learner)': 'Sensing',
        'Saya lebih tertarik pada konsep, teori, dan kemungkinan inovatif (Intuitive Learner)': 'Intuitive',
        'Saya lebih mudah memahami materi dengan gambar, diagram, atau visualisasi (Visual Learner)': 'Visual',
        'Saya lebih mudah memahami materi lewat penjelasan kata-kata, baik lisan maupun tulisan (Verbal Learner)': 'Verbal',
        'Saya lebih suka belajar langkah demi langkah secara sistematis (Sequential Learner)': 'Sequential',
        'Saya lebih suka memahami materi secara keseluruhan dan kemudian menyusun koneksi (Global Learner)': 'Global',
    }

    target_cols = ['target_memproses', 'target_menerima',
                   'target_menyerap', 'target_memahami']

    if verbose:
        print("\nSEBELUM mapping (contoh jawaban asli, 1 baris pertama):")
        display(pd.DataFrame({
            'kolom_target': target_cols,
            'jawaban_asli': [str(df[c].iloc[0])[:70] + '...' for c in target_cols],
        }))

    for col in target_cols:
        df[col] = df[col].map(label_map).fillna(df[col])

    if verbose:
        print("\nSESUDAH mapping (pemetaan jawaban → label):")
        display(pd.DataFrame({
            'jawaban_asli': [k[:70] + '...' for k in label_map.keys()],
            'label': list(label_map.values()),
        }))

        print("\nDistribusi label per dimensi:")
        baris_dist = []
        for col in target_cols:
            vc = df[col].value_counts()
            for label, jml in vc.items():
                baris_dist.append({'dimensi': col.replace('target_', '').capitalize(),
                                   'label': label, 'jumlah': int(jml)})
        display(pd.DataFrame(baris_dist))
        print("\n✅ Mapping label selesai.")

    return df


def standardize_jurusan(df, verbose=True):
    """TAHAP 3: Standarisasi jurusan sekolah."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 3: STANDARISASI JURUSAN SEKOLAH")
        print("=" * 65)
        print(f"\nSEBELUM standarisasi ({df['jurusan_sekolah'].nunique()} variasi unik):")
        display(df['jurusan_sekolah'].value_counts()
                .rename_axis('penulisan_asli').reset_index(name='jumlah'))

    def standarisasi_jurusan(val):
        val = str(val).strip().lower()
        if val in ['.', 'tidak ada', 'nan']:
            return 'Lainnya'
        if 'kurikulum merdeka' in val or 'kelas umum' in val:
            return 'Kurikulum Merdeka'
        if 'ipa' in val:
            return 'IPA'
        if 'ips' in val:
            return 'IPS'
        if 'tkj' in val:
            return 'TKJ'
        if 'rpl' in val or 'rekayasa perangkat lunak' in val:
            return 'RPL'
        if 'multimedia' in val:
            return 'Multimedia'
        if 'agama' in val or 'keagamaan' in val or 'tahfidz' in val:
            return 'Keagamaan'
        if 'bahasa' in val:
            return 'Bahasa'
        if 'akuntansi' in val:
            return 'Akuntansi'
        return 'Lainnya'

    df['jurusan_std'] = df['jurusan_sekolah'].apply(standarisasi_jurusan)

    if verbose:
        print("\nTabel Perbandingan Hasil Standarisasi Jurusan Sekolah:")
        bandingkan(
            df['jurusan_sekolah'], df['jurusan_std'],
            col_before='jurusan_sekolah', col_after='jurusan_std', n=10
        )

        total = len(df)
        print(f"\nSESUDAH standarisasi ({df['jurusan_std'].nunique()} kategori baku):")
        tabel_jurusan_sesudah = (df['jurusan_std'].value_counts()
                                 .rename_axis('kategori_standar').reset_index(name='jumlah'))
        tabel_jurusan_sesudah['persentase'] = (
            tabel_jurusan_sesudah['jumlah'] / total * 100).round(1)
        display(tabel_jurusan_sesudah)

        print(f"\n  Reduksi variasi: {df['jurusan_sekolah'].nunique()} → {df['jurusan_std'].nunique()} kategori")
        print("\n✅ Standarisasi jurusan selesai.")

    return df


def standardize_kabupaten(df, verbose=True):
    """TAHAP 4: Standarisasi kabupaten."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 4: STANDARISASI KABUPATEN")
        print("=" * 65)
        print(f"\nSEBELUM standarisasi ({df['kabupaten'].nunique()} variasi unik, 20 teratas):")
        display(df['kabupaten'].value_counts().head(20)
                .rename_axis('penulisan_asli').reset_index(name='jumlah'))

    def standarisasi_kabupaten(val):
        val = str(val).strip().lower()
        for prefix in ['kota ', 'kabupaten ', 'kab. ', 'kab ']:
            if val.startswith(prefix):
                val = val[len(prefix):]
        val = val.strip().title()
        koreksi = {
            'Makassar': 'Makassar',
            'Maros': 'Maros',
            'Gowa': 'Gowa',
            'Bone': 'Bone',
            'Luwu Timur': 'Luwu Timur',
            'Luwu': 'Luwu',
            'Wajo': 'Wajo',
            'Pangkep': 'Pangkep',
            'Pangkajene Dan Kepulauan': 'Pangkep',  # Added to standardize with 'Pangkep'
            'Enrekang': 'Enrekang',
            'Sinjai': 'Sinjai',
            'Soppeng': 'Soppeng',
            'Pinrang': 'Pinrang',
        }
        return koreksi.get(val, val)

    df['kabupaten_std'] = df['kabupaten'].apply(standarisasi_kabupaten)

    # Kelompokkan kabupaten dengan jumlah < 3 ke "Lainnya"
    frekuensi_kab = df['kabupaten_std'].value_counts()
    kab_jarang = frekuensi_kab[frekuensi_kab < 3].index.tolist()
    df['kabupaten_std'] = df['kabupaten_std'].apply(
        lambda x: 'Lainnya' if x in kab_jarang else x)

    if verbose:
        total = len(df)
        print("\nTabel Perbandingan Hasil Standarisasi Kabupaten:")
        bandingkan(
            df['kabupaten'], df['kabupaten_std'],
            col_before='kabupaten', col_after='kabupaten_std', n=10
        )

        print(f"\nSESUDAH standarisasi ({df['kabupaten_std'].nunique()} kategori):")
        tabel_kab_sesudah = (df['kabupaten_std'].value_counts()
                             .rename_axis('kategori').reset_index(name='jumlah'))
        tabel_kab_sesudah['persentase'] = (
            tabel_kab_sesudah['jumlah'] / total * 100).round(1)
        display(tabel_kab_sesudah)

        print(f"\n  Reduksi variasi: {df['kabupaten'].nunique()} → {df['kabupaten_std'].nunique()} kategori")
        print("\n✅ Standarisasi kabupaten selesai.")

    return df


def label_encoding(df, verbose=True):
    """TAHAP 5: Label encoding (jenis kelamin, jurusan, kabupaten)."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 5: LABEL ENCODING")
        print("=" * 65)
        print("\nSEBELUM encoding (5 baris pertama):")
        display(df[['jenis_kelamin', 'jurusan_std', 'kabupaten_std']].head(5))

    le_gender = LabelEncoder()
    le_jurusan = LabelEncoder()
    le_kabupaten = LabelEncoder()

    df['jenis_kelamin_enc'] = le_gender.fit_transform(df['jenis_kelamin'])
    df['jurusan_sekolah_enc'] = le_jurusan.fit_transform(df['jurusan_std'])
    df['kabupaten_enc'] = le_kabupaten.fit_transform(df['kabupaten_std'])

    if verbose:
        def tabel_encoding(le, kolom_asli):
            return pd.DataFrame({
                'kategori': le.classes_,
                'nilai_encoded': le.transform(le.classes_),
                'jumlah': [int((df[kolom_asli] == c).sum()) for c in le.classes_],
            })

        print("\nSESUDAH encoding — Jenis Kelamin:")
        display(tabel_encoding(le_gender, 'jenis_kelamin'))

        print("\nSESUDAH encoding — Jurusan Sekolah:")
        display(tabel_encoding(le_jurusan, 'jurusan_std'))

        print(f"\nSESUDAH encoding — Kabupaten ({len(le_kabupaten.classes_)} kategori):")
        display(tabel_encoding(le_kabupaten, 'kabupaten_std'))

        print("\n5 baris pertama sesudah encoding:")
        display(df[['jenis_kelamin_enc', 'jurusan_sekolah_enc', 'kabupaten_enc']].head(5))
        print("\n✅ Label encoding selesai.")

    return df, le_gender, le_jurusan, le_kabupaten


def encode_pendapatan(df, verbose=True):
    """TAHAP 6: Ordinal encoding pendapatan (Ayah & Ibu)."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 6: ORDINAL ENCODING PENDAPATAN")
        print("=" * 65)

    ordinal_pendapatan = {
        'Kurang dari Rp 2.000.000 per bulan': 1,
        'Rp 2.000.000 - Rp 5.000.000 per bulan': 2,
        'Rp 5.000.000 - Rp 8.000.000 per bulan': 3,
        'Lebih dari Rp. 8.000.000 per bulan': 4,
    }
    urutan_kategori = list(ordinal_pendapatan.keys())

    if verbose:
        print("\nSEBELUM encoding:")
        display(pd.DataFrame({
            'kategori_pendapatan': urutan_kategori,
            'pendapatan_ayah': [int((df['pendapatan_ayah'] == k).sum()) for k in urutan_kategori],
            'pendapatan_ibu': [int((df['pendapatan_ibu'] == k).sum()) for k in urutan_kategori],
        }))

    df['pendapatan_ayah_enc'] = df['pendapatan_ayah'].map(ordinal_pendapatan)
    df['pendapatan_ibu_enc'] = df['pendapatan_ibu'].map(ordinal_pendapatan)
    null_ayah = df['pendapatan_ayah_enc'].isna().sum()
    null_ibu = df['pendapatan_ibu_enc'].isna().sum()
    df['pendapatan_ayah_enc'] = df['pendapatan_ayah_enc'].fillna(df['pendapatan_ayah_enc'].median())
    df['pendapatan_ibu_enc'] = df['pendapatan_ibu_enc'].fillna(df['pendapatan_ibu_enc'].median())

    if verbose:
        print("\nSESUDAH encoding (pemetaan ordinal dan distribusi):")
        display(pd.DataFrame({
            'kategori_pendapatan': urutan_kategori,
            'nilai_encoded': list(ordinal_pendapatan.values()),
            'jumlah_ayah': [int((df['pendapatan_ayah_enc'] == v).sum())
                            for v in ordinal_pendapatan.values()],
            'jumlah_ibu': [int((df['pendapatan_ibu_enc'] == v).sum())
                           for v in ordinal_pendapatan.values()],
        }))

        if null_ayah > 0 or null_ibu > 0:
            print(f"\n  ⚠️  Nilai kosong: Ayah={null_ayah}, Ibu={null_ibu} → diisi median")
        print("\n✅ Ordinal encoding pendapatan selesai.")

    return df


def encode_durasi(df, verbose=True):
    """TAHAP 7: Ordinal encoding durasi belajar."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 7: ORDINAL ENCODING DURASI BELAJAR")
        print("=" * 65)

    ordinal_durasi = {
        'Kurang dari 30 menit': 1,
        '30 menit - 1 jam': 2,
        '1 - 2 jam': 3,
        'Lebih dari 2 jam': 4,
    }

    if verbose:
        print("\nSEBELUM encoding:")
        display(df['durasi_belajar'].value_counts()
                .rename_axis('kategori_durasi').reset_index(name='jumlah'))

    df['durasi_belajar_enc'] = df['durasi_belajar'].map(ordinal_durasi)
    df['durasi_belajar_enc'] = df['durasi_belajar_enc'].fillna(df['durasi_belajar_enc'].median())

    if verbose:
        print("\nSESUDAH encoding:")
        display(pd.DataFrame({
            'kategori_durasi': list(ordinal_durasi.keys()),
            'nilai_encoded': list(ordinal_durasi.values()),
            'jumlah': [int((df['durasi_belajar'] == k).sum()) for k in ordinal_durasi.keys()],
        }))
        print("\n✅ Ordinal encoding durasi belajar selesai.")

    return df


def encode_ipk(df, verbose=True):
    """TAHAP 8: Encoding / cleaning IPK."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 8: ENCODING IPK")
        print("=" * 65)

    def bersihkan_ipk(val):
        val = str(val).strip().replace(',', '.')
        try:
            v = float(val)
            return v if 0.0 <= v <= 4.0 else np.nan
        except Exception:
            return np.nan

    if verbose:
        print("\nSEBELUM cleaning (10 nilai unik terbanyak):")
        display(df['ipk'].value_counts().head(10)
                .rename_axis('nilai_ipk_asli').reset_index(name='jumlah'))

        contoh_invalid = [v for v in df['ipk'].unique()
                          if not str(v).replace('.', '').replace(',', '').isdigit()][:5]
        print("\nContoh nilai tidak valid:")
        display(pd.DataFrame({'contoh_nilai_tidak_valid': contoh_invalid}))

    df['ipk_enc'] = df['ipk'].apply(bersihkan_ipk)
    null_ipk = df['ipk_enc'].isna().sum()
    median_ipk_val = df['ipk_enc'].median()
    df['ipk_enc'] = df['ipk_enc'].fillna(median_ipk_val)

    if verbose:
        print("\nTabel Perbandingan Hasil Cleaning IPK:")
        bandingkan(
            df['ipk'], df['ipk_enc'],
            col_before='ipk', col_after='ipk_enc', n=10
        )

        print("\nSESUDAH cleaning:")
        display(pd.DataFrame({
            'keterangan': ['Nilai tidak valid/kosong (diisi median)',
                           'Median IPK', 'IPK minimum', 'IPK maksimum', 'Rata-rata IPK'],
            'nilai': [int(null_ipk), round(median_ipk_val, 2),
                     round(df['ipk_enc'].min(), 2), round(df['ipk_enc'].max(), 2),
                     round(df['ipk_enc'].mean(), 2)],
        }))

    bins = [0, 2.75, 3.0, 3.5, 4.01]
    labels = ['< 2.75', '2.75–3.00', '3.00–3.50', '3.50–4.00']
    df['ipk_kategori'] = pd.cut(df['ipk_enc'], bins=bins, labels=labels, right=False)

    if verbose:
        print("\nDistribusi IPK per kategori:")
        display(df['ipk_kategori'].value_counts().sort_index()
                .rename_axis('kategori_ipk').reset_index(name='jumlah'))
        print("\n✅ Encoding IPK selesai.")

    return df


def encode_media(df, verbose=True):
    """TAHAP 9: Encoding media pembelajaran (multi-label binarization)."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 9: ENCODING MEDIA PEMBELAJARAN")
        print("  (Multi-label Binarization)")
        print("=" * 65)

    media_map = {
        'Buku teks atau modul cetak': 'media_buku',
        'Slide presentasi (PowerPoint) dari dosen': 'media_slide',
        'Video pembelajaran (YouTube, e-learning, dll.)': 'media_video',
        'Media interaktif (simulasi, animasi, aplikasi edukasi)': 'media_interaktif',
        'Website atau platform pembelajaran daring (Coursera, Udemy, Dicoding, dll.)': 'media_platform',
        'Forum diskusi online atau grup belajar (WhatsApp, Telegram, LMS, dll.)': 'media_forum',
        'Coding platform atau IDE (misalnya: repl.it, Jupyter, VS Code)': 'media_coding',
    }
    media_cols = list(media_map.values())

    if verbose:
        print("\nSEBELUM encoding (5 contoh jawaban asli terbanyak):")
        display(pd.DataFrame({
            'contoh_jawaban_asli': [v[:70] for v in
                                    df['media_belajar'].value_counts().head(5).index],
        }))

    for col in media_cols:
        df[col] = 0
    for idx, row in df.iterrows():
        pilihan = str(row['media_belajar'])
        for nama_media, nama_kolom in media_map.items():
            if nama_media in pilihan:
                df.at[idx, nama_kolom] = 1

    if verbose:
        print("\nSESUDAH encoding (setiap media jadi kolom biner 0/1):")
        display(pd.DataFrame({
            'kolom_biner': media_cols,
            'media': [k[:55] for k in media_map.keys()],
            'jumlah_pemilih': [int(df[c].sum()) for c in media_cols],
        }))

        print("\nContoh 3 baris sesudah encoding:")
        display(df[media_cols].head(3))
        print(f"\n✅ Encoding media pembelajaran selesai. {len(media_cols)} kolom biner ditambahkan.")

    return df, media_cols


def normalize_features(df, media_cols, verbose=True):
    """TAHAP 10: Normalisasi (StandardScaler) + definisi fitur/target akhir."""
    if verbose:
        print("=" * 65)
        print("  DATA TRANSFORMATION — TAHAP 10: NORMALISASI (StandardScaler)")
        print("=" * 65)

    FEATURE_COLS = [
        'jenis_kelamin_enc',    # 1. Jenis Kelamin
        'pendapatan_ayah_enc',  # 2. Pendapatan Ayah
        'pendapatan_ibu_enc',   # 3. Pendapatan Ibu
        'kabupaten_enc',        # 4. Kabupaten
        'jurusan_sekolah_enc',  # 5. Jurusan Sekolah
        'ipk_enc',              # 6. IPK
        'durasi_belajar_enc',   # 7. Durasi Belajar
    ] + media_cols              # 8. Media Pembelajaran (7 kolom biner)

    feat_labels = [
        'Jenis Kelamin', 'Pendapatan Ayah', 'Pendapatan Ibu',
        'Kabupaten', 'Jurusan Sekolah', 'IPK', 'Durasi Belajar',
        'Media: Buku', 'Media: Slide', 'Media: Video',
        'Media: Interaktif', 'Media: Platform', 'Media: Forum', 'Media: Coding',
    ]

    X = df[FEATURE_COLS].values
    X_raw = X.copy()

    def tabel_statistik(matriks, label_fitur):
        return pd.DataFrame({
            'fitur': label_fitur,
            'min': matriks.min(axis=0).round(4),
            'max': matriks.max(axis=0).round(4),
            'mean': matriks.mean(axis=0).round(4),
            'std': matriks.std(axis=0).round(4),
        })

    if verbose:
        print("\nSEBELUM normalisasi (5 baris pertama, 7 fitur utama):")
        display(pd.DataFrame(X_raw[:5, :7], columns=feat_labels[:7]))
        print("\nStatistik SEBELUM normalisasi:")
        display(tabel_statistik(X_raw[:, :7], feat_labels[:7]))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if verbose:
        print("\nSESUDAH normalisasi (5 baris pertama, 7 fitur utama):")
        display(pd.DataFrame(X_scaled[:5, :7], columns=feat_labels[:7]).round(4))
        print("\nStatistik SESUDAH normalisasi (mean≈0, std≈1):")
        display(tabel_statistik(X_scaled[:, :7], feat_labels[:7]))

    TARGET_COLS = {
        'Memproses (Active vs Reflective)': 'target_memproses',
        'Menerima (Sensing vs Intuitive)': 'target_menerima',
        'Menyerap (Visual vs Verbal)': 'target_menyerap',
        'Memahami (Sequential vs Global)': 'target_memahami',
    }

    # English feature names & dimension names (untuk confusion-matrix dan
    # decision-tree plots, diterjemahkan ke bahasa Inggris)
    feat_labels_en = [
        'Gender', "Father's Income", "Mother's Income",
        'Regency', 'School Major', 'GPA', 'Study Duration',
        'Media: Book', 'Media: Slide', 'Media: Video',
        'Media: Interactive', 'Media: Platform', 'Media: Forum', 'Media: Coding',
    ]

    DIM_EN = {
        'Memproses (Active vs Reflective)': 'Processing (Active vs Reflective)',
        'Menerima (Sensing vs Intuitive)': 'Perception (Sensing vs Intuitive)',
        'Menyerap (Visual vs Verbal)': 'Input (Visual vs Verbal)',
        'Memahami (Sequential vs Global)': 'Understanding (Sequential vs Global)',
    }

    if verbose:
        print("\nRingkasan akhir preprocessing:")
        print(f"  Total responden : {len(df)}")
        print(f"  Total fitur     : {len(FEATURE_COLS)} (7 fitur + 7 kolom media)")
        baris_target = []
        for nama, col in TARGET_COLS.items():
            for label, jml in df[col].value_counts().items():
                baris_target.append({'dimensi': nama, 'kelas': label, 'jumlah': int(jml)})
        display(pd.DataFrame(baris_target))
        print("\n✅ Normalisasi selesai. Preprocessing lengkap.")

    return {
        'X': X,
        'X_raw': X_raw,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'FEATURE_COLS': FEATURE_COLS,
        'feat_labels': feat_labels,
        'feat_labels_en': feat_labels_en,
        'TARGET_COLS': TARGET_COLS,
        'DIM_EN': DIM_EN,
    }


# --------------------------------------------------------------------------
# Orkestrasi seluruh tahap preprocessing
# --------------------------------------------------------------------------

def run_preprocessing(filename='dataset.xlsx', verbose=True):
    """Menjalankan seluruh tahap preprocessing (data cleansing + data
    transformation) secara berurutan dan mengembalikan semua artefak yang
    dibutuhkan oleh tahap pemodelan (split data, SMOTE, klasifikasi, dll).
    """
    # --- Data Cleansing ---
    df_raw = load_data(filename, verbose=verbose)
    df_raw = case_folding(df_raw, verbose=verbose)
    df_clean = remove_duplicates(df_raw, verbose=verbose)

    # --- Data Transformation ---
    df = rename_columns(df_clean, verbose=verbose)
    df = map_learning_labels(df, verbose=verbose)
    df = standardize_jurusan(df, verbose=verbose)
    df = standardize_kabupaten(df, verbose=verbose)
    df, le_gender, le_jurusan, le_kabupaten = label_encoding(df, verbose=verbose)
    df = encode_pendapatan(df, verbose=verbose)
    df = encode_durasi(df, verbose=verbose)
    df = encode_ipk(df, verbose=verbose)
    df, media_cols = encode_media(df, verbose=verbose)
    fitur = normalize_features(df, media_cols, verbose=verbose)

    hasil = {
        'df': df,
        'df_raw': df_raw,
        'le_gender': le_gender,
        'le_jurusan': le_jurusan,
        'le_kabupaten': le_kabupaten,
        'media_cols': media_cols,
    }
    hasil.update(fitur)
    return hasil


if __name__ == '__main__':
    run_preprocessing('dataset.xlsx', verbose=True)
