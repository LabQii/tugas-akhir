# OPTIMALISASI KLASTER UMKM DI KECAMATAN SAMPANG DENGAN METODE FUZZY K-MEDOIDS TYPE-2

## Deskripsi Proyek

Proyek ini bertujuan untuk mengoptimalkan klasterisasi UMKM (Usaha Mikro Kecil dan Menengah) di Kecamatan Sampang dengan menggunakan metode **Fuzzy K-Medoids Type-2**. Proses klasterisasi ini membantu untuk menganalisis dan mengategorikan UMKM berdasarkan berbagai faktor data yang dapat digunakan untuk pengambilan keputusan yang lebih baik dalam pengembangan ekonomi lokal.

![Homepage](https://github.com/LabQii/tugas-akhir/blob/main/homepage.png)

## Fitur

- **Beranda**: Halaman utama aplikasi.
- **Upload File**: Pengguna dapat mengunggah file data lokal (.csv, .xlsx, dll) atau memasukkan tautan Google Drive / Spreadsheet.
- **Normalisasi Data**: Halaman untuk melakukan normalisasi data sebelum menerapkan algoritma klasterisasi.
- **K-Medoids**: Metode klasterisasi K-Medoids tradisional untuk mengkategorikan data UMKM.
- **Fuzzy K-Medoids Type-2**: Versi lebih lanjut dari K-Medoids yang menggunakan logika fuzzy untuk menangani ketidakpastian data.
- **Hasil Analisa**: Menampilkan hasil analisis dan visualisasi dari proses klasterisasi.

## Penggunaan

1. **Upload File**: Anda dapat mengunggah dataset lokal Anda dalam format .csv, .xlsx atau tautan ke file Google Drive / Spreadsheet publik.
2. **Normalisasi Data**: Setelah file diunggah, normalisasi data akan diterapkan untuk menstandarisasi nilai-nilai data.
3. **Klasterisasi**: Terapkan metode **K-Medoids** atau **Fuzzy K-Medoids Type-2** untuk mengklasterisasi data UMKM.
4. **Lihat Hasil**: Setelah klasterisasi selesai, bagian **Hasil Analisa** akan menampilkan hasil analisis dan membantu pengguna dalam menginterpretasikan klaster yang dihasilkan.

## Stack Teknologi

- **Python** (Flask/Django atau Streamlit untuk backend)
- **Pandas, NumPy** (untuk pemrosesan data)
- **Implementasi algoritma Fuzzy K-Medoids**
- **HTML, CSS, dan Bootstrap** untuk frontend
- **JavaScript** untuk elemen dinamis (jika diperlukan)

## Instalasi

Untuk menjalankan proyek ini secara lokal, Anda memerlukan Python terinstal. Ikuti langkah-langkah berikut:

1. Clone/Download repositori:
   ```bash
   git clone https://github.com/LabQii/tugas-akhir.git
2. Buka Tools (Visual Studio Code):
   ```bash
   Buka Visual Studio Code (Visual Studio Code) atau editor pilihan Anda, dan buka folder repositori yang telah di-clone.
3. Instal library di terminal Visual Studio Code:
   ```bash
   pip install streamlit pandas numpy requests seaborn matplotlib scikit-learn pyclustering streamlit-option-menu openpyxl 
4. Jalanakan aplikasi:
   ```bash
   streamlit run C:\Users\acer\Downloads\tugas-akhir-main\insyaallah.py
5. Akses aplikasi di server lokal:
   ```bash
   http://127.0.0.1:8000
