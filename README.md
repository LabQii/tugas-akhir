<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Optimisasi Klaster UMKM di Kecamatan Sampang dengan Metode Fuzzy K-Medoids Type-2</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KyZXEJzQf5H2N6m67I1duJXyAB0gsxARVqGg9nD8fX9b6Xn5HegB85jwdFzjpt5p" crossorigin="anonymous">
    <style>
        .container {
            margin-top: 20px;
        }
        .section-title {
            margin-top: 30px;
            font-size: 1.8rem;
            font-weight: bold;
            text-align: center;
        }
        .content {
            font-size: 1rem;
            line-height: 1.6;
        }
        .code-block {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center my-5">OPTIMALISASI KLASTER UMKM DI KECAMATAN SAMPANG DENGAN METODE FUZZY K-MEDOIDS TYPE-2</h1>
        
        <div class="section-title">Deskripsi Proyek</div>
        <p class="content">
            Proyek ini bertujuan untuk mengoptimalkan klasterisasi UMKM (Usaha Mikro Kecil dan Menengah) di Kecamatan Sampang dengan menggunakan metode <strong>Fuzzy K-Medoids Type-2</strong>. Proses klasterisasi ini membantu untuk menganalisis dan mengategorikan UMKM berdasarkan berbagai faktor data yang dapat digunakan untuk pengambilan keputusan yang lebih baik dalam pengembangan ekonomi lokal.
        </p>

        <div class="section-title">Fitur</div>
        <ul class="content">
            <li><strong>Beranda</strong>: Halaman utama aplikasi.</li>
            <li><strong>Upload File</strong>: Pengguna dapat mengunggah file data lokal (.csv, .xlsx, dll) atau memasukkan tautan Google Drive / Spreadsheet.</li>
            <li><strong>Normalisasi Data</strong>: Halaman untuk melakukan normalisasi data sebelum menerapkan algoritma klasterisasi.</li>
            <li><strong>K-Medoids</strong>: Metode klasterisasi K-Medoids tradisional untuk mengkategorikan data UMKM.</li>
            <li><strong>Fuzzy K-Medoids Type-2</strong>: Versi lebih lanjut dari K-Medoids yang menggunakan logika fuzzy untuk menangani ketidakpastian data.</li>
            <li><strong>Hasil Analisa</strong>: Menampilkan hasil analisis dan visualisasi dari proses klasterisasi.</li>
        </ul>

        <div class="section-title">Penggunaan</div>
        <ol class="content">
            <li><strong>Upload File</strong>: Anda dapat mengunggah dataset lokal Anda dalam format .csv, .xlsx atau tautan ke file Google Drive / Spreadsheet publik.</li>
            <li><strong>Normalisasi Data</strong>: Setelah file diunggah, normalisasi data akan diterapkan untuk menstandarisasi nilai-nilai data.</li>
            <li><strong>Klasterisasi</strong>: Terapkan metode <strong>K-Medoids</strong> atau <strong>Fuzzy K-Medoids Type-2</strong> untuk mengklasterisasi data UMKM.</li>
            <li><strong>Lihat Hasil</strong>: Setelah klasterisasi selesai, bagian <strong>Hasil Analisa</strong> akan menampilkan hasil analisis dan membantu pengguna dalam menginterpretasikan klaster yang dihasilkan.</li>
        </ol>

        <div class="section-title">Stack Teknologi</div>
        <ul class="content">
            <li>Python (Flask/Django atau Streamlit untuk backend)</li>
            <li>Pandas, NumPy (untuk pemrosesan data)</li>
            <li>Implementasi algoritma Fuzzy K-Medoids</li>
            <li>HTML, CSS, dan Bootstrap untuk frontend</li>
            <li>JavaScript untuk elemen dinamis (jika diperlukan)</li>
        </ul>

        <div class="section-title">Instalasi</div>
        <p class="content">Untuk menjalankan proyek ini secara lokal, Anda memerlukan Python terinstal. Ikuti langkah-langkah berikut:</p>
        <ol class="content">
            <li>Clone repositori:
                <div class="code-block">
                    <code>git clone https://github.com/username/optimalisasi-klaster-umkm.git</code>
                </div>
            </li>
            <li>Pindah ke direktori proyek:
                <div class="code-block">
                    <code>cd optimalisasi-klaster-umkm</code>
                </div>
            </li>
            <li>Instal dependensi yang diperlukan:
                <div class="code-block">
                    <code>pip install -r requirements.txt</code>
                </div>
            </li>
            <li>Jalankan aplikasi:
                <div class="code-block">
                    <code>python app.py</code>
                </div>
            </li>
            <li>Akses aplikasi di server lokal:
                <div class="code-block">
                    <code>http://127.0.0.1:5000</code>
                </div>
            </li>
        </ol>

        <div class="section-title">Kontribusi</div>
        <p class="content">
            Silakan fork repositori ini, lakukan perbaikan, dan kirimkan pull request. Kontribusi selalu diterima!
        </p>

        <div class="section-title">Lisensi</div>
        <p class="content">
            Proyek ini dilisensikan di bawah Lisensi MIT - lihat file <a href="LICENSE" target="_blank">LICENSE</a> untuk detailnya.
        </p>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js" integrity="sha384-pzjw8f+ua7Kw1TIq0e0vZj8+6fjzQ6J45CT6VxKh+haW49xB5GptX6+pAkI+z6iP" crossorigin="anonymous"></script>
</body>
</html>
