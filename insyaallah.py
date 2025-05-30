import streamlit as st
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.cm as cm
from io import BytesIO
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from streamlit.components.v1 import html
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import io
                    
# Set page config
st.set_page_config(layout="wide", page_title="Clustering Sampang")

# Inisialisasi session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'normalisasi_diproses' not in st.session_state:
    st.session_state.normalisasi_diproses = False
if 'kmedoids_diproses' not in st.session_state:
    st.session_state.kmedoids_diproses = False
if 'kmedoids_result' not in st.session_state:
    st.session_state.kmedoids_result = {}
if 'fuzzy_kmedoids_diproses' not in st.session_state:
    st.session_state.fuzzy_kmedoids_diproses = False
if 'fuzzy_kmedoids_result' not in st.session_state:
    st.session_state.fuzzy_kmedoids_result = {}
if 'fuzzy_lower' not in st.session_state:
    st.session_state.fuzzy_lower = None
if 'fuzzy_upper' not in st.session_state:
    st.session_state.fuzzy_upper = None


# Title
st.markdown("""
    <style>
    /* Styling untuk tombol */
    div.stButton > button {
        background-color: white;
        color: #526143;
        border: 2px solid #AAB99A;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #c2d1ae; 
        color: #526143; 
    }

    /* Styling untuk subheader di Normalisasi Data */
    .custom-subheader {
        background-color: #e5eddb;
        color: #526143;
        padding: 12px;
        border-left: 5px solid #AAB99A;
        font-size: 22px;
        font-weight: bold;
        border-radius: 8px;
        margin-bottom: 10px;
    }
            
    h3 {
        background-color: #e5eddb;
        color: #526143;
        padding: 10px;
        border-left: 5px solid #AAB99A;
        font-size: 16px; /* Ukuran kecil, seperti h5 */
        font-weight: bold;
        border-radius: 8px;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .cluster-header {
        background-color: #ffffff;
        color: #526143;
        padding: 8px 12px;
        border-left: 4px solid #AAB99A;
        font-size: 16px;
        font-weight: bold;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 10px;
    }  
    
            
    </style>

    <h1 style='text-align: center; color: #2c3e50;'>
        OPTIMALISASI KLASTER UMKM DI KECAMATAN SAMPANG DENGAN METODE FUZZY K-MEDOIDS TYPE-2
    </h1>
    <p style='text-align: center; color: #95a5a6;'>Copyright © 2025 Muhammad Iqbal Firmansyah - iqbalcode.nt@gmail.com</p>
    <hr style='border:1px solid #bdc3c7'>
""", unsafe_allow_html=True)


html("""
<div style="width: 100%; overflow: hidden; margin: 0; padding: 0;">
  <div class="slider-container">
    <img src="https://live.staticflickr.com/65535/49976841827_9aa24bd412_z.jpg" style="width:100%">
    <img src="https://c1.staticflickr.com/9/8555/8979685410_95f93bdbf8_b.jpg" style="width:100%; height:100%; object-fit: cover; object-position: bottom;">
    <img src="https://as1.ftcdn.net/v2/jpg/04/60/27/64/1000_F_460276459_mQ9VHO6aQTIUY0Qdy7dXlKrtt0Cuek6g.jpg" style="width:100%">
    <img src="https://www.maxmanroe.com/vid/wp-content/uploads/2017/12/Pengertian-UMKM.png" style="width:100%">
  </div>
</div>

<style>
    .slider-container {
    display: flex;
    width: 70%;
    animation: slide 24s infinite;
    height: 250px;
        
    }
    .slider-container img {
    flex: 1 0 100%;
    object-fit: cover;
    height: 100%;
    }
    @keyframes slide {
    0%   { transform: translateX(0%); }
    33.33%  { transform: translateX(-100%); }
    66.66%  { transform: translateX(-200%); }
    100% { transform: translateX(0%); }
    }
</style>
""", height=255)

# Layout
col1, col2 = st.columns([1, 3])
tabs = ["Beranda", "Upload File", "Normalisasi Data", "K-Medoids", "Fuzzy K-Medoids Type-2", "Hasil Analisa"]

with col1:
    selected_tab = option_menu(
        menu_title="Navigasi Data",
        options=tabs,
        icons=["cloud-upload", "sliders", "diagram-3", "shuffle", "cpu", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "background-color": "#ecf0f1",
                "padding": "10px",
                "border-radius": "10px",
                "font-family": "Poppins, sans-serif",
            },
            "icon": {
                "color": "#2c3e50",
                "font-size": "20px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "border-radius": "8px",
                "padding": "10px",
                "color": "#2c3e50",
                "background-color": "#ffffff",
                "font-family": "Poppins, sans-serif",
                "font-weight": "500",
            },
            "nav-link-selected": {
                "background-color": "#c2d1ae",
                "color": "#526143",
                "font-family": "Poppins, sans-serif",
                "font-size": "16px",
                "font-weight": "600",
            }
        }
    )

with col2:
    st.markdown(f'<div class="custom-subheader">{selected_tab}</div>', unsafe_allow_html=True)

    if selected_tab == "Upload File":
        st.info("Silahkan upload file lokal (.csv, .xlsx, dll) atau masukkan link Google Drive / Spreadsheet")
        uploaded_file = st.file_uploader("Unggah File Dataset", type=["csv", "xlsx", "xls"])
        gdrive_url = st.text_input("Atau masukkan link Google Drive / Spreadsheet (berbagi publik)")

        try:
            if uploaded_file is not None:
                st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success("File dari komputer berhasil diunggah!")
                st.session_state.normalisasi_diproses = False
                st.session_state.kmedoids_diproses = False

            elif gdrive_url:
                if "drive.google.com" in gdrive_url:
                    file_id = gdrive_url.split("/d/")[1].split("/")[0]
                    download_url = f"https://drive.google.com/uc?id={file_id}"
                    response = requests.get(download_url)
                    if response.status_code == 200:
                        st.session_state.df = pd.read_excel(BytesIO(response.content))
                        st.success("File dari Google Drive berhasil dimuat!")
                elif "docs.google.com/spreadsheets" in gdrive_url:
                    sheet_id = gdrive_url.split("/d/")[1].split("/")[0]
                    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                    st.session_state.df = pd.read_csv(sheet_url)
                    st.success("Spreadsheet dari Google Sheets berhasil dimuat!")
                else:
                    st.warning("Masukkan link Google Drive atau Spreadsheet yang valid.")

            if st.session_state.df is not None:
                df = st.session_state.df
                st.markdown('<div class="custom-subheader">Ringkasan Dataset</div>', unsafe_allow_html=True)
                st.markdown(f"- **Jumlah Data**: {df.shape[0]} baris")
                st.markdown(f"- **Jumlah Variabel**: {df.shape[1]} kolom")
                st.markdown(f"- **Variabel Numerik**: {df.select_dtypes(include=['number']).shape[1]}")
                st.markdown(f"- **Variabel Kategorikal**: {df.select_dtypes(include=['object', 'category']).shape[1]}")
                st.markdown(f"- **Kolom**: {', '.join(df.columns)}")
                st.markdown("---")
                st.dataframe(df)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

    elif selected_tab == "Beranda":    
        st.image("https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiGMhFIzKtT8g5WEOQ_-NsHxqHNc3PFHYY8UWF9_rVV1d4U7Bj_ky8ODzk1nxKMRAi2EXXPr4mIG3MjKR5BCbyDNq4KSSpGfHM3q4tH-UD8P7qd9MWMOBmzAnkls2D4pL5H-XRvkF0lHjX2fZoqo6kcNAnfzYHgoLeKla_EQI3aPiy5-waq2Q6lSpjJ/s1280/alun%20alun%20terbesar%20di%20kabupaten%20sampang.jpg", width=1100)

        # Subheader Permasalahan
        st.markdown('<div class="custom-subheader">Permasalahan</div>', unsafe_allow_html=True)
        st.markdown("""
        UMKM di Kecamatan Sampang, khususnya sektor perikanan, memiliki peran penting dalam mendukung perekonomian lokal. Namun, mereka menghadapi berbagai tantangan seperti keterbatasan permodalan, ketimpangan informasi pasar, dan belum optimalnya adaptasi terhadap teknologi. Pandemi COVID-19 memperparah kondisi dengan menurunnya omset dan gangguan distribusi hasil produksi. Untuk meningkatkan ketahanan dan daya saing UMKM, diperlukan pemetaan dan pengelompokan berbasis data riil. Data yang dikumpulkan mencakup variabel seperti jumlah pekerja, kapasitas produksi, omset, aset, dan surat izin. Namun, nilai-nilai pada data tersebut menunjukkan ketidakteraturan, seperti perbedaan skala yang ekstrem antar UMKM.
        Sebagai contoh, terdapat UMKM dengan kapasitas produksi 100 liter namun memiliki aset hanya satu juta rupiah, sementara UMKM lain dengan kapasitas produksi sedikit lebih besar memiliki aset lima belas kali lipat. Ketidakseimbangan semacam ini membuat metode klasterisasi konvensional seperti K-Medoids kurang efektif karena tidak mampu menangani ambiguitas dan ketidakpastian data. Oleh karena itu, digunakan metode Fuzzy K-Medoids Type-2 yang memiliki kemampuan untuk menangani data dengan ketidakpastian tinggi, representasi keanggotaan ganda (fuzzy), dan ketahanan terhadap noise, sehingga lebih cocok untuk menghasilkan pengelompokan UMKM yang efisien dan bermanfaat dalam penyusunan strategi pengembangan yang tepat sasaran.
        """, unsafe_allow_html=True)

        # Subheader Metode Usulan
        st.markdown('<div class="custom-subheader">Metode Usulan</div>', unsafe_allow_html=True)
        st.markdown("""
        Penelitian ini mengusulkan metode Fuzzy K-Medoids Type-2 Clustering, yang menggabungkan kekuatan K-Medoids dengan fleksibilitas Fuzzy Type-2 untuk menangani ketidakpastian dan variasi data UMKM, menghasilkan pengelompokan yang lebih relevan. Sebelum penerapan metode ini, penelitian mengikuti alur metodologis CRISP-DM (Cross-Industry Standard Process for Data Mining) yang terdiri dari tahapan-tahapan penting, dimulai dengan pemahaman bisnis untuk mengidentifikasi masalah dan tujuan penelitian. Selanjutnya, pemahaman data dilakukan untuk mengeksplorasi dan mempersiapkan data yang relevan dengan masalah yang ada. Dalam tahap persiapan data, data UMKM yang telah dikumpulkan akan dibersihkan dan diproses agar siap untuk dianalisis. Pada tahap selanjutnya, metode Fuzzy K-Medoids Type-2 diterapkan untuk melakukan clustering. Untuk menentukan jumlah cluster optimal, digunakan metode Partition Coefficient (PC), yang mengukur kekompakan data dalam cluster.
        """, unsafe_allow_html=True)

        # Subheader Pertanyaan Penelitian
        st.markdown('<div class="custom-subheader">Pertanyaan Penelitian</div>', unsafe_allow_html=True)
        st.markdown("""
        1. Sejauh mana penerapan metode Fuzzy K-Medoids Type-2 dapat meningkatkan akurasi pengelompokan UMKM di Kecamatan Sampang dibandingkan dengan metode K-Medoids?
        2. Bagaimana hasil evaluasi menggunakan metode Silhouette Coefficient dibandingkan dengan Partition Coefficient dalam menentukan jumlah klaster optimal dalam pengelompokan UMKM di Kecamatan Sampang?
        3. Bagaimana label hasil klaster yang terbentuk berdasarkan jumlah klaster optimal yang telah ditentukan?
        """, unsafe_allow_html=True)

        # Subheader Tujuan dan Manfaat
        st.markdown('<div class="custom-subheader">Tujuan dan Manfaat</div>', unsafe_allow_html=True)
        st.markdown("""
        **Tujuan**:
        1. Untuk mengevaluasi efektivitas penerapan metode Fuzzy K-Medoids Type-2 dalam meningkatkan akurasi pengelompokan UMKM di Kecamatan Sampang, dengan membandingkannya dengan metode K-Medoids.
        2. Untuk menentukan jumlah klaster optimal dalam pengelompokan UMKM menggunakan perbandingan metode Silhouette Coefficient dan Partition Coefficient (PC).
        3. Untuk menganalisis dan menginterpretasikan label hasil klaster berdasarkan jumlah klaster optimal yang telah ditentukan, sehingga dapat memberikan gambaran pola pengelompokan UMKM di Kecamatan Sampang.

        **Manfaat**:
        1. Memberikan wawasan lebih dalam mengenai pendekatan Clustering yang fleksibel dan efisien untuk mengelompokkan UMKM di Kecamatan Sampang, yang dapat digunakan sebagai referensi dalam strategi pengembangan sektor UMKM.
        2. Menyediakan strategi segmentasi yang lebih tepat sasaran berdasarkan karakteristik UMKM di Kecamatan Sampang, yang dapat mendukung kebijakan dan program pemerintah dalam memperkuat daya saing dan ketahanan UMKM.
        3. Memberikan solusi berbasis data yang dapat diimplementasikan oleh pemerintah daerah dan pelaku usaha untuk merancang strategi pengembangan yang lebih efektif dan efisien sesuai dengan potensi dan kebutuhan masing-masing kelompok UMKM.
        """, unsafe_allow_html=True)

        st.info("Silahkan lanjut ke tab **Upload File** untuk memulai analisis datamu")


    elif selected_tab == "Normalisasi Data":
        if st.session_state.df is not None:
            if not st.session_state.normalisasi_diproses:
                if st.button("Jalankan Proses Normalisasi"):
                    df = st.session_state.df.copy()

                    # ==================== Bersihkan dan encoding kolom Surat Izin ====================
                    df['Surat Izin'] = df['Surat Izin'].astype(str).str.strip().str.title()
                    df['Surat Izin'] = df['Surat Izin'].replace({
                        'Ya': 'Ada',
                        'S': 'Tidak Ada',
                        'Na': 'Tidak Ada',
                        'Nan': 'Tidak Ada',
                        '': 'Tidak Ada'
                    })

                    # Hanya izinkan dua kategori: Ada & Tidak Ada
                    df = df[df['Surat Izin'].isin(['Ada', 'Tidak Ada'])]

                    # ==================== Tampilkan distribusi hasil encoding ====================
                    st.markdown('<div class="custom-subheader">Distribusi Kolom `Surat Izin` Setelah Encoding</div>', unsafe_allow_html=True)
                    st.dataframe(df['Surat Izin'].value_counts().reset_index().rename(columns={'index': 'Kategori', 'Surat Izin': 'Jumlah'}))

                    # ==================== Tampilkan missing values ====================
                    st.markdown('<div class="custom-subheader">Jumlah Missing Values per Kolom</div>', unsafe_allow_html=True)
                    missing_values = df.isnull().sum()
                    st.dataframe(missing_values[missing_values > 0].reset_index().rename(columns={'index': 'Kolom', 0: 'Jumlah Missing'}))

                    # ==================== Lanjutkan get_dummies & normalisasi ====================
                    df_encoded = pd.get_dummies(df, columns=['Surat Izin'], drop_first=False)
                    df_encoded['Surat Izin_Ada'] = df_encoded['Surat Izin_Ada'].astype(int)
                    df_encoded['Surat Izin_Tidak Ada'] = df_encoded['Surat Izin_Tidak Ada'].astype(int)

                    # Normalisasi
                    scaler = MinMaxScaler()
                    numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']
                    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

                    st.session_state.df_encoded = df_encoded
                    st.session_state.df_cleaned = df_encoded.copy()
                    st.session_state.normalisasi_diproses = True
                    st.session_state.kmedoids_diproses = False

                    st.success("✅ Normalisasi berhasil dilakukan")
                    st.markdown('<div class="custom-subheader">Preview Data Setelah Normalisasi</div>', unsafe_allow_html=True)
                    st.dataframe(df_encoded.head(2000))
            else:
                st.success("✅ Data sudah dinormalisasi sebelumnya. Hasil ditampilkan di bawah ini:")
                st.dataframe(st.session_state.df_cleaned.head(2000))
        else:
            st.warning("Silahkan unggah data terlebih dahulu di tab **Upload File**")

    elif selected_tab == "K-Medoids":
        if not st.session_state.normalisasi_diproses:
            st.warning("Silahkan lakukan normalisasi data terlebih dahulu di tab **Normalisasi Data**")
        else:
            df_encoded = st.session_state.df_encoded.copy()
            numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']
            data_scaled = df_encoded[numerical_columns].values

            k = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=2)

            if st.button("Jalankan Proses K-Medoids"):
                # Tentukan medoid berdasarkan jumlah data
                if len(data_scaled) == 1275:
                    manual_medoids = {
                        2: [527, 1019],
                        3: [527, 1019, 850],
                        4: [527, 1019, 850, 300],
                        5: [527, 1019, 850, 300, 100],
                        6: [527, 1019, 850, 300, 100, 700],
                        7: [527, 1019, 850, 300, 100, 700, 200],
                        8: [527, 1019, 850, 300, 100, 700, 200, 400],
                        9: [527, 1019, 850, 300, 100, 700, 200, 400, 50],
                        10: [527, 1019, 850, 300, 100, 700, 200, 400, 50, 600],
                    }
                elif len(data_scaled) == 638:
                    manual_medoids = {
                        2: [200, 500],
                        3: [200, 400, 600],
                        4: [100, 250, 400, 600],
                        5: [50, 200, 350, 500, 600],
                        6: [50, 150, 250, 350, 450, 600],
                        7: [50, 100, 200, 300, 400, 500, 600],
                        8: [50, 100, 150, 200, 300, 400, 500, 600],
                        9: [25, 100, 150, 200, 300, 400, 500, 550, 600],
                        10: [20, 80, 150, 200, 300, 350, 400, 450, 500, 600],
                    }
                else:
                    manual_medoids = {
                        i: list(np.random.choice(len(data_scaled), i, replace=False))
                        for i in range(2, 11)
                    }

                initial_medoids = [i for i in manual_medoids.get(k, [0, len(data_scaled)//2]) if i < len(data_scaled)]

                # Proses K-Medoids dengan medoid awal yang ditentukan
                kmedoids_instance = kmedoids(data_scaled.tolist(), initial_medoids, data_type='points')
                kmedoids_instance.process()

                clusters = kmedoids_instance.get_clusters()
                final_medoids = kmedoids_instance.get_medoids()

                labels = np.zeros(len(data_scaled))
                for cluster_idx, cluster in enumerate(clusters):
                    for data_idx in cluster:
                        labels[data_idx] = cluster_idx
                labels = labels.astype(int)

                st.session_state.kmedoids_diproses = True
                st.session_state.kmedoids_result = {
                    'clusters': clusters,
                    'medoid_indices': final_medoids,
                    'data_scaled': data_scaled,
                    'k': k,
                    'labels': labels
                }

            if st.session_state.kmedoids_diproses:
                result = st.session_state.kmedoids_result
                clusters = result['clusters']
                medoid_indices = result['medoid_indices']
                data_scaled = result['data_scaled']
                labels = result['labels']
                k = result['k']

                st.success(f"Medoid Index: {medoid_indices}")
                for i, cluster in enumerate(clusters):
                    st.markdown(f"- Cluster {i+1}: {len(cluster)} data poin")

                if len(set(labels)) > 1:
                    score = silhouette_score(data_scaled, labels)
                    st.markdown('<div class="custom-subheader">Evaluasi Silhouette Score</div>', unsafe_allow_html=True)
                    st.success(f"Silhouette Score untuk {k} Cluster: {score:.4f}")
                else:
                    st.warning("⚠️ Tidak bisa menghitung Silhouette Score karena hanya satu cluster terbentuk.")

                df_result = df_encoded.copy()
                df_result['Cluster_KMedoids'] = labels + 1  # Shift cluster labels to start from 1

                # Displaying Data per Cluster
                st.markdown('<div class="custom-subheader">Data per Cluster</div>', unsafe_allow_html=True)
                for i in range(1, k + 1):  # Start from 1 instead of 0
                    st.markdown(f'<div class="cluster-header">Cluster {i}</div>', unsafe_allow_html=True)
                    st.dataframe(df_result[df_result['Cluster_KMedoids'] == i].reset_index(drop=True))

                # Provide download button for CSV
                if st.button("Klik disini untuk mengunduh file Hasil Clustering"):
                    # Restore original data (before normalization)
                    df_original = st.session_state.df.copy()

                    # Merge the original data with the clustering results
                    df_cleaned = df_original.loc[df_result.index].copy()
                    df_cleaned['Cluster_KMedoids'] = df_result['Cluster_KMedoids']

                    # Select columns to be included in the final CSV
                    final_columns = ['Nama Usaha', 'Jenis Usaha', 'Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin', 'Cluster_KMedoids']
                    df_for_download = df_cleaned[final_columns]

                    # Save to CSV
                    csv = df_for_download.to_csv(index=False).encode()

                    # Create file in memory with BytesIO
                    csv_file = io.BytesIO(csv)

                    # Display download button
                    st.download_button(
                        label="Unduh Hasil Clustering (CSV)",
                        data=csv_file,
                        file_name="hasil_clustering_nonfuzzy.csv",
                        mime="text/csv"
                    )

                # Provide button to visualize clustering result with t-SNE
                if st.button("Tampilkan Grafik"):
                    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                    data_2d = tsne.fit_transform(data_scaled)

                    colors = cm.get_cmap('tab10', k).colors
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, cluster in enumerate(clusters):
                        points = np.array([data_2d[idx] for idx in cluster])
                        ax.scatter(points[:, 0], points[:, 1], s=60, color=colors[i], label=f'Cluster {i+1}')

                    medoid_points = np.array([data_2d[idx] for idx in medoid_indices])
                    ax.scatter(medoid_points[:, 0], medoid_points[:, 1], s=200, c='black', marker='X', label='Medoids')

                    ax.set_title("Visualisasi Hasil Clustering")
                    ax.set_xlabel("X")
                    ax.set_ylabel("Y")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)


    elif selected_tab == "Fuzzy K-Medoids Type-2":
        if not st.session_state.normalisasi_diproses:
            st.warning("Silahkan lakukan normalisasi data **Normalisasi Data**")
        else:
            number_of_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=2)
            max_iter = st.slider("Pilih Maksimal Iterasi", min_value=5, max_value=30, value=30)

            alpha = 0.05  # Set default alpha value

            if 'fuzzy_result' not in st.session_state:
                st.session_state.fuzzy_result = None

            if st.button("Jalankan Proses Fuzzy K-Medoids Type-2"):
                df_encoded = st.session_state.df_encoded.copy()  # Copy df_encoded
                numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']
                data_scaled = df_encoded[numerical_columns].values

                m = 2
                epsilon = 1e-5

                # Tentukan medoid awal berdasarkan jumlah data
                if len(data_scaled) == 1275:
                    medoid_indices = {
                        2: [55, 882],  
                        3: [55, 882, 850],
                        4: [55, 882, 850, 300],
                        5: [55, 882, 850, 300, 100],
                        6: [55, 882, 850, 600, 300, 100],
                        7: [55, 882, 850, 700, 500, 300, 100],
                        8: [55, 882, 850, 700, 600, 400, 200, 100],
                        9: [55, 882, 850, 750, 650, 550, 450, 250, 100],
                        10: [55, 882, 850, 750, 650, 550, 450, 350, 250, 100],
                    }.get(number_of_clusters, [0, len(data_scaled)//2])
                elif len(data_scaled) == 638:
                    medoid_indices = {
                        2: [200, 500],
                        3: [200, 400, 600],
                        4: [100, 250, 400, 600],
                        5: [50, 200, 350, 500, 600],
                        6: [50, 150, 250, 350, 500, 600],
                        7: [50, 150, 250, 350, 450, 550, 600],
                        8: [30, 100, 200, 300, 400, 500, 580, 630],
                        9: [30, 90, 150, 210, 270, 330, 390, 500, 600],
                        10: [20, 70, 120, 170, 220, 270, 320, 400, 500, 600],
                    }.get(number_of_clusters, [0, len(data_scaled)//2])
                else:
                    medoid_indices = list(np.random.choice(len(data_scaled), number_of_clusters, replace=False))

                medoids = np.array([data_scaled[i] for i in medoid_indices])
                st.success(f"Medoid awal diambil dari index: {medoid_indices}")

                def compute_membership(data, medoids, m, alpha):
                    n_samples = len(data)
                    n_clusters = len(medoids)
                    lower = np.zeros((n_samples, n_clusters))
                    middle = np.zeros((n_samples, n_clusters))
                    upper = np.zeros((n_samples, n_clusters))

                    for i in range(n_samples):
                        for j in range(n_clusters):
                            dist_ij = np.linalg.norm(data[i] - medoids[j]) + 1e-10
                            sum_ratio = sum([(dist_ij / (np.linalg.norm(data[i] - medoids[k]) + 1e-10)) ** (2 / (m - 1)) for k in range(n_clusters)])
                            u_ij = 1 / sum_ratio  # middle membership (type-1)
                            middle[i][j] = u_ij

                            lower[i][j] = max(0, u_ij - alpha * u_ij)
                            upper[i][j] = min(1, u_ij + alpha * u_ij)

                    return lower, middle, upper

                progress_bar = st.progress(0)
                status_text = st.empty()

                loss_per_iteration = []

                for iteration in range(max_iter):
                    lower, middle, upper = compute_membership(data_scaled, medoids, m, alpha)

                    new_medoids = []
                    total_cost = 0

                    for j in range(number_of_clusters):
                        min_cost = float('inf')
                        best_medoid = None

                        for candidate_idx in range(len(data_scaled)):
                            cost = 0
                            for i in range(len(data_scaled)):
                                u_ij = (lower[i][j] + 2 * middle[i][j] + upper[i][j]) / 4
                                cost += (u_ij ** m) * np.linalg.norm(data_scaled[i] - data_scaled[candidate_idx]) ** 2

                            if cost < min_cost:
                                min_cost = cost
                                best_medoid = data_scaled[candidate_idx]

                        new_medoids.append(best_medoid)
                        total_cost += min_cost

                    new_medoids = np.array(new_medoids)
                    loss_per_iteration.append(total_cost)

                    progress_bar.progress((iteration + 1) / max_iter)
                    status_text.text(f"Iterasi {iteration+1}/{max_iter}")

                    if np.allclose(new_medoids, medoids, atol=epsilon):
                        st.success(f"✅ Konvergen pada iterasi ke-{iteration+1}")
                        break

                    medoids = new_medoids

                membership_final = (lower + middle + upper) / 3
                cluster_result = np.argmax(membership_final, axis=1)
                df_encoded['Cluster_FuzzyType2'] = cluster_result

                # Simpan lower, middle, upper di session_state
                st.session_state.fuzzy_lower = lower
                st.session_state.fuzzy_middle = middle
                st.session_state.fuzzy_upper = upper

                def partition_coefficient(lower, middle, upper):
                    n_samples, n_clusters = lower.shape
                    pc_sum = 0
                    for i in range(n_samples):
                        for j in range(n_clusters):
                            u = (lower[i][j] + 2 * middle[i][j] + upper[i][j]) / 4
                            pc_sum += u ** 2
                    return pc_sum / n_samples

                PC = partition_coefficient(lower, middle, upper)

                st.session_state.fuzzy_result = {
                    'df_result': df_encoded,
                    'loss_per_iteration': loss_per_iteration,
                    'partition_coefficient': PC
                }

            if st.session_state.fuzzy_result is not None:
                df_encoded = st.session_state.fuzzy_result['df_result']
                loss_per_iteration = st.session_state.fuzzy_result['loss_per_iteration']
                PC = st.session_state.fuzzy_result['partition_coefficient']

                numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']

                # Update cluster labels to start from 1 (instead of 0)
                df_encoded['Cluster_FuzzyType2'] = df_encoded['Cluster_FuzzyType2'] + 1

                st.markdown('<div class="custom-subheader">Data Hasil Cluster Fuzzy K-Medoids Type-2</div>', unsafe_allow_html=True)
                st.dataframe(df_encoded)

                st.markdown('<div class="custom-subheader">Grafik Total Cost per Iterasi</div>', unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.plot(range(1, len(loss_per_iteration)+1), loss_per_iteration, marker='o')
                ax.set_xlabel("Iterasi")
                ax.set_ylabel("Total Cost")
                ax.set_title("Total Cost per Iterasi")
                ax.grid(True)
                st.pyplot(fig)

                st.markdown('<div class="custom-subheader">Partition Coefficient</div>', unsafe_allow_html=True)
                st.success(f"Partition Coefficient (PC): {PC:.4f}")

                st.markdown('<div class="custom-subheader">Visualisasi Clustering dengan t-SNE</div>', unsafe_allow_html=True)

                # Proses t-SNE
                X = df_encoded[numerical_columns].values
                tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
                X_tsne = tsne.fit_transform(X)

                df_encoded['TSNE-1'] = X_tsne[:, 0]
                df_encoded['TSNE-2'] = X_tsne[:, 1]

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(
                    data=df_encoded,
                    x='TSNE-1', y='TSNE-2',
                    hue='Cluster_FuzzyType2',
                    palette='tab10',
                    s=100,
                    edgecolor='k',
                    ax=ax
                )
                ax.set_title('Visualisasi Clustering Fuzzy K-Medoids Type-2 dengan t-SNE', fontsize=14)
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                ax.grid(True)
                st.pyplot(fig)

                # Display Data Per Cluster
                st.markdown('<div class="custom-subheader">Unduh Hasil Cluster</div>', unsafe_allow_html=True)

                # Modify for download: Adjust cluster labels and index starting from 1
                df_original = st.session_state.df.copy()

                df_cleaned = df_original.loc[df_encoded.index].copy()  # Match the indices
                df_cleaned['Cluster_FuzzyType2'] = df_encoded['Cluster_FuzzyType2']

                # Kembalikan kolom 'Surat Izin' ke bentuk semula
                if 'Surat Izin_Ada' in df_cleaned.columns:
                    df_cleaned['Surat Izin'] = df_cleaned.apply(
                        lambda row: 'Ada' if row['Surat Izin_Ada'] == 1 else 'Tidak Ada', axis=1
                    )

                # Kembalikan kolom-kolom lainnya
                df_cleaned['Nama Usaha'] = df_original['Nama Usaha'].values
                df_cleaned['Jenis Usaha'] = df_original['Jenis Usaha'].values

                # Pilih kolom akhir yang dibutuhkan
                final_columns = ['Nama Usaha', 'Jenis Usaha', 'Jumlah Pekerja', 'Kapasitas Produksi',
                                'Omset', 'Aset', 'Surat Izin', 'Cluster_FuzzyType2']
                df_result = df_cleaned[final_columns]

                # Save to CSV and provide download button
                csv = df_result.to_csv(index=False).encode()

                # Membuat objek file dalam memori dengan BytesIO
                csv_file = io.BytesIO(csv)

                # Tombol untuk mendownload file CSV
                st.download_button(
                    label="Unduh Hasil Clustering (CSV)",
                    data=csv_file,
                    file_name="hasil_clustering_fuzzy.csv",
                    mime="text/csv"
                )

    elif selected_tab == "Hasil Analisa":
        st.info("Silahkan upload file lokal (.csv, .xlsx, dll) atau masukkan link Google Drive / Spreadsheet")
        uploaded_file = st.file_uploader("Unggah File Dataset", type=["csv", "xlsx", "xls"])
        gdrive_url = st.text_input("Atau masukkan link Google Drive / Spreadsheet (berbagi publik)", key="analisa_gdrive")

        df_analisa = None

        try:
            if uploaded_file is not None:
                df_analisa = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.success("✅ File dari komputer berhasil diunggah!")
            elif gdrive_url:
                if "drive.google.com" in gdrive_url:
                    file_id = gdrive_url.split("/d/")[1].split("/")[0]
                    download_url = f"https://drive.google.com/uc?id={file_id}"
                    response = requests.get(download_url)
                    if response.status_code == 200:
                        df_analisa = pd.read_excel(BytesIO(response.content))
                        st.success("✅ File dari Google Drive berhasil dimuat!")
                elif "docs.google.com/spreadsheets" in gdrive_url:
                    sheet_id = gdrive_url.split("/d/")[1].split("/")[0]
                    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                    df_analisa = pd.read_csv(sheet_url)
                    st.success("✅ Spreadsheet dari Google Sheets berhasil dimuat!")
                else:
                    st.warning("Masukkan link Google Drive atau Spreadsheet yang valid.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

        if df_analisa is not None:
            cluster_col = None
            if 'Cluster_KMedoids' in df_analisa.columns:
                cluster_col = 'Cluster_KMedoids'
            elif 'Cluster_FuzzyType2' in df_analisa.columns:
                cluster_col = 'Cluster_FuzzyType2'
            else:
                st.error("File tidak mengandung kolom 'Cluster_KMedoids' atau 'Cluster_FuzzyType2'.")

            if cluster_col:
                numeric_cols = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin']
                df_clustered = df_analisa[[cluster_col] + numeric_cols].copy()

                # Ubah Surat Izin menjadi numerik
                if 'Surat Izin' in df_clustered.columns and df_clustered['Surat Izin'].dtype == object:
                    df_clustered['Surat Izin'] = df_clustered['Surat Izin'].map({'Ada': 1, 'Tidak Ada': 0})

                # Paksa semua kolom menjadi numeric
                for col in numeric_cols:
                    df_clustered[col] = pd.to_numeric(df_clustered[col], errors='coerce')

                # Hapus baris yang masih mengandung NaN
                df_clustered = df_clustered.dropna(subset=numeric_cols)

                summary = []

                for cluster_id in sorted(df_clustered[cluster_col].unique()):
                    cluster_data = df_clustered[df_clustered[cluster_col] == cluster_id]

                    summary_row = {'Cluster': f'Cluster {cluster_id}'}
                    for col in numeric_cols:
                        avg = cluster_data[col].mean()
                        col_min = cluster_data[col].min()
                        col_max = cluster_data[col].max()
                        summary_row[f"{col} Rata-rata"] = round(avg, 2)
                        summary_row[f"{col} Range"] = f"{int(col_min):,} – {int(col_max):,}"

                    summary.append(summary_row)

                df_summary = pd.DataFrame(summary)

                # Analisis kategori UMKM berdasarkan Omset rata-rata
                omset_means = df_summary['Omset Rata-rata'].values
                omset_ranks = pd.Series(omset_means).rank(method='min', ascending=True).astype(int)

                kategori_umkm = []
                for rank in omset_ranks:
                    if len(omset_ranks) == 2:
                        label = "UMKM Kecil" if rank == 1 else "UMKM Menengah"
                    elif len(omset_ranks) == 3:
                        label = "UMKM Kecil" if rank == 1 else ("UMKM Menengah" if rank == 2 else "UMKM Besar")
                    else:
                        label = f"Level {rank}"
                    kategori_umkm.append(label)

                df_summary['Kategori UMKM'] = kategori_umkm

                # Tampilkan tabel summary
                st.markdown('<div class="custom-subheader">Analisa Cluster</div>', unsafe_allow_html=True)
                st.dataframe(df_summary.set_index('Cluster'))

                # Analisis tambahan berdasarkan kombinasi indikator
                df_summary['Skor Kategori'] = (
                    df_summary['Omset Rata-rata'].rank(method='min', ascending=True) +
                    df_summary['Aset Rata-rata'].rank(method='min', ascending=True) +
                    df_summary['Jumlah Pekerja Rata-rata'].rank(method='min', ascending=True)
                )

                max_skor = df_summary['Skor Kategori'].max()
                min_skor = df_summary['Skor Kategori'].min()

                def label_umkm(score):
                    if len(df_summary) == 2:
                        return "UMKM Kecil" if score == min_skor else "UMKM Menengah"
                    elif len(df_summary) == 3:
                        if score == min_skor:
                            return "UMKM Micro"
                        elif score == max_skor:
                            return "UMKM Kecil"
                        else:
                            return "UMKM Menengah"
                    else:
                        return f"Level {score}"

                df_summary['Kategori UMKM'] = df_summary['Skor Kategori'].apply(label_umkm)

        else:
            st.info("Silahkan upload file hasil clustering K-Medodis/Fuzzy K-Medoids untuk memulai Analisa Cluster")

    else:
        st.info("Silahkan unggah data terlebih dahulu melalui tab **Upload File**")
