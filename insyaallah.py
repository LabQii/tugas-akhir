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
        OPTIMASI CLUSTER DI KECAMATAN SAMPANG <br> DENGAN METODE K-MEDOIDS DAN FUZZY K-MEDOIDS TYPE-2
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
tabs = ["Upload File", "Normalisasi Data", "K-Medoids", "Fuzzy K-Medoids Type-2", "Klasifikasi SVM", "Hasil Analisa"]

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
        st.info("Silakan upload file lokal (.csv, .xlsx, dll) atau masukkan link Google Drive / Spreadsheet")
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

    elif selected_tab == "Normalisasi Data":
        if st.session_state.df is not None:
            if not st.session_state.normalisasi_diproses:
                if st.button("Jalankan Proses Normalisasi"):
                    df = st.session_state.df.copy()
                    df_encoded = pd.get_dummies(df, columns=['Surat Izin'], drop_first=False)
                    df_encoded['Surat Izin_Ada'] = df_encoded['Surat Izin_Ada'].astype(int)
                    df_encoded['Surat Izin_Tidak Ada'] = df_encoded['Surat Izin_Tidak Ada'].astype(int)

                    # Normalisasi tanpa penghapusan outlier
                    scaler = MinMaxScaler()
                    numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']
                    df_encoded[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

                    st.session_state.df_encoded = df_encoded
                    st.session_state.df_cleaned = df_encoded.copy()
                    st.session_state.normalisasi_diproses = True
                    st.session_state.kmedoids_diproses = False

                    st.success("Normalisasi berhasil dilakukan")
                    st.dataframe(df_encoded.head(1000))
            else:
                st.success("✅ Data sudah dinormalisasi sebelumnya. Hasil ditampilkan di bawah ini:")
                st.dataframe(st.session_state.df_cleaned.head(1000))
        else:
            st.warning("Silakan unggah data terlebih dahulu di tab 'Upload File'.")


    elif selected_tab == "K-Medoids":
        if not st.session_state.normalisasi_diproses:
            st.warning("Silakan lakukan normalisasi data terlebih dahulu di tab 'Normalisasi Data'.")
        else:
            df_encoded = st.session_state.df_encoded.copy()
            numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']
            data_scaled = df_encoded[numerical_columns].values

            k = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=2)

            if st.button("Jalankan Proses K-Medoids"):
                # Gunakan medoid manual yang sudah ditentukan sebelumnya
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
                initial_medoids = [i for i in manual_medoids.get(k, [0, len(data_scaled)//2]) if i < len(data_scaled)]

                kmedoids_instance = kmedoids(data_scaled.tolist(), initial_medoids, data_type='points')
                kmedoids_instance.process()

                clusters = kmedoids_instance.get_clusters()
                final_medoids = kmedoids_instance.get_medoids()

                # Buat label untuk setiap data
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

                score = silhouette_score(data_scaled, labels)
                st.markdown('<div class="custom-subheader">Evaluasi Silhouette Score</div>', unsafe_allow_html=True)
                st.success(f"Silhouette Score untuk {k} Cluster: {score:.4f}")

                # Tampilkan data per cluster
                df_result = df_encoded.copy()
                df_result['Cluster'] = labels

                st.markdown('<div class="custom-subheader">Data per Cluster</div>', unsafe_allow_html=True)
                for i in range(k):
                    st.markdown(f'<div class="cluster-header"> Cluster {i+1}</div>', unsafe_allow_html=True)
                    st.dataframe(df_result[df_result['Cluster'] == i].reset_index(drop=True))

                # Tombol untuk menampilkan TSNE
                if st.button("Tampilkan Grafik TSNE"):
                    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
                    data_2d = tsne.fit_transform(data_scaled)

                    colors = cm.get_cmap('tab10', k).colors
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, cluster in enumerate(clusters):
                        points = np.array([data_2d[idx] for idx in cluster])
                        ax.scatter(points[:, 0], points[:, 1], s=60, color=colors[i], label=f'Cluster {i+1}')

                    # Plot medoid
                    medoid_points = np.array([data_2d[idx] for idx in medoid_indices])
                    ax.scatter(medoid_points[:, 0], medoid_points[:, 1], s=200, c='black', marker='X', label='Medoids')

                    ax.set_title("Visualisasi Hasil Clustering dengan TSNE")
                    ax.set_xlabel("TSNE-1")
                    ax.set_ylabel("TSNE-2")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)
    
    elif selected_tab == "Fuzzy K-Medoids Type-2":
        if not st.session_state.normalisasi_diproses:
            st.warning("Silakan lakukan normalisasi data terlebih dahulu di tab 'Normalisasi Data'.")
        else:
            number_of_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=5, value=2)
            max_iter = st.slider("Pilih Maksimal Iterasi", min_value=10, max_value=100, value=20)

            if 'fuzzy_result' not in st.session_state:
                st.session_state.fuzzy_result = None

            if st.button("Jalankan Proses Fuzzy K-Medoids Type-2"):
                df_encoded = st.session_state.df_encoded.copy()
                numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']
                data_scaled = df_encoded[numerical_columns].values

                m = 2
                epsilon = 1e-5

                dist_matrix = pairwise_distances(data_scaled)
                max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                idx1, idx2 = max_dist_idx
                medoids = np.array([data_scaled[idx1], data_scaled[idx2]])[:number_of_clusters]

                st.success(f"Medoid awal diambil dari index {idx1} dan {idx2}")

                def compute_membership(data, medoids, m):
                    n_samples = len(data)
                    n_clusters = len(medoids)
                    lower = np.zeros((n_samples, n_clusters))
                    upper = np.zeros((n_samples, n_clusters))

                    for i in range(n_samples):
                        for j in range(n_clusters):
                            dist_ij = np.linalg.norm(data[i] - medoids[j]) + 1e-10
                            sum_ratio = sum([(dist_ij / (np.linalg.norm(data[i] - medoids[k]) + 1e-10)) ** (2 / (m - 1)) for k in range(n_clusters)])
                            u_ij = 1 / sum_ratio
                            lower[i][j] = max(0, u_ij - 0.05)
                            upper[i][j] = min(1, u_ij + 0.05)

                    return lower, upper

                progress_bar = st.progress(0)
                status_text = st.empty()

                loss_per_iteration = []

                for iteration in range(max_iter):
                    lower, upper = compute_membership(data_scaled, medoids, m)

                    new_medoids = []
                    total_cost = 0

                    for j in range(number_of_clusters):
                        min_cost = float('inf')
                        best_medoid = None

                        for candidate_idx in range(len(data_scaled)):
                            cost = 0
                            for i in range(len(data_scaled)):
                                u_ij = (lower[i][j] + upper[i][j]) / 2
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

                cluster_result = np.argmax((lower + upper) / 2, axis=1)
                df_encoded['Cluster_FuzzyType2'] = cluster_result

                # Simpan lower dan upper
                st.session_state.fuzzy_lower = lower
                st.session_state.fuzzy_upper = upper

                st.session_state.fuzzy_result = {
                    'df_result': df_encoded,
                    'loss_per_iteration': loss_per_iteration,
                    'partition_coefficient': sum(((lower[i][j] + upper[i][j]) / 2) ** 2 for i in range(len(lower)) for j in range(len(lower[0]))) / len(lower)
                }

            if st.session_state.fuzzy_result is not None:
                df_encoded = st.session_state.fuzzy_result['df_result']
                loss_per_iteration = st.session_state.fuzzy_result['loss_per_iteration']
                PC = st.session_state.fuzzy_result['partition_coefficient']

                numerical_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin_Ada', 'Surat Izin_Tidak Ada']

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
                from sklearn.manifold import TSNE
                import seaborn as sns

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

                st.markdown('<div class="custom-subheader">Data Per Cluster</div>', unsafe_allow_html=True)
                for i in range(number_of_clusters):
                    st.markdown(f'<div class="cluster-header"> Cluster {i+1}</div>', unsafe_allow_html=True)
                    cluster_data = df_encoded[df_encoded['Cluster_FuzzyType2'] == i]
                    st.dataframe(cluster_data.reset_index(drop=True))

    elif selected_tab == "Klasifikasi SVM":
        st.info("Silakan upload file lokal (.csv, .xlsx, dll) atau masukkan link Google Drive / Spreadsheet")
        uploaded_file_svm = st.file_uploader("Unggah File Dataset", type=["csv", "xlsx", "xls"])
        gdrive_url = st.text_input("Atau masukkan link Google Drive / Spreadsheet (berbagi publik)")

        df_uploaded = None

        try:
            if uploaded_file_svm is not None:
                df_uploaded = pd.read_csv(uploaded_file_svm) if uploaded_file_svm.name.endswith(".csv") else pd.read_excel(uploaded_file_svm)
                st.success("✅ File dari komputer berhasil diunggah!")
            elif gdrive_url:
                if "drive.google.com" in gdrive_url:
                    file_id = gdrive_url.split("/d/")[1].split("/")[0]
                    download_url = f"https://drive.google.com/uc?id={file_id}"
                    response = requests.get(download_url)
                    if response.status_code == 200:
                        df_uploaded = pd.read_excel(BytesIO(response.content))
                        st.success("✅ File dari Google Drive berhasil dimuat!")
                elif "docs.google.com/spreadsheets" in gdrive_url:
                    sheet_id = gdrive_url.split("/d/")[1].split("/")[0]
                    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                    df_uploaded = pd.read_csv(sheet_url)
                    st.success("✅ Spreadsheet dari Google Sheets berhasil dimuat!")
                else:
                    st.warning("Masukkan link Google Drive atau Spreadsheet yang valid.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")

        if df_uploaded is not None:
            st.dataframe(df_uploaded.head(10))

            available_clusters = []
            if 'Cluster_KMedoids' in df_uploaded.columns:
                available_clusters.append('Cluster_KMedoids')
            if 'Cluster_FuzzyType2' in df_uploaded.columns:
                available_clusters.append('Cluster_FuzzyType2')

            if available_clusters:
                target_column = st.selectbox(
                    "Pilih Kolom Cluster untuk Klasifikasi", 
                    available_clusters, 
                    key="select_target_svm"
                )

                # Siapkan data X, y
                df_clean = df_uploaded.copy()

                if 'Surat Izin' in df_clean.columns and df_clean['Surat Izin'].dtype == object:
                    df_clean['Surat Izin'] = df_clean['Surat Izin'].map({'Ada': 1, 'Tidak Ada': 0})

                fitur_columns = ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset', 'Surat Izin']
                X = df_clean[fitur_columns]
                y = df_clean[target_column]

                # Tambahkan ini setelah ambil X
                for col in ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset']:
                    X[col] = pd.to_numeric(X[col], errors='coerce')


                for col in ['Jumlah Pekerja', 'Kapasitas Produksi', 'Omset', 'Aset']:
                    max_val = X[col].max()
                    if max_val != 0:
                        X[col] = X[col] / max_val

                X = X.dropna()
                y = y[X.index]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )

                if st.button("Jalankan Klasifikasi SVM"):
                    svm = SVC(kernel='rbf', random_state=42)
                    svm.fit(X_train, y_train)
                    y_pred = svm.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    st.success(f"Akurasi Model SVM : `{accuracy:.2f}`")

                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.markdown('<div class="custom-subheader">Classification Report</div>', unsafe_allow_html=True)
                    st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

                    cm = confusion_matrix(y_test, y_pred)
                    st.markdown('<div class="custom-subheader">Confusion Matrix</div>', unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title("Confusion Matrix - SVM")
                    st.pyplot(fig)

                if st.button("Hitung Cross Validation (CV=5)"):
                    cv_scores = cross_val_score(SVC(kernel='rbf', random_state=42), X, y, cv=5, scoring='accuracy')
                    st.markdown('<div class="custom-subheader">Hasil Cross Validation</div>', unsafe_allow_html=True)
                    st.write(f"Cross-validation scores: `{cv_scores}`")
                    st.write(f"Mean accuracy: `{np.mean(cv_scores):.2f}`")
                    st.write(f"Standard deviation: `{np.std(cv_scores):.2f}`")

                if st.button("Cek Overfitting / Underfitting"):
                    train_accuracies = []
                    test_accuracies = []
                    kernels = ['linear', 'rbf']

                    for kernel in kernels:
                        model = SVC(kernel=kernel, random_state=42)
                        model.fit(X_train, y_train)
                        train_acc = accuracy_score(y_train, model.predict(X_train))
                        test_acc = accuracy_score(y_test, model.predict(X_test))
                        train_accuracies.append(train_acc)
                        test_accuracies.append(test_acc)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    index = np.arange(len(kernels))
                    bar_width = 0.35

                    ax.bar(index, train_accuracies, bar_width, label='Training Accuracy', color='green', alpha=0.7)
                    ax.bar(index + bar_width, test_accuracies, bar_width, label='Test Accuracy', color='red', alpha=0.7)
                    ax.set_xlabel("Kernel Type")
                    ax.set_ylabel("Accuracy")
                    ax.set_title("Analisis Overfitting & Underfitting")
                    ax.set_xticks(index + bar_width / 2)
                    ax.set_xticklabels(kernels)
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    max_train_acc = max(train_accuracies)
                    max_test_acc = max(test_accuracies)

                    st.markdown('<div class="custom-subheader">Diagnosis Model</div>', unsafe_allow_html=True)
                    st.write(f"Max training accuracy: `{max_train_acc:.2f}`")
                    st.write(f"Max test accuracy: `{max_test_acc:.2f}`")

                    if max_train_acc - max_test_acc > 0.1:
                        st.warning("⚠️ Kemungkinan Overfitting terdeteksi!")
                    elif max_train_acc < 0.7 and max_test_acc < 0.7:
                        st.warning("⚠️ Kemungkinan Underfitting terdeteksi!")
                    else:
                        st.success("✅ Model berjalan dengan baik dan seimbang.")
            else:
                st.error("File tidak memiliki kolom 'Cluster_KMedoids' atau 'Cluster_FuzzyType2'.")
        else:
            st.info("Silakan upload file hasil clustering K-Medodis/Fuzzy K-Medoids untuk memulai Klasifikasi")

    elif selected_tab == "Hasil Analisa":
        st.info("Silakan upload file lokal (.csv, .xlsx, dll) atau masukkan link Google Drive / Spreadsheet")
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
                st.markdown('<div class="custom-subheader">Analisa Cluster Berdasarkan SPK</div>', unsafe_allow_html=True)
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
            st.info("Silakan upload file hasil clustering K-Medodis/Fuzzy K-Medoids untuk memulai Analisa Cluster")



    else:
        st.info("Silakan unggah data terlebih dahulu melalui tab 'Upload File'.")