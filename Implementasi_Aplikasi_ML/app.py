import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:
    CATBOOST_OK = False
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Prediksi Penyakit — 5 Model Klasifikasi",
    page_icon="🏥",
    layout="wide"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .header-box {
        background: linear-gradient(135deg, #7b2ff7 0%, #4a00e0 100%);
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 30px;
        color: white;
    }
    .header-box h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .header-box p  { font-size: 1rem; margin: 5px 0 0; opacity: 0.85; }

    .header-ridge {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        border-radius: 16px;
        padding: 30px 40px;
        margin-bottom: 30px;
        color: white;
    }
    .header-ridge h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .header-ridge p  { font-size: 1rem; margin: 5px 0 0; opacity: 0.85; }

    .empty-state {
        background: #1e2130;
        border-radius: 20px;
        padding: 80px 40px;
        text-align: center;
        border: 2px dashed #7b2ff7;
        margin: 40px 0;
    }
    .empty-state h2 { color: #a855f7; font-size: 1.8rem; font-weight: 800; }
    .empty-state p  { color: #aaa; font-size: 1rem; margin-top: 10px; }

    .step-box {
        background: #1e2130;
        border-radius: 14px;
        padding: 20px 25px;
        border-left: 5px solid #7b2ff7;
        margin: 10px 0;
    }
    .step-box h4 { color: #a855f7; margin: 0 0 6px; font-size: 1rem; }
    .step-box p  { color: #ccc; margin: 0; font-size: 0.9rem; }

    .metric-card {
        background: #1e2130;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        border-left: 5px solid;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    .metric-card .label { font-size: 0.85rem; color: #aaa; font-weight: 600; text-transform: uppercase; }
    .metric-card .value { font-size: 2rem; font-weight: 800; margin: 5px 0; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #a855f7;
        border-left: 4px solid #7b2ff7;
        padding-left: 12px;
        margin: 25px 0 15px;
    }
    .section-title-blue {
        font-size: 1.3rem;
        font-weight: 700;
        color: #64b5f6;
        border-left: 4px solid #1565c0;
        padding-left: 12px;
        margin: 25px 0 15px;
    }

    .info-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        border: 1px solid #2d3250;
        color: #ccc;
        line-height: 1.8;
    }

    .pkl-box {
        background: #1a1d2e;
        border-radius: 12px;
        padding: 18px 22px;
        border: 2px solid #7b2ff7;
        color: #a855f7;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 10px 0;
    }
    .pkl-box-blue {
        background: #1a1d2e;
        border-radius: 12px;
        padding: 18px 22px;
        border: 2px solid #1565c0;
        color: #64b5f6;
        font-family: monospace;
        font-size: 0.9rem;
        margin: 10px 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #7b2ff7, #4a00e0);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        cursor: pointer;
    }

    .btn-switcher > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1) !important;
    }

    .pred-box {
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 15px;
    }
    .pred-sakit    { background: #3d0000; border: 2px solid #e63946; color: #ff6b6b; }
    .pred-sehat    { background: #003d1a; border: 2px solid #2ecc71; color: #2ecc71; }
    .pred-diabetes { background: #3d1a00; border: 2px solid #ff6b00; color: #ffaa55; }
    .pred-normal   { background: #003d1a; border: 2px solid #2ecc71; color: #2ecc71; }

    .model-badge-gnb   { display:inline-block; background:#7b2ff7; color:white; border-radius:8px; padding:4px 14px; font-weight:700; font-size:0.9rem; }
    .model-badge-ridge { display:inline-block; background:#1565c0; color:white; border-radius:8px; padding:4px 14px; font-weight:700; font-size:0.9rem; }

    .header-teal {
        background: linear-gradient(135deg, #00695c 0%, #004d40 100%);
        border-radius: 16px; padding: 30px 40px; margin-bottom: 30px; color: white;
    }
    .header-teal h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .header-teal p  { font-size: 1rem; margin: 5px 0 0; opacity: 0.85; }

    .header-green {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        border-radius: 16px; padding: 30px 40px; margin-bottom: 30px; color: white;
    }
    .header-green h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .header-green p  { font-size: 1rem; margin: 5px 0 0; opacity: 0.85; }

    .header-orange {
        background: linear-gradient(135deg, #e65100 0%, #bf360c 100%);
        border-radius: 16px; padding: 30px 40px; margin-bottom: 30px; color: white;
    }
    .header-orange h1 { font-size: 2.2rem; font-weight: 800; margin: 0; }
    .header-orange p  { font-size: 1rem; margin: 5px 0 0; opacity: 0.85; }

    .section-title-teal {
        font-size: 1.3rem; font-weight: 700; color: #4db6ac;
        border-left: 4px solid #00695c; padding-left: 12px; margin: 25px 0 15px;
    }
    .section-title-green {
        font-size: 1.3rem; font-weight: 700; color: #81c784;
        border-left: 4px solid #2e7d32; padding-left: 12px; margin: 25px 0 15px;
    }
    .section-title-orange {
        font-size: 1.3rem; font-weight: 700; color: #ffb74d;
        border-left: 4px solid #e65100; padding-left: 12px; margin: 25px 0 15px;
    }
    .pkl-box-teal {
        background: #1a1d2e; border-radius: 12px; padding: 18px 22px;
        border: 2px solid #00695c; color: #4db6ac; font-family: monospace; font-size: 0.9rem; margin: 10px 0;
    }
    .pkl-box-green {
        background: #1a1d2e; border-radius: 12px; padding: 18px 22px;
        border: 2px solid #2e7d32; color: #81c784; font-family: monospace; font-size: 0.9rem; margin: 10px 0;
    }
    .pkl-box-orange {
        background: #1a1d2e; border-radius: 12px; padding: 18px 22px;
        border: 2px solid #e65100; color: #ffb74d; font-family: monospace; font-size: 0.9rem; margin: 10px 0;
    }
    .pred-thalassemia { background: #003d35; border: 2px solid #4db6ac; color: #4db6ac; }
    .pred-diabetes2   { background: #3d1a00; border: 2px solid #ff6b00; color: #ffaa55; }
    .pred-ginjal      { background: #3d0a00; border: 2px solid #ff7043; color: #ff7043; }
    .pred-ginjal-ok   { background: #003d1a; border: 2px solid #2ecc71; color: #2ecc71; }

    div[data-testid="stSidebar"] { background: #1a1d2e; }
</style>
""", unsafe_allow_html=True)

# ================================
# SESSION STATE — aktif model
# ================================
if "active_model" not in st.session_state:
    st.session_state.active_model = "GNB"

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")

    st.markdown("### 🔀 Pilih Model Aktif")
    sb1, sb2 = st.columns(2)
    with sb1:
        if st.button("💜 GNB\n(Samuel)", use_container_width=True):
            st.session_state.active_model = "GNB"
        if st.button("🩵 CatBoost\n(Azka)", use_container_width=True):
            st.session_state.active_model = "CatBoost"
    with sb2:
        if st.button("💙 Ridge\n(Rofi)", use_container_width=True):
            st.session_state.active_model = "Ridge"
        if st.button("💚 QDA\n(Bian)", use_container_width=True):
            st.session_state.active_model = "QDA"
    if st.button("🧡 Bagging\n(Rifki)", use_container_width=True):
        st.session_state.active_model = "Bagging"

    active = st.session_state.active_model
    active_label = {
        "GNB": "💜 Gaussian Naive Bayes",
        "Ridge": "💙 Ridge Classifier",
        "CatBoost": "🩵 CatBoost Classifier",
        "QDA": "💚 QDA",
        "Bagging": "🧡 Bagging Classifier",
    }.get(active, active)
    st.markdown(f"**Model Aktif:** {active_label}")
    st.markdown("---")

    if active == "GNB":
        st.markdown("#### 📂 Upload (GNB — Heart Disease)")
        uploaded_file = st.file_uploader("Dataset CSV (Heart)", type=["csv"], key="gnb_csv")
        uploaded_pkl  = st.file_uploader("Model PKL (GNB)",     type=["pkl"], key="gnb_pkl")
        sep_option = st.selectbox("Separator CSV", [",", ";"], index=0, key="gnb_sep")
        test_size  = st.slider("Ukuran Data Test (%)", 10, 40, 20, step=5, key="gnb_ts")
        st.markdown("---")
        st.markdown(f"- Train: **{100-test_size}%** | Test: **{test_size}%**")
    elif active == "Ridge":
        st.markdown("#### 📂 Upload (Ridge — Diabetes)")
        uploaded_file_r = st.file_uploader("Dataset CSV (Diabetes)", type=["csv"], key="ridge_csv")
        uploaded_pkl_r  = st.file_uploader("Model PKL (Ridge)",      type=["pkl"], key="ridge_pkl")
        sep_option_r = st.selectbox("Separator CSV", [",", ";"], index=0, key="ridge_sep")
        test_size_r  = st.slider("Ukuran Data Test (%)", 10, 40, 20, step=5, key="ridge_ts")
        st.markdown("---")
        st.markdown(f"- Train: **{100-test_size_r}%** | Test: **{test_size_r}%**")
    elif active == "CatBoost":
        st.markdown("#### 📂 Upload (CatBoost — Thalassemia)")
        uploaded_file_cb = st.file_uploader("Dataset XLSX (Thalassemia)", type=["xlsx"], key="cb_xlsx")
        uploaded_pkl_cb  = st.file_uploader("Model PKL (CatBoost)",       type=["pkl"], key="cb_pkl")
        test_size_cb = st.slider("Ukuran Data Test (%)", 10, 40, 20, step=5, key="cb_ts")
        st.markdown("---")
        st.markdown(f"- Train: **{100-test_size_cb}%** | Test: **{test_size_cb}%**")
    elif active == "QDA":
        st.markdown("#### 📂 Upload (QDA — Gejala Diabetes)")
        uploaded_file_qda = st.file_uploader("Dataset CSV (Diabetes Symptoms)", type=["csv"], key="qda_csv")
        uploaded_pkl_qda  = st.file_uploader("Model PKL (QDA)",                 type=["pkl"], key="qda_pkl")
        sep_option_qda = st.selectbox("Separator CSV", [",", ";"], index=0, key="qda_sep")
        test_size_qda  = st.slider("Ukuran Data Test (%)", 10, 40, 20, step=5, key="qda_ts")
        st.markdown("---")
        st.markdown(f"- Train: **{100-test_size_qda}%** | Test: **{test_size_qda}%**")
    elif active == "Bagging":
        st.markdown("#### 📂 Upload (Bagging — Penyakit Ginjal)")
        uploaded_file_bg = st.file_uploader("Dataset CSV (Ginjal)", type=["csv"], key="bg_csv")
        uploaded_pkl_bg  = st.file_uploader("Model PKL (Bagging)", type=["pkl"], key="bg_pkl")
        sep_option_bg = st.selectbox("Separator CSV", [",", ";"], index=0, key="bg_sep")
        test_size_bg  = st.slider("Ukuran Data Test (%)", 10, 40, 20, step=5, key="bg_ts")
        st.markdown("---")
        st.markdown(f"- Train: **{100-test_size_bg}%** | Test: **{test_size_bg}%**")

# ================================
# ============================================================
#   BAGIAN GNB — Samuel Tambunan
# ============================================================
# ================================
if st.session_state.active_model == "GNB":

    # HEADER
    st.markdown("""
    <div class="header-box">
        <h1>🫀 Prediksi Penyakit Jantung</h1>
        <p>Metode: Klasifikasi &nbsp;|&nbsp; Model: Gaussian Naive Bayes &nbsp;|&nbsp; Dataset: Heart Disease UCI &nbsp;|&nbsp; Oleh: Samuel Tambunan</p>
    </div>
    """, unsafe_allow_html=True)

    # EMPTY STATE
    if uploaded_file is None:
        st.markdown("""
        <div class="empty-state">
            <h2>📂 Belum Ada Dataset</h2>
            <p>Upload file CSV dan model PKL melalui sidebar kiri untuk memulai analisis</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="step-box"><h4>① Upload Dataset CSV</h4><p>Upload file heart_preprocessed.csv melalui sidebar</p></div>
            <div class="step-box"><h4>② Upload Model PKL</h4><p>Upload file gnb_model.pkl yang sudah ditraining</p></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step-box"><h4>③ Lihat Hasil Analisis</h4><p>Semua tab otomatis menampilkan data, preprocessing, training & evaluasi</p></div>
            <div class="step-box"><h4>④ Coba Prediksi</h4><p>Masukkan data pasien baru di tab Prediksi untuk hasil diagnosis</p></div>
            """, unsafe_allow_html=True)
        st.stop()

    # LOAD DATA
    df = pd.read_csv(uploaded_file, sep=sep_option)
    df = df.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    st.success(f"✅ Dataset berhasil dimuat! **{df.shape[0]} baris** dan **{df.shape[1]} kolom**")

    # LOAD MODEL
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )

    if uploaded_pkl is not None:
        model_data = pickle.load(uploaded_pkl)
        gnb_model = model_data['model'] if isinstance(model_data, dict) else model_data
        st.success("✅ Model **gnb_model.pkl** berhasil dimuat!")
        model_source = "pkl"
    else:
        st.warning("⚠️ PKL belum diupload. Model ditraining ulang dari data.")
        gnb_model = GaussianNB(var_smoothing=1e-9)
        gnb_model.fit(X_train, y_train)
        model_source = "retrain"

    y_pred = gnb_model.predict(X_test)
    acc  = accuracy_score(y_test,  y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test,    y_pred, zero_division=0)
    f1   = f1_score(y_test,        y_pred, zero_division=0)

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📂 Data", "🔧 Preprocessing", "📐 Training GNB", "📊 Evaluasi", "🔮 Prediksi"
    ])

    # --- TAB 1: DATA ---
    with tab1:
        st.markdown('<div class="section-title">📂 Eksplorasi Dataset</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Data",     f"{df.shape[0]} baris")
        c2.metric("📊 Total Kolom",    f"{df.shape[1]} kolom")
        c3.metric("🎯 Fitur Input",    f"{df.shape[1]-1} fitur")
        c4.metric("❓ Missing Values", "0 ✅")

        st.markdown('<div class="section-title">👀 Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">📈 Statistik Deskriptif</div>', unsafe_allow_html=True)
            st.dataframe(df.describe().round(3), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title">🎯 Distribusi Target</div>', unsafe_allow_html=True)
            tc = df['target'].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Tidak Sakit (0)', 'Sakit Jantung (1)'],
                          [tc.get(0,0), tc.get(1,0)],
                          color=['#7b2ff7','#a855f7'], edgecolor='white', linewidth=1.2)
            for bar, val in zip(bars, [tc.get(0,0), tc.get(1,0)]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        str(val), ha='center', color='white', fontweight='bold')
            ax.set_title('Distribusi Target', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#444')
            st.pyplot(fig)

        st.markdown('<div class="section-title">🔥 Heatmap Korelasi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 7))
        fig.patch.set_facecolor('#1e2130')
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                    linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title('Korelasi Antar Fitur', color='white', fontweight='bold', fontsize=13)
        ax.tick_params(colors='white')
        st.pyplot(fig)

    # --- TAB 2: PREPROCESSING ---
    with tab2:
        st.markdown('<div class="section-title">🔧 Informasi Preprocessing</div>', unsafe_allow_html=True)
        st.info("ℹ️ Dataset **heart_preprocessed.csv** sudah melalui proses preprocessing sebelumnya.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-box">
                <b>✅ Sudah dilakukan:</b><br>
                • Normalisasi/Standarisasi fitur numerik<br>
                • One-Hot Encoding fitur kategorik (cp, thal, slope)<br>
                • Penghapusan missing values<br>
                • Konversi tipe data boolean ke integer
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box">
                <b>📋 Detail Fitur:</b><br>
                • Fitur numerik : age, trestbps, chol, thalach, oldpeak<br>
                • Fitur biner   : sex, fbs, restecg, exang, ca<br>
                • One-Hot       : cp (4), thal (4), slope (3)<br>
                • Target        : 0 = Tidak Sakit, 1 = Sakit Jantung
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">✂️ Split Data Train & Test</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Total Data",  f"{len(df)}")
        c2.metric("🏋️ Data Train", f"{len(X_train)} ({100-test_size}%)")
        c3.metric("🧪 Data Test",  f"{len(X_test)} ({test_size}%)")
        st.dataframe(pd.DataFrame(X_train, columns=X.columns).head(), use_container_width=True)
        st.success("✅ Data siap digunakan untuk Gaussian Naive Bayes!")

    # --- TAB 3: TRAINING ---
    with tab3:
        st.markdown('<div class="section-title">📐 Model Gaussian Naive Bayes</div>', unsafe_allow_html=True)
        if model_source == "pkl":
            st.markdown('<div class="pkl-box">✅ Model dimuat dari file: <b>gnb_model.pkl</b><br>🔒 Model tidak ditraining ulang</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pkl-box">⚠️ Model ditraining ulang karena PKL belum diupload<br>💡 Upload gnb_model.pkl di sidebar</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("📐 Model",         "Gaussian Naive Bayes")
        c2.metric("🔧 Var Smoothing",  "1e-9")
        c3.metric("🎯 Akurasi",        f"{acc*100:.2f}%")

        st.markdown("""
        <div class="info-box">
            <b>📖 Cara Kerja Gaussian Naive Bayes:</b><br><br>
            1. Menghitung <b>probabilitas prior</b> setiap kelas (Sakit / Tidak Sakit)<br>
            2. Menghitung <b>mean & variance</b> setiap fitur untuk tiap kelas<br>
            3. Mengasumsikan setiap fitur mengikuti <b>distribusi Gaussian (Normal)</b><br>
            4. Prediksi menggunakan <b>Teorema Bayes</b> → pilih kelas dengan probabilitas tertinggi
        </div>""", unsafe_allow_html=True)

        # Distribusi Gaussian
        st.markdown('<div class="section-title">📊 Distribusi Gaussian Per Kelas</div>', unsafe_allow_html=True)
        num_features = [f for f in ['age','trestbps','chol','thalach','oldpeak'] if f in X.columns]
        fig, axes = plt.subplots(1, len(num_features), figsize=(14, 4))
        fig.patch.set_facecolor('#1e2130')
        for i, feat in enumerate(num_features):
            ax = axes[i]; ax.set_facecolor('#1e2130')
            for label, color, name in [(0,'#7b2ff7','Tidak Sakit'),(1,'#f1c40f','Sakit')]:
                data = X_train[X_train.index.isin(y_train[y_train==label].index)][feat]
                ax.hist(data, bins=15, alpha=0.6, color=color, label=name, edgecolor='none')
            ax.set_title(feat, color='white', fontsize=10, fontweight='bold')
            ax.tick_params(colors='white', labelsize=8)
            ax.legend(fontsize=7, facecolor='#2d3250', labelcolor='white')
            for spine in ax.spines.values(): spine.set_color('#444')
        fig.suptitle('Distribusi Fitur per Kelas', color='white', fontweight='bold', fontsize=13)
        plt.tight_layout(); st.pyplot(fig)

        # Prior
        st.markdown('<div class="section-title">📋 Probabilitas Prior</div>', unsafe_allow_html=True)
        prior_df = pd.DataFrame({
            'Kelas':              ['Tidak Sakit (0)', 'Sakit Jantung (1)'],
            'Jumlah Data Train':  [sum(y_train==0), sum(y_train==1)],
            'Probabilitas Prior': [f"{gnb_model.class_prior_[0]*100:.2f}%",
                                   f"{gnb_model.class_prior_[1]*100:.2f}%"]
        })
        st.dataframe(prior_df, use_container_width=True)

    # --- TAB 4: EVALUASI ---
    with tab4:
        st.markdown('<div class="section-title">📊 Hasil Evaluasi Gaussian Naive Bayes</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1,"Accuracy",acc,"#a855f7"),(c2,"Precision",prec,"#3498db"),
            (c3,"Recall",rec,"#f39c12"),(c4,"F1-Score",f1,"#e63946")]:
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130')
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                        xticklabels=['Tidak Sakit','Sakit Jantung'],
                        yticklabels=['Tidak Sakit','Sakit Jantung'],
                        linewidths=1, linecolor='#333', ax=ax,
                        annot_kws={"size":16,"weight":"bold","color":"white"})
            ax.set_title('Confusion Matrix - Gaussian NB', color='white', fontweight='bold')
            ax.set_xlabel('Prediksi', color='white'); ax.set_ylabel('Aktual', color='white')
            ax.tick_params(colors='white'); st.pyplot(fig)

        with c2:
            st.markdown('<div class="section-title">📈 Grafik Metrik Evaluasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Accuracy','Precision','Recall','F1-Score'],
                          [acc,prec,rec,f1],
                          color=['#a855f7','#3498db','#f39c12','#e63946'],
                          edgecolor='white', linewidth=1, width=0.5)
            for bar, val in zip(bars, [acc,prec,rec,f1]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val*100:.2f}%', ha='center', color='white', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_title('Metrik Evaluasi - GNB', color='white', fontweight='bold')
            ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2)
            for spine in ax.spines.values(): spine.set_color('#444')
            st.pyplot(fig)

        # Error analysis
        st.markdown('<div class="section-title">❌ Analisis Kesalahan Model</div>', unsafe_allow_html=True)
        tn, fp, fn, tp = cm.ravel()
        error_rate = 1 - acc
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate*100:.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#27ae60"><div class="label">Total Benar</div><div class="value" style="color:#27ae60">{tp+tn}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <b>📌 Interpretasi Kesalahan:</b><br>
            • <b>False Positive (FP = {fp})</b>: Model memprediksi <i>Sakit</i> padahal sebenarnya <i>Tidak Sakit</i> → over-diagnosis<br>
            • <b>False Negative (FN = {fn})</b>: Model memprediksi <i>Tidak Sakit</i> padahal sebenarnya <i>Sakit</i> → under-diagnosis (lebih berbahaya)<br>
            • <b>Error Rate = {error_rate*100:.2f}%</b> → dari {len(y_test)} data test, {fp+fn} prediksi salah
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred,
                                       target_names=['Tidak Sakit (0)','Sakit Jantung (1)'],
                                       output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        st.markdown('<div class="section-title">📝 Kesimpulan</div>', unsafe_allow_html=True)
        grade, color = ("BAIK 🟢","#2ecc71") if acc>=0.85 else ("CUKUP 🟡","#f39c12") if acc>=0.70 else ("KURANG 🔴","#e63946")
        st.markdown(f"""
        <div class="info-box">
            <b>Nama</b>        : Samuel Tambunan<br>
            <b>Dataset</b>     : Heart Disease UCI (Preprocessed) &nbsp;|&nbsp;
            <b>Total Data</b>  : {len(df)} baris &nbsp;|&nbsp;
            <b>Fitur</b>       : {df.shape[1]-1} fitur<br><br>
            <b>Model</b>       : Gaussian Naive Bayes &nbsp;|&nbsp;
            <b>Sumber</b>      : {'📦 gnb_model.pkl' if model_source=='pkl' else '🔄 Retrain'} &nbsp;|&nbsp;
            <b>Split</b>       : {100-test_size}% Train / {test_size}% Test<br><br>
            <b>Accuracy</b>    : <span style="color:{color}; font-weight:800">{acc*100:.2f}%</span> &nbsp;→&nbsp;
            Performa: <span style="color:{color}; font-weight:800">{grade}</span>
        </div>""", unsafe_allow_html=True)

    # --- TAB 5: PREDIKSI ---
    with tab5:
        st.markdown('<div class="section-title">🔮 Prediksi Data Pasien Baru — GNB (Heart Disease)</div>', unsafe_allow_html=True)
        st.markdown("Masukkan data klinis pasien (nilai asli) untuk memprediksi apakah terkena penyakit jantung atau tidak.")

        # Scaling params dari UCI Heart Disease original dataset
        # Fitur model: age, sex, cp, trestbps, thalach, exang, oldpeak, ca (8 fitur)
        HEART_SCALE = {
            'age':      (54.4,  9.08),
            'trestbps': (131.7, 17.6),
            'thalach':  (149.6, 22.9),
            'oldpeak':  (1.04,  1.16),
        }

        c1, c2 = st.columns(2)
        with c1:
            age_raw      = st.number_input("Usia (age, tahun)",                  min_value=20,  max_value=80,  value=54,  step=1,   key="gnb_age")
            sex          = st.selectbox("Jenis Kelamin (sex)",                   [1, 0], format_func=lambda x: "Laki-laki (1)" if x==1 else "Perempuan (0)", key="gnb_sex")
            cp           = st.selectbox("Tipe Nyeri Dada (cp)",                  [0, 1, 2, 3],
                                        format_func=lambda x: {0:"0 - Typical Angina", 1:"1 - Atypical Angina", 2:"2 - Non-anginal Pain", 3:"3 - Asymptomatic"}[x],
                                        key="gnb_cp")
            trestbps_raw = st.number_input("Tekanan Darah Istirahat (trestbps, mmHg)", min_value=80, max_value=210, value=130, step=1, key="gnb_trestbps")
        with c2:
            thalach_raw  = st.number_input("Detak Jantung Maks (thalach, bpm)", min_value=60,  max_value=210, value=150, step=1,   key="gnb_thalach")
            exang        = st.selectbox("Angina saat Olahraga (exang)",         [0, 1], format_func=lambda x: "Tidak (0)" if x==0 else "Ya (1)", key="gnb_exang")
            oldpeak_raw  = st.number_input("Depresi ST (oldpeak)",              min_value=0.0, max_value=7.0, value=1.0, step=0.1, key="gnb_oldpeak")
            ca           = st.selectbox("Jumlah Pembuluh Besar (ca)",           [0, 1, 2, 3], key="gnb_ca")

        if st.button("🔮 Prediksi Sekarang! (GNB)", key="gnb_predict"):
            # Transformasi nilai asli ke scaled (StandardScaler)
            age_sc      = (age_raw      - HEART_SCALE['age'][0])      / HEART_SCALE['age'][1]
            trestbps_sc = (trestbps_raw - HEART_SCALE['trestbps'][0]) / HEART_SCALE['trestbps'][1]
            thalach_sc  = (thalach_raw  - HEART_SCALE['thalach'][0])  / HEART_SCALE['thalach'][1]
            oldpeak_sc  = (oldpeak_raw  - HEART_SCALE['oldpeak'][0])  / HEART_SCALE['oldpeak'][1]

            # 8 fitur sesuai dataset: age, sex, cp, trestbps, thalach, exang, oldpeak, ca
            input_data = np.array([[age_sc, sex, cp, trestbps_sc, thalach_sc, exang, oldpeak_sc, ca]])
            prediction = gnb_model.predict(input_data)[0]
            proba      = gnb_model.predict_proba(input_data)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown('<div class="pred-box pred-sakit">❤️‍🔥 TERDETEKSI SAKIT JANTUNG</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-box pred-sehat">💚 TIDAK TERDETEKSI SAKIT JANTUNG</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Probabilitas Tidak Sakit",   f"{proba[0]*100:.2f}%")
            c2.metric("Probabilitas Sakit Jantung", f"{proba[1]*100:.2f}%")

            # Akurasi & kesalahan model di bawah hasil prediksi
            st.markdown("---")
            st.markdown('<div class="section-title">📊 Performa Model pada Data Test</div>', unsafe_allow_html=True)
            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.markdown(f'<div class="metric-card" style="border-left-color:#a855f7"><div class="label">Akurasi Model</div><div class="value" style="color:#a855f7">{acc*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca2.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{(1-acc)*100:.2f}%</div></div>', unsafe_allow_html=True)
            cm_t = confusion_matrix(y_test, y_pred); tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
            ca3.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_t}</div></div>', unsafe_allow_html=True)
            ca4.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_t}</div></div>', unsafe_allow_html=True)


# ================================
# ============================================================
#   BAGIAN RIDGE — Saya
# ============================================================
# ================================
elif st.session_state.active_model == "Ridge":

    # HEADER
    st.markdown("""
    <div class="header-ridge">
        <h1>🩺 Prediksi Diabetes</h1>
        <p>Metode: Klasifikasi &nbsp;|&nbsp; Model: Ridge Classifier &nbsp;|&nbsp; Dataset: Pima Indians Diabetes &nbsp;|&nbsp; Preprocessing: SMOTE + StandardScaler &nbsp;|&nbsp; Oleh: Muh.Rofi Azmi</p>
    </div>
    """, unsafe_allow_html=True)

    # EMPTY STATE
    if uploaded_file_r is None:
        st.markdown("""
        <div class="empty-state">
            <h2>📂 Belum Ada Dataset</h2>
            <p>Upload file CSV diabetes dan model PKL Ridge melalui sidebar kiri</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="step-box"><h4>① Upload Dataset CSV</h4><p>Upload file diabetes_preprocessed_smote.csv melalui sidebar</p></div>
            <div class="step-box"><h4>② Upload Model PKL</h4><p>Upload file ridge_classifier_model.pkl yang sudah ditraining</p></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step-box"><h4>③ Lihat Hasil Analisis</h4><p>Semua tab menampilkan data, preprocessing, training & evaluasi Ridge</p></div>
            <div class="step-box"><h4>④ Coba Prediksi Diabetes</h4><p>Masukkan data pasien baru di tab Prediksi untuk hasil diagnosis</p></div>
            """, unsafe_allow_html=True)
        st.stop()

    # LOAD DATA
    df_r = pd.read_csv(uploaded_file_r, sep=sep_option_r)
    df_r = df_r.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    st.success(f"✅ Dataset Diabetes berhasil dimuat! **{df_r.shape[0]} baris** dan **{df_r.shape[1]} kolom**")

    # FEATURES
    target_col = 'Outcome'
    X_r = df_r.drop(target_col, axis=1)
    y_r = df_r[target_col]
    feature_cols = list(X_r.columns)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_r, y_r, test_size=test_size_r/100, random_state=42, stratify=y_r
    )

    # LOAD MODEL RIDGE
    ridge_model = None
    model_source_r = "retrain"

    if uploaded_pkl_r is not None:
        try:
            try:
                model_data_r = joblib.load(uploaded_pkl_r)
            except Exception:
                uploaded_pkl_r.seek(0)
                model_data_r = pickle.load(uploaded_pkl_r)

            if isinstance(model_data_r, dict):
                ridge_model = model_data_r.get('model', model_data_r)
            else:
                ridge_model = model_data_r
            st.success("✅ Model **ridge_classifier_model.pkl** berhasil dimuat!")
            model_source_r = "pkl"
        except Exception as e:
            st.warning(f"⚠️ Gagal memuat PKL ({e}). Model ditraining ulang.")

    if ridge_model is None:
        if model_source_r != "pkl":
            st.warning("⚠️ PKL belum diupload atau gagal dimuat. Model Ridge ditraining ulang dari data.")
        ridge_model = RidgeClassifier(alpha=1.0)
        ridge_model.fit(X_train_r, y_train_r)
        model_source_r = "retrain"

    y_pred_r = ridge_model.predict(X_test_r)
    acc_r  = accuracy_score(y_test_r,  y_pred_r)
    prec_r = precision_score(y_test_r, y_pred_r, zero_division=0)
    rec_r  = recall_score(y_test_r,    y_pred_r, zero_division=0)
    f1_r   = f1_score(y_test_r,        y_pred_r, zero_division=0)

    # TABS
    rtab1, rtab2, rtab3, rtab4, rtab5 = st.tabs([
        "📂 Data", "🔧 Preprocessing", "📐 Training Ridge", "📊 Evaluasi", "🔮 Prediksi"
    ])

    # --- TAB 1: DATA ---
    with rtab1:
        st.markdown('<div class="section-title-blue">📂 Eksplorasi Dataset Diabetes</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📋 Total Data",     f"{df_r.shape[0]} baris")
        c2.metric("📊 Total Kolom",    f"{df_r.shape[1]} kolom")
        c3.metric("🎯 Fitur Input",    f"{df_r.shape[1]-1} fitur")
        c4.metric("❓ Missing Values", "0 ✅")

        st.markdown('<div class="section-title-blue">👀 Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(df_r.head(10), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-blue">📈 Statistik Deskriptif</div>', unsafe_allow_html=True)
            st.dataframe(df_r.describe().round(3), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title-blue">🎯 Distribusi Target (Outcome)</div>', unsafe_allow_html=True)
            tc_r = df_r[target_col].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Non-Diabetes (0)', 'Diabetes (1)'],
                          [tc_r.get(0,0), tc_r.get(1,0)],
                          color=['#1565c0','#64b5f6'], edgecolor='white', linewidth=1.2)
            for bar, val in zip(bars, [tc_r.get(0,0), tc_r.get(1,0)]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        str(val), ha='center', color='white', fontweight='bold')
            ax.set_title('Distribusi Target — Outcome', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#444')
            st.pyplot(fig)

        st.markdown('<div class="section-title-blue">🔥 Heatmap Korelasi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e2130')
        sns.heatmap(df_r.corr(), annot=True, fmt='.2f', cmap='Blues',
                    linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title('Korelasi Antar Fitur — Diabetes', color='white', fontweight='bold', fontsize=13)
        ax.tick_params(colors='white')
        st.pyplot(fig)

    # --- TAB 2: PREPROCESSING ---
    with rtab2:
        st.markdown('<div class="section-title-blue">🔧 Preprocessing Pipeline — Diabetes</div>', unsafe_allow_html=True)
        st.info("ℹ️ Dataset sudah melalui preprocessing lengkap sebelum diupload.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-box">
                <b>✅ Pipeline Preprocessing:</b><br>
                1. Replace nilai 0 tidak valid (Glucose, BloodPressure, SkinThickness, Insulin, BMI) → NaN<br>
                2. Median Imputation — isi NaN dengan nilai median kolom<br>
                3. Outlier Handling — capping IQR method (winsorization)<br>
                4. StandardScaler — normalisasi mean=0, std=1<br>
                5. SMOTE — oversample kelas minoritas → balance 50:50
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box">
                <b>📋 Detail Fitur (8 fitur):</b><br>
                • <b>Pregnancies</b>       : Jumlah kehamilan<br>
                • <b>Glucose</b>           : Konsentrasi glukosa plasma<br>
                • <b>BloodPressure</b>     : Tekanan darah diastolik<br>
                • <b>SkinThickness</b>     : Ketebalan lipatan kulit trisep<br>
                • <b>Insulin</b>           : Serum insulin 2 jam<br>
                • <b>BMI</b>              : Indeks massa tubuh<br>
                • <b>DiabetesPedigreeFunction</b>: Fungsi silsilah diabetes<br>
                • <b>Age</b>              : Usia (tahun)<br>
                • <b>Target → Outcome</b>  : 0 = Non-Diabetes, 1 = Diabetes
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-blue">✂️ Split Data Train & Test</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Total Data",  f"{len(df_r)}")
        c2.metric("🏋️ Data Train", f"{len(X_train_r)} ({100-test_size_r}%)")
        c3.metric("🧪 Data Test",  f"{len(X_test_r)} ({test_size_r}%)")
        st.dataframe(X_train_r.head(), use_container_width=True)

        # Distribusi fitur setelah scaling
        st.markdown('<div class="section-title-blue">📊 Distribusi Fitur (Setelah Preprocessing)</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        fig.patch.set_facecolor('#1e2130')
        axes = axes.flatten()
        for i, col in enumerate(feature_cols):
            axes[i].set_facecolor('#1e2130')
            axes[i].hist(df_r[col], bins=25, color='#1565c0', edgecolor='white', alpha=0.85)
            axes[i].set_title(col, color='white', fontweight='bold', fontsize=9)
            axes[i].tick_params(colors='white', labelsize=7)
            for spine in axes[i].spines.values(): spine.set_color('#444')
        fig.suptitle('Distribusi Fitur Diabetes — Setelah Preprocessing & SMOTE', color='white', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)
        st.success("✅ Data siap digunakan untuk Ridge Classifier!")

    # --- TAB 3: TRAINING ---
    with rtab3:
        st.markdown('<div class="section-title-blue">📐 Model Ridge Classifier</div>', unsafe_allow_html=True)
        if model_source_r == "pkl":
            st.markdown('<div class="pkl-box-blue">✅ Model dimuat dari file: <b>ridge_classifier_model.pkl</b><br>🔒 Model tidak ditraining ulang</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pkl-box-blue">⚠️ Model ditraining ulang karena PKL belum diupload / gagal dimuat<br>💡 Upload ridge_classifier_model.pkl di sidebar</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("📐 Model",    "Ridge Classifier")
        c2.metric("🔧 Alpha",   "1.0 (regularisasi L2)")
        c3.metric("🎯 Akurasi", f"{acc_r*100:.2f}%")

        st.markdown("""
        <div class="info-box">
            <b>📖 Cara Kerja Ridge Classifier:</b><br><br>
            1. Ridge Classifier adalah adaptasi <b>Ridge Regression</b> untuk klasifikasi biner<br>
            2. Menggunakan <b>regularisasi L2</b> → meminimalkan |β|² untuk mencegah overfitting<br>
            3. Fungsi loss: <b>min (||Xβ - y||² + α||β||²)</b><br>
            4. Parameter <b>α (alpha)</b> mengontrol kekuatan regularisasi — semakin besar, semakin smooth<br>
            5. Prediksi: sign(Xβ) → threshold 0 → kelas 0 atau 1
        </div>""", unsafe_allow_html=True)

        # Koefisien model Ridge
        st.markdown('<div class="section-title-blue">📊 Koefisien Ridge Classifier per Fitur</div>', unsafe_allow_html=True)
        coef = ridge_model.coef_[0] if hasattr(ridge_model, 'coef_') else np.zeros(len(feature_cols))
        coef_df = pd.DataFrame({'Fitur': feature_cols, 'Koefisien': coef}).sort_values('Koefisien', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        colors_bar = ['#e63946' if v < 0 else '#1565c0' for v in coef_df['Koefisien']]
        bars = ax.barh(coef_df['Fitur'], coef_df['Koefisien'], color=colors_bar, edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='white', linewidth=1, linestyle='--', alpha=0.5)
        ax.set_title('Koefisien Model Ridge per Fitur\n(merah=negatif, biru=positif)', color='white', fontweight='bold')
        ax.tick_params(colors='white', labelsize=9)
        ax.set_xlabel('Nilai Koefisien', color='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        plt.tight_layout(); st.pyplot(fig)

        st.dataframe(coef_df.set_index('Fitur').round(4), use_container_width=True)

    # --- TAB 4: EVALUASI ---
    with rtab4:
        st.markdown('<div class="section-title-blue">📊 Hasil Evaluasi Ridge Classifier</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, label, val, color in [
            (c1,"Accuracy",acc_r,"#64b5f6"),(c2,"Precision",prec_r,"#26c6da"),
            (c3,"Recall",rec_r,"#f39c12"),(c4,"F1-Score",f1_r,"#e63946")]:
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-blue">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
            cm_r = confusion_matrix(y_test_r, y_pred_r)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130')
            sns.heatmap(cm_r, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Non-Diabetes','Diabetes'],
                        yticklabels=['Non-Diabetes','Diabetes'],
                        linewidths=1, linecolor='#333', ax=ax,
                        annot_kws={"size":16,"weight":"bold","color":"white"})
            ax.set_title('Confusion Matrix - Ridge Classifier', color='white', fontweight='bold')
            ax.set_xlabel('Prediksi', color='white'); ax.set_ylabel('Aktual', color='white')
            ax.tick_params(colors='white'); st.pyplot(fig)

        with c2:
            st.markdown('<div class="section-title-blue">📈 Grafik Metrik Evaluasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Accuracy','Precision','Recall','F1-Score'],
                          [acc_r,prec_r,rec_r,f1_r],
                          color=['#64b5f6','#26c6da','#f39c12','#e63946'],
                          edgecolor='white', linewidth=1, width=0.5)
            for bar, val in zip(bars, [acc_r,prec_r,rec_r,f1_r]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val*100:.2f}%', ha='center', color='white', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_title('Metrik Evaluasi - Ridge', color='white', fontweight='bold')
            ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2)
            for spine in ax.spines.values(): spine.set_color('#444')
            st.pyplot(fig)

        # Error analysis
        st.markdown('<div class="section-title-blue">❌ Analisis Kesalahan Model</div>', unsafe_allow_html=True)
        tn_r, fp_r, fn_r, tp_r = cm_r.ravel()
        error_rate_r = 1 - acc_r
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_r*100:.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_r}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_r}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#27ae60"><div class="label">Total Benar</div><div class="value" style="color:#27ae60">{tp_r+tn_r}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <b>📌 Interpretasi Kesalahan:</b><br>
            • <b>False Positive (FP = {fp_r})</b>: Model memprediksi <i>Diabetes</i> padahal sebenarnya <i>Non-Diabetes</i> → over-diagnosis<br>
            • <b>False Negative (FN = {fn_r})</b>: Model memprediksi <i>Non-Diabetes</i> padahal sebenarnya <i>Diabetes</i> → miss-diagnosis (lebih berbahaya)<br>
            • <b>Error Rate = {error_rate_r*100:.2f}%</b> → dari {len(y_test_r)} data test, {fp_r+fn_r} prediksi salah
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-blue">📋 Classification Report</div>', unsafe_allow_html=True)
        report_r = classification_report(y_test_r, y_pred_r,
                                         target_names=['Non-Diabetes (0)','Diabetes (1)'],
                                         output_dict=True)
        st.dataframe(pd.DataFrame(report_r).transpose().round(4), use_container_width=True)

        st.markdown('<div class="section-title-blue">📝 Kesimpulan</div>', unsafe_allow_html=True)
        grade_r, color_r = ("BAIK 🟢","#2ecc71") if acc_r>=0.85 else ("CUKUP 🟡","#f39c12") if acc_r>=0.70 else ("KURANG 🔴","#e63946")
        st.markdown(f"""
        <div class="info-box">
            <b>Dataset</b>     : Pima Indians Diabetes &nbsp;|&nbsp;
            <b>Total Data</b>  : {len(df_r)} baris &nbsp;|&nbsp;
            <b>Fitur</b>       : {df_r.shape[1]-1} fitur<br><br>
            <b>Model</b>       : Ridge Classifier &nbsp;|&nbsp;
            <b>Sumber</b>      : {'📦 ridge_classifier_model.pkl' if model_source_r=='pkl' else '🔄 Retrain'} &nbsp;|&nbsp;
            <b>Split</b>       : {100-test_size_r}% Train / {test_size_r}% Test<br><br>
            <b>Accuracy</b>    : <span style="color:{color_r}; font-weight:800">{acc_r*100:.2f}%</span> &nbsp;→&nbsp;
            Performa: <span style="color:{color_r}; font-weight:800">{grade_r}</span>
        </div>""", unsafe_allow_html=True)

    # --- TAB 5: PREDIKSI ---
    with rtab5:
        st.markdown('<div class="section-title-blue">🔮 Prediksi Data Pasien Baru — Ridge Classifier (Diabetes)</div>', unsafe_allow_html=True)
        st.markdown("Masukkan data klinis pasien (nilai asli) untuk memprediksi apakah terkena diabetes atau tidak.")

        # Scaling params dari Pima Indians Diabetes original dataset (sebelum preprocessing)
        DIAB_SCALE = {
            'Pregnancies':              (3.84,  3.37),
            'Glucose':                  (121.7, 30.5),
            'BloodPressure':            (72.4,  12.1),
            'SkinThickness':            (29.1,  10.5),
            'Insulin':                  (155.5, 85.0),
            'BMI':                      (32.5,  6.9),
            'DiabetesPedigreeFunction': (0.47,  0.33),
            'Age':                      (33.2,  11.8),
        }

        c1, c2 = st.columns(2)
        with c1:
            pregnancies_raw = st.number_input("Pregnancies (Jumlah Kehamilan)",       min_value=0,    max_value=17,   value=3,    step=1,   key="r_preg",
                                              help="Jumlah kehamilan")
            glucose_raw     = st.number_input("Glucose (Glukosa Plasma, mg/dL)",      min_value=40,   max_value=200,  value=120,  step=1,   key="r_gluc",
                                              help="Konsentrasi glukosa plasma 2 jam setelah tes toleransi glukosa")
            bloodpressure_raw = st.number_input("BloodPressure (Tekanan Darah Diastolik, mmHg)", min_value=30, max_value=130, value=72, step=1, key="r_bp",
                                                help="Tekanan darah diastolik (mm Hg)")
            skinthickness_raw = st.number_input("SkinThickness (Ketebalan Kulit, mm)", min_value=5,  max_value=55,   value=29,   step=1,   key="r_skin",
                                                help="Ketebalan lipatan kulit trisep (mm)")
        with c2:
            insulin_raw = st.number_input("Insulin (Serum Insulin, µU/mL)",           min_value=10,   max_value=300,  value=80,   step=1,   key="r_ins",
                                          help="Serum insulin 2 jam (µU/mL)")
            bmi_raw     = st.number_input("BMI (Indeks Massa Tubuh, kg/m²)",           min_value=15.0, max_value=55.0, value=32.0, step=0.1, key="r_bmi",
                                          help="Indeks Massa Tubuh = berat(kg) / tinggi(m)²")
            dpf_raw     = st.number_input("DiabetesPedigreeFunction",                  min_value=0.0,  max_value=2.5,  value=0.47, step=0.01,key="r_dpf",
                                          help="Fungsi silsilah diabetes (riwayat keluarga)")
            age_raw_r   = st.number_input("Age (Usia, tahun)",                         min_value=18,   max_value=85,   value=33,   step=1,   key="r_age",
                                          help="Usia dalam tahun")

        if st.button("🔮 Prediksi Diabetes Sekarang! (Ridge)", key="ridge_predict"):
            # Transformasi nilai asli ke scaled (StandardScaler)
            def sc(val, col): return (val - DIAB_SCALE[col][0]) / DIAB_SCALE[col][1]
            input_r = np.array([[sc(pregnancies_raw, 'Pregnancies'),
                                  sc(glucose_raw,    'Glucose'),
                                  sc(bloodpressure_raw, 'BloodPressure'),
                                  sc(skinthickness_raw, 'SkinThickness'),
                                  sc(insulin_raw,    'Insulin'),
                                  sc(bmi_raw,        'BMI'),
                                  sc(dpf_raw,        'DiabetesPedigreeFunction'),
                                  sc(age_raw_r,      'Age')]])
            prediction_r = ridge_model.predict(input_r)[0]

            # Ridge tidak punya predict_proba — gunakan decision_function
            dec_score = ridge_model.decision_function(input_r)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if prediction_r == 1:
                st.markdown('<div class="pred-box pred-diabetes">🩸 TERDETEKSI DIABETES</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-box pred-normal">💚 TIDAK TERDETEKSI DIABETES</div>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Kelas Prediksi",    "Diabetes (1)" if prediction_r==1 else "Non-Diabetes (0)")
            c2.metric("Decision Score",    f"{dec_score:.4f}", help="Score Ridge: positif → Diabetes, negatif → Non-Diabetes")

            st.markdown(f"""
            <div class="info-box">
                <b>ℹ️ Interpretasi Decision Score = {dec_score:.4f}</b><br>
                Ridge Classifier tidak menghasilkan probabilitas langsung. Decision score menunjukkan seberapa jauh
                dari batas keputusan (threshold = 0).<br>
                • Score <b>&gt; 0</b> → prediksi <b>Diabetes</b><br>
                • Score <b>&lt; 0</b> → prediksi <b>Non-Diabetes</b><br>
                • Semakin jauh dari 0, semakin yakin model terhadap prediksinya.
            </div>""", unsafe_allow_html=True)

            # Akurasi & kesalahan model
            st.markdown("---")
            st.markdown('<div class="section-title-blue">📊 Performa Model Ridge pada Data Test</div>', unsafe_allow_html=True)
            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.markdown(f'<div class="metric-card" style="border-left-color:#64b5f6"><div class="label">Akurasi Model</div><div class="value" style="color:#64b5f6">{acc_r*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca2.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{(1-acc_r)*100:.2f}%</div></div>', unsafe_allow_html=True)
            cm_rt = confusion_matrix(y_test_r, y_pred_r); _, fp_rt, fn_rt, _ = cm_rt.ravel()
            ca3.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_rt}</div></div>', unsafe_allow_html=True)
            ca4.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_rt}</div></div>', unsafe_allow_html=True)

# ================================
# ============================================================
#   BAGIAN CATBOOST — Azka (Thalassemia)
# ============================================================
# ================================
elif st.session_state.active_model == "CatBoost":

    st.markdown("""
    <div class="header-teal">
        <h1>🧬 Prediksi Thalassemia</h1>
        <p>Metode: Klasifikasi Multi-Kelas &nbsp;|&nbsp; Model: CatBoost Classifier &nbsp;|&nbsp; Dataset: Gabungan 3 Dataset Thalassemia &nbsp;|&nbsp; Oleh: Azka Hanif Azkia</p>
    </div>
    """, unsafe_allow_html=True)

    if 'uploaded_file_cb' not in dir():
        uploaded_file_cb = None
        uploaded_pkl_cb  = None
        test_size_cb     = 20

    if uploaded_file_cb is None:
        st.markdown("""
        <div class="empty-state">
            <h2>📂 Belum Ada Dataset</h2>
            <p>Upload file XLSX thalassemia dan model PKL CatBoost melalui sidebar kiri</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="step-box"><h4>① Upload Dataset XLSX</h4><p>Upload file thalassemia_processed.xlsx melalui sidebar</p></div>
            <div class="step-box"><h4>② Upload Model PKL</h4><p>Upload file catboost_model.pkl yang sudah ditraining</p></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step-box"><h4>③ Lihat Hasil Analisis</h4><p>Tab Data, Preprocessing, Training & Evaluasi CatBoost</p></div>
            <div class="step-box"><h4>④ Coba Prediksi</h4><p>Masukkan data pasien untuk prediksi jenis Thalassemia</p></div>
            """, unsafe_allow_html=True)
        st.stop()

    # LOAD DATA
    import io
    df_cb = pd.read_excel(uploaded_file_cb)
    df_cb = df_cb.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    st.success(f"✅ Dataset Thalassemia berhasil dimuat! **{df_cb.shape[0]} baris** dan **{df_cb.shape[1]} kolom**")

    # Model features (dari CatBoost yang sudah ditraining)
    CB_FEATS    = ['Sex 0: male / 1: female', 'CS/PS', 'MCH', 'MCHC', 'SEA-THAI', 'RDW', 'Age', 'RBC count']
    CB_CAT_FEATS = ['CS/PS', 'SEA-THAI']
    TARGET_CB   = 'Diagnosis'
    DIAG_LABEL  = {11: '🟢 Normal', 12: '🔵 Alpha-Thalassemia Trait (α)', 13: '🟡 Beta-Thalassemia Trait (β)', 15: '🟠 HbE Trait'}
    DIAG_SHORT  = {11: 'Normal', 12: 'Alpha-Thal Trait', 13: 'Beta-Thal Trait', 15: 'HbE Trait'}

    # Encode kategorik untuk sklearn metrics
    df_cb_enc = df_cb.copy()
    for c in CB_CAT_FEATS:
        if c in df_cb_enc.columns:
            df_cb_enc[c] = df_cb_enc[c].astype('category').cat.codes

    X_cb_enc = df_cb_enc[CB_FEATS]
    y_cb     = df_cb[TARGET_CB]

    X_cb_raw = df_cb[CB_FEATS]

    X_tr_cb, X_te_cb, y_tr_cb, y_te_cb = train_test_split(
        X_cb_enc, y_cb, test_size=test_size_cb/100, random_state=42, stratify=y_cb
    )
    X_tr_raw, X_te_raw, _, _ = train_test_split(
        X_cb_raw, y_cb, test_size=test_size_cb/100, random_state=42, stratify=y_cb
    )

    # LOAD MODEL
    cb_model = None
    model_src_cb = "retrain"
    if uploaded_pkl_cb is not None:
        try:
            cb_model = joblib.load(uploaded_pkl_cb)
            st.success("✅ Model **catboost_model.pkl** berhasil dimuat!")
            model_src_cb = "pkl"
        except Exception as e:
            try:
                uploaded_pkl_cb.seek(0)
                cb_model = pickle.load(uploaded_pkl_cb)
                st.success("✅ Model **catboost_model.pkl** berhasil dimuat!")
                model_src_cb = "pkl"
            except Exception as e2:
                st.warning(f"⚠️ Gagal memuat PKL ({e2}). Model CatBoost ditraining ulang.")

    if cb_model is None:
        if not CATBOOST_OK:
            st.error("❌ CatBoost tidak terinstall. Jalankan: pip install catboost")
            st.stop()
        st.warning("⚠️ PKL belum diupload. Model CatBoost ditraining ulang dari data.")
        cb_model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, random_seed=42, verbose=0)
        cb_model.fit(X_tr_raw, y_tr_cb, cat_features=CB_CAT_FEATS)
        model_src_cb = "retrain"

    # Predict — gunakan raw (categorical) jika bisa, encoded jika tidak
    try:
        y_pred_cb = cb_model.predict(X_te_raw).flatten().astype(int)
    except Exception:
        y_pred_cb = cb_model.predict(X_te_cb).flatten().astype(int)

    y_te_cb_arr = y_te_cb.values
    acc_cb  = accuracy_score(y_te_cb_arr, y_pred_cb)
    prec_cb = precision_score(y_te_cb_arr, y_pred_cb, average='weighted', zero_division=0)
    rec_cb  = recall_score(y_te_cb_arr, y_pred_cb, average='weighted', zero_division=0)
    f1_cb   = f1_score(y_te_cb_arr, y_pred_cb, average='weighted', zero_division=0)

    # TABS
    ct1, ct2, ct3, ct4, ct5 = st.tabs([
        "📂 Data", "🔧 Preprocessing", "📐 Training CatBoost", "📊 Evaluasi", "🔮 Prediksi"
    ])

    with ct1:
        st.markdown('<div class="section-title-teal">📂 Eksplorasi Dataset Thalassemia</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📋 Total Data",  f"{df_cb.shape[0]} baris")
        c2.metric("📊 Kolom",       f"{df_cb.shape[1]} kolom")
        c3.metric("🎯 Fitur Model", f"{len(CB_FEATS)} fitur")
        c4.metric("🏷️ Kelas",      f"4 kelas")

        st.markdown('<div class="section-title-teal">👀 Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(df_cb.head(10), use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-teal">📈 Statistik Deskriptif</div>', unsafe_allow_html=True)
            st.dataframe(df_cb.describe().round(3), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title-teal">🎯 Distribusi Target (Diagnosis)</div>', unsafe_allow_html=True)
            diag_counts = df_cb[TARGET_CB].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            labels = [DIAG_SHORT.get(k, str(k)) for k in diag_counts.index]
            bars = ax.bar(labels, diag_counts.values,
                          color=['#26a69a','#00695c','#80cbc4','#004d40'], edgecolor='white')
            for bar, val in zip(bars, diag_counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                        str(val), ha='center', color='white', fontweight='bold')
            ax.set_title('Distribusi Diagnosis Thalassemia', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#444')
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="section-title-teal">🔥 Heatmap Korelasi (Fitur Numerik)</div>', unsafe_allow_html=True)
        num_cols_cb = ['Age','MCH','MCHC','RDW','RBC count','Sex 0: male / 1: female', TARGET_CB]
        available_num = [c for c in num_cols_cb if c in df_cb.columns]
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e2130')
        sns.heatmap(df_cb[available_num].corr(), annot=True, fmt='.2f', cmap='YlGn',
                    linewidths=0.5, ax=ax, annot_kws={"size": 9})
        ax.set_title('Korelasi Fitur Numerik — Thalassemia', color='white', fontweight='bold')
        ax.tick_params(colors='white'); st.pyplot(fig)

    with ct2:
        st.markdown('<div class="section-title-teal">🔧 Preprocessing Pipeline — Thalassemia</div>', unsafe_allow_html=True)
        st.info("ℹ️ Dataset merupakan **gabungan dari 3 dataset** thalassemia yang telah melalui feature selection.")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-box">
                <b>✅ Pipeline Preprocessing:</b><br>
                1. <b>Penggabungan 3 dataset</b> thalassemia menjadi satu dataset terpadu<br>
                2. <b>Feature Selection</b> — dipilih 8 fitur paling relevan untuk klasifikasi<br>
                3. <b>Encoding kategorik</b> — CS/PS dan SEA-THAI diproses CatBoost secara internal<br>
                4. <b>Handling missing values</b> — imputasi nilai tidak valid<br>
                5. Dataset asli berformat <b>.xlsx</b> dengan 15 kolom awal → dipilih 8 fitur model
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box">
                <b>📋 8 Fitur Model (Feature Selection):</b><br>
                • <b>Sex 0: male / 1: female</b> : Jenis kelamin (0=male, 1=female)<br>
                • <b>CS/PS</b>                   : Jenis mutasi CS/PS (kategorik)<br>
                • <b>MCH</b>                     : Mean Corpuscular Hemoglobin<br>
                • <b>MCHC</b>                    : Mean Corpuscular Hb Concentration<br>
                • <b>SEA-THAI</b>                : Tipe mutasi SEA/Thai (kategorik)<br>
                • <b>RDW</b>                     : Red Cell Distribution Width<br>
                • <b>Age</b>                     : Usia pasien<br>
                • <b>RBC count</b>               : Jumlah sel darah merah<br>
                • <b>Target → Diagnosis</b>       : 11=Normal, 12=α-Thal, 13=β-Thal, 15=HbE
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-teal">✂️ Split Data Train & Test</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("📦 Total Data",  f"{len(df_cb)}")
        c2.metric("🏋️ Data Train", f"{len(X_tr_cb)} ({100-test_size_cb}%)")
        c3.metric("🧪 Data Test",  f"{len(X_te_cb)} ({test_size_cb}%)")
        st.dataframe(df_cb[CB_FEATS].head(), use_container_width=True)

        # Distribusi fitur numerik
        st.markdown('<div class="section-title-teal">📊 Distribusi Fitur Numerik</div>', unsafe_allow_html=True)
        num_feats_cb = ['Age','MCH','MCHC','RDW','RBC count']
        fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        fig.patch.set_facecolor('#1e2130')
        for i, col in enumerate(num_feats_cb):
            axes[i].set_facecolor('#1e2130')
            for diag_val, color, lbl in [(11,'#26a69a','Normal'),(12,'#00695c','α-Thal'),(13,'#80cbc4','β-Thal'),(15,'#004d40','HbE')]:
                subset = df_cb[df_cb[TARGET_CB]==diag_val][col].dropna()
                axes[i].hist(subset, bins=15, alpha=0.6, color=color, label=lbl, edgecolor='none')
            axes[i].set_title(col, color='white', fontsize=9, fontweight='bold')
            axes[i].tick_params(colors='white', labelsize=7)
            axes[i].legend(fontsize=6, facecolor='#2d3250', labelcolor='white')
            for sp in axes[i].spines.values(): sp.set_color('#444')
        plt.tight_layout(); st.pyplot(fig)
        st.success("✅ Data siap digunakan untuk CatBoost Classifier!")

    with ct3:
        st.markdown('<div class="section-title-teal">📐 Model CatBoost Classifier</div>', unsafe_allow_html=True)
        if model_src_cb == "pkl":
            st.markdown('<div class="pkl-box-teal">✅ Model dimuat dari file: <b>catboost_model.pkl</b><br>🔒 Model tidak ditraining ulang</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pkl-box-teal">⚠️ Model ditraining ulang karena PKL belum diupload</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("📐 Model",    "CatBoost Classifier")
        c2.metric("🔧 Kelas",   "4 (11, 12, 13, 15)")
        c3.metric("🎯 Akurasi", f"{acc_cb*100:.2f}%")

        st.markdown("""
        <div class="info-box">
            <b>📖 Cara Kerja CatBoost Classifier:</b><br><br>
            1. CatBoost adalah algoritma <b>Gradient Boosting berbasis pohon keputusan</b> yang dikembangkan oleh Yandex<br>
            2. Keunggulan utama: <b>menangani fitur kategorik secara native</b> tanpa perlu encoding manual (one-hot / label encoding)<br>
            3. Menggunakan teknik <b>Ordered Boosting</b> untuk menghindari overfitting pada data kecil<br>
            4. Setiap iterasi, model baru ditraining untuk <b>memperbaiki residual error</b> dari model sebelumnya<br>
            5. Final prediksi = <b>ensemble dari semua pohon</b> → argmax kelas dengan probabilitas tertinggi
        </div>""", unsafe_allow_html=True)

        # Feature importance
        st.markdown('<div class="section-title-teal">📊 Feature Importance CatBoost</div>', unsafe_allow_html=True)
        try:
            fi = cb_model.get_feature_importance()
            fi_df = pd.DataFrame({'Fitur': CB_FEATS, 'Importance': fi}).sort_values('Importance', ascending=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.barh(fi_df['Fitur'], fi_df['Importance'], color='#26a69a', edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, fi_df['Importance']):
                ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                        f'{val:.2f}', va='center', color='white', fontsize=9)
            ax.set_title('Feature Importance — CatBoost', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#444')
            plt.tight_layout(); st.pyplot(fig)
            st.dataframe(fi_df.set_index('Fitur').sort_values('Importance', ascending=False).round(3), use_container_width=True)
        except Exception as e:
            st.info(f"Feature importance tidak tersedia: {e}")

    with ct4:
        st.markdown('<div class="section-title-teal">📊 Hasil Evaluasi CatBoost Classifier</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col, label, val, color in [
            (c1,"Accuracy",acc_cb,"#4db6ac"),(c2,"Precision",prec_cb,"#26c6da"),
            (c3,"Recall",rec_cb,"#f39c12"),(c4,"F1-Score",f1_cb,"#e63946")]:
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-teal">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
            cm_cb = confusion_matrix(y_te_cb_arr, y_pred_cb)
            labels_cm = [DIAG_SHORT.get(k, str(k)) for k in sorted(y_cb.unique())]
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.patch.set_facecolor('#1e2130')
            sns.heatmap(cm_cb, annot=True, fmt='d', cmap='YlGn',
                        xticklabels=labels_cm, yticklabels=labels_cm,
                        linewidths=1, linecolor='#333', ax=ax,
                        annot_kws={"size":14,"weight":"bold"})
            ax.set_title('Confusion Matrix - CatBoost', color='white', fontweight='bold')
            ax.set_xlabel('Prediksi', color='white'); ax.set_ylabel('Aktual', color='white')
            ax.tick_params(colors='white'); plt.tight_layout(); st.pyplot(fig)

        with c2:
            st.markdown('<div class="section-title-teal">📈 Grafik Metrik Evaluasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Accuracy','Precision','Recall','F1-Score'],
                          [acc_cb,prec_cb,rec_cb,f1_cb],
                          color=['#4db6ac','#26c6da','#f39c12','#e63946'],
                          edgecolor='white', linewidth=1, width=0.5)
            for bar, val in zip(bars, [acc_cb,prec_cb,rec_cb,f1_cb]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val*100:.2f}%', ha='center', color='white', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_title('Metrik Evaluasi - CatBoost', color='white', fontweight='bold')
            ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2)
            for sp in ax.spines.values(): sp.set_color('#444')
            st.pyplot(fig)

        # Error analysis
        st.markdown('<div class="section-title-teal">❌ Analisis Kesalahan Model</div>', unsafe_allow_html=True)
        error_rate_cb = 1 - acc_cb
        total_wrong = int(sum(y_te_cb_arr != y_pred_cb))
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_cb*100:.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">Total Salah</div><div class="value" style="color:#f39c12">{total_wrong}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#4db6ac"><div class="label">Total Benar</div><div class="value" style="color:#4db6ac">{len(y_te_cb_arr)-total_wrong}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#64b5f6"><div class="label">Total Test</div><div class="value" style="color:#64b5f6">{len(y_te_cb_arr)}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <b>📌 Interpretasi Kesalahan (Multi-Kelas):</b><br>
            • Model memprediksi 4 kelas: Normal (11), Alpha-Thal (12), Beta-Thal (13), HbE (15)<br>
            • <b>Error Rate = {error_rate_cb*100:.2f}%</b> → dari {len(y_te_cb_arr)} data test, {total_wrong} prediksi salah<br>
            • Metrik menggunakan <b>weighted average</b> untuk menangani ketidakseimbangan kelas<br>
            • Confusion matrix menunjukkan distribusi kesalahan antar kelas
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-teal">📋 Classification Report</div>', unsafe_allow_html=True)
        report_cb = classification_report(y_te_cb_arr, y_pred_cb,
                                          target_names=[DIAG_SHORT.get(k, str(k)) for k in sorted(y_cb.unique())],
                                          output_dict=True)
        st.dataframe(pd.DataFrame(report_cb).transpose().round(4), use_container_width=True)

        st.markdown('<div class="section-title-teal">📝 Kesimpulan</div>', unsafe_allow_html=True)
        grade_cb, color_cb = ("BAIK 🟢","#2ecc71") if acc_cb>=0.85 else ("CUKUP 🟡","#f39c12") if acc_cb>=0.70 else ("KURANG 🔴","#e63946")
        st.markdown(f"""
        <div class="info-box">
            <b>Oleh</b>        : Azka<br>
            <b>Dataset</b>     : Gabungan 3 Dataset Thalassemia (.xlsx) &nbsp;|&nbsp;
            <b>Total Data</b>  : {len(df_cb)} baris &nbsp;|&nbsp;
            <b>Fitur</b>       : {len(CB_FEATS)} fitur (feature selection)<br><br>
            <b>Model</b>       : CatBoost Classifier &nbsp;|&nbsp;
            <b>Sumber</b>      : {'📦 catboost_model.pkl' if model_src_cb=='pkl' else '🔄 Retrain'} &nbsp;|&nbsp;
            <b>Split</b>       : {100-test_size_cb}% Train / {test_size_cb}% Test<br><br>
            <b>Accuracy</b>    : <span style="color:{color_cb}; font-weight:800">{acc_cb*100:.2f}%</span> &nbsp;→&nbsp;
            Performa: <span style="color:{color_cb}; font-weight:800">{grade_cb}</span>
        </div>""", unsafe_allow_html=True)

    with ct5:
        st.markdown('<div class="section-title-teal">🔮 Prediksi Jenis Thalassemia — CatBoost</div>', unsafe_allow_html=True)
        st.markdown("Masukkan data pasien untuk memprediksi jenis thalassemia berdasarkan 8 fitur terpilih.")

        c1,c2 = st.columns(2)
        with c1:
            cb_sex  = st.selectbox("Sex (Jenis Kelamin)", [0.0, 1.0], format_func=lambda x: "Male (0)" if x==0.0 else "Female (1)", key="cb_sex")
            cb_csps = st.selectbox("CS/PS (Tipe Mutasi)", ['Neg','CST','HCs','HCS','PS','Unknown'], key="cb_csps")
            cb_mch  = st.number_input("MCH (pg)", min_value=10.0, max_value=40.0, value=25.0, step=0.1, key="cb_mch")
            cb_mchc = st.number_input("MCHC (g/dL)", min_value=20.0, max_value=40.0, value=31.0, step=0.1, key="cb_mchc")
        with c2:
            cb_sea  = st.selectbox("SEA-THAI (Tipe Mutasi)", ['Neg','SEA','THAI','Unknown'], key="cb_sea")
            cb_rdw  = st.number_input("RDW (%)", min_value=10.0, max_value=35.0, value=14.5, step=0.1, key="cb_rdw")
            cb_age  = st.number_input("Age (Usia)", min_value=1, max_value=80, value=30, step=1, key="cb_age")
            cb_rbc  = st.number_input("RBC count (×10⁶/µL)", min_value=1.0, max_value=8.0, value=4.5, step=0.01, key="cb_rbc")

        if st.button("🔮 Prediksi Thalassemia! (CatBoost)", key="cb_predict"):
            input_cb = pd.DataFrame([[cb_sex, cb_csps, cb_mch, cb_mchc, cb_sea, cb_rdw, cb_age, cb_rbc]],
                                     columns=CB_FEATS)
            try:
                pred_cb = int(cb_model.predict(input_cb)[0][0])
                proba_cb = cb_model.predict_proba(input_cb)[0]
                classes_cb = [int(c) for c in cb_model.classes_]
            except Exception as e:
                # fallback encoded
                input_cb_enc = input_cb.copy()
                for c in CB_CAT_FEATS:
                    input_cb_enc[c] = input_cb_enc[c].astype('category').cat.codes
                pred_cb = int(cb_model.predict(input_cb_enc)[0])
                proba_cb = cb_model.predict_proba(input_cb_enc)[0]
                classes_cb = [int(c) for c in cb_model.classes_]

            pred_label = DIAG_LABEL.get(pred_cb, f"Kode {pred_cb}")
            st.markdown("<br>", unsafe_allow_html=True)
            if pred_cb == 11:
                st.markdown(f'<div class="pred-box pred-sehat">🟢 HASIL: {pred_label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="pred-box pred-thalassemia">🧬 HASIL: {pred_label}</div>', unsafe_allow_html=True)

            st.markdown("**Probabilitas per Kelas:**")
            prob_cols = st.columns(len(classes_cb))
            for i, (cls, prob) in enumerate(zip(classes_cb, proba_cb)):
                prob_cols[i].metric(DIAG_SHORT.get(cls, str(cls)), f"{prob*100:.2f}%")

            # Performa model
            st.markdown("---")
            st.markdown('<div class="section-title-teal">📊 Performa Model CatBoost pada Data Test</div>', unsafe_allow_html=True)
            ca1,ca2,ca3,ca4 = st.columns(4)
            ca1.markdown(f'<div class="metric-card" style="border-left-color:#4db6ac"><div class="label">Akurasi Model</div><div class="value" style="color:#4db6ac">{acc_cb*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca2.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_cb*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca3.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">Total Salah</div><div class="value" style="color:#f39c12">{total_wrong}</div></div>', unsafe_allow_html=True)
            ca4.markdown(f'<div class="metric-card" style="border-left-color:#64b5f6"><div class="label">F1-Score</div><div class="value" style="color:#64b5f6">{f1_cb*100:.2f}%</div></div>', unsafe_allow_html=True)


# ================================
# ============================================================
#   BAGIAN QDA — Bian (Gejala Diabetes)
# ============================================================
# ================================
elif st.session_state.active_model == "QDA":

    st.markdown("""
    <div class="header-green">
        <h1>🩸 Prediksi Gejala Diabetes</h1>
        <p>Metode: Klasifikasi &nbsp;|&nbsp; Model: QDA (Quadratic Discriminant Analysis) &nbsp;|&nbsp; Dataset: Diabetes Symptoms &nbsp;|&nbsp; Oleh: Muh.Zhabiansyah Lazuardi</p>
    </div>
    """, unsafe_allow_html=True)

    if 'uploaded_file_qda' not in dir():
        uploaded_file_qda = None
        uploaded_pkl_qda  = None
        test_size_qda     = 20
        sep_option_qda    = ","

    if uploaded_file_qda is None:
        st.markdown("""
        <div class="empty-state">
            <h2>📂 Belum Ada Dataset</h2>
            <p>Upload file CSV diabetes symptoms dan model PKL QDA melalui sidebar kiri</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="step-box"><h4>① Upload Dataset CSV</h4><p>Upload file diabetes_data_upload.csv melalui sidebar</p></div>
            <div class="step-box"><h4>② Upload Model PKL</h4><p>Upload file qda_model_8_features.pkl yang sudah ditraining</p></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step-box"><h4>③ Lihat Hasil Analisis</h4><p>Tab Data, Preprocessing, Training & Evaluasi QDA</p></div>
            <div class="step-box"><h4>④ Coba Prediksi</h4><p>Masukkan gejala pasien untuk prediksi risiko diabetes</p></div>
            """, unsafe_allow_html=True)
        st.stop()

    # LOAD DATA
    df_qda = pd.read_csv(uploaded_file_qda, sep=sep_option_qda)

    # Encode: Yes/No → 1/0, Gender → 1/0, class → 1/0
    df_qda_enc = df_qda.copy()
    for c in df_qda_enc.columns:
        if df_qda_enc[c].dtype == object:
            if set(df_qda_enc[c].dropna().str.strip().unique()) <= {'Yes','No','yes','no'}:
                df_qda_enc[c] = df_qda_enc[c].str.strip().map({'Yes':1,'No':0,'yes':1,'no':0})
            elif c == 'Gender':
                df_qda_enc[c] = df_qda_enc[c].str.strip().map({'Male':1,'Female':0,'male':1,'female':0})
            elif c == 'class':
                df_qda_enc[c] = df_qda_enc[c].str.strip().map({'Positive':1,'Negative':0,'positive':1,'negative':0})

    st.success(f"✅ Dataset Diabetes Symptoms berhasil dimuat! **{df_qda_enc.shape[0]} baris** dan **{df_qda_enc.shape[1]} kolom**")

    QDA_FEATS  = ['Polyuria','Polydipsia','partial paresis','sudden weight loss','Itching','Alopecia','Polyphagia','Gender']
    TARGET_QDA = 'class'

    X_qda = df_qda_enc[QDA_FEATS]
    y_qda = df_qda_enc[TARGET_QDA]

    X_tr_qda, X_te_qda, y_tr_qda, y_te_qda = train_test_split(
        X_qda, y_qda, test_size=test_size_qda/100, random_state=42, stratify=y_qda
    )

    # LOAD MODEL
    qda_model = None
    model_src_qda = "retrain"
    if uploaded_pkl_qda is not None:
        try:
            qda_model = joblib.load(uploaded_pkl_qda)
            st.success("✅ Model **qda_model_8_features.pkl** berhasil dimuat!")
            model_src_qda = "pkl"
        except Exception as e:
            try:
                uploaded_pkl_qda.seek(0)
                qda_model = pickle.load(uploaded_pkl_qda)
                model_src_qda = "pkl"
            except Exception as e2:
                st.warning(f"⚠️ Gagal memuat PKL. Model QDA ditraining ulang.")

    if qda_model is None:
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        st.warning("⚠️ PKL belum diupload. Model QDA ditraining ulang.")
        qda_model = QuadraticDiscriminantAnalysis()
        qda_model.fit(X_tr_qda, y_tr_qda)
        model_src_qda = "retrain"

    y_pred_qda = qda_model.predict(X_te_qda)
    acc_qda  = accuracy_score(y_te_qda, y_pred_qda)
    prec_qda = precision_score(y_te_qda, y_pred_qda, zero_division=0)
    rec_qda  = recall_score(y_te_qda,   y_pred_qda, zero_division=0)
    f1_qda   = f1_score(y_te_qda,       y_pred_qda, zero_division=0)

    # TABS
    qt1, qt2, qt3, qt4, qt5 = st.tabs([
        "📂 Data", "🔧 Preprocessing", "📐 Training QDA", "📊 Evaluasi", "🔮 Prediksi"
    ])

    with qt1:
        st.markdown('<div class="section-title-green">📂 Eksplorasi Dataset Gejala Diabetes</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📋 Total Data",  f"{df_qda.shape[0]} baris")
        c2.metric("📊 Kolom",       f"{df_qda.shape[1]} kolom")
        c3.metric("🎯 Fitur Model", f"{len(QDA_FEATS)} fitur")
        c4.metric("❓ Missing",     "0 ✅")

        st.markdown('<div class="section-title-green">👀 Preview Data (Asli)</div>', unsafe_allow_html=True)
        st.dataframe(df_qda.head(10), use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-green">📈 Statistik Deskriptif (Encoded)</div>', unsafe_allow_html=True)
            st.dataframe(df_qda_enc.describe().round(3), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title-green">🎯 Distribusi Target (class)</div>', unsafe_allow_html=True)
            tc_qda = df_qda_enc[TARGET_QDA].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Negative (0)','Positive (1)'],
                          [tc_qda.get(0,0), tc_qda.get(1,0)],
                          color=['#2e7d32','#81c784'], edgecolor='white', linewidth=1.2)
            for bar, val in zip(bars, [tc_qda.get(0,0), tc_qda.get(1,0)]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        str(val), ha='center', color='white', fontweight='bold')
            ax.set_title('Distribusi Target — Diabetes Symptoms', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#444')
            st.pyplot(fig)

        st.markdown('<div class="section-title-green">📊 Frekuensi Gejala (8 Fitur Model)</div>', unsafe_allow_html=True)
        gejala_sum = X_qda.sum().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        bars = ax.barh(gejala_sum.index, gejala_sum.values, color='#2e7d32', edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, gejala_sum.values):
            ax.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2,
                    str(int(val)), va='center', color='white', fontsize=9)
        ax.set_title('Frekuensi Kemunculan Gejala di Dataset', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#444')
        plt.tight_layout(); st.pyplot(fig)

    with qt2:
        st.markdown('<div class="section-title-green">🔧 Preprocessing Pipeline — Gejala Diabetes</div>', unsafe_allow_html=True)
        st.info("ℹ️ Dataset berisi gejala klinis diabetes dalam format Yes/No yang sudah diproses.")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-box">
                <b>✅ Preprocessing yang Dilakukan:</b><br>
                1. <b>Label Encoding</b> — Yes/No → 1/0 untuk semua kolom gejala<br>
                2. <b>Gender Encoding</b> — Male → 1, Female → 0<br>
                3. <b>Target Encoding</b> — Positive → 1, Negative → 0<br>
                4. <b>Feature Selection</b> — dipilih 8 fitur paling signifikan dari 16 fitur<br>
                5. Data <b>bersih</b> — tidak ada missing value
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box">
                <b>📋 8 Fitur Model (Feature Selection):</b><br>
                • <b>Polyuria</b>           : Sering buang air kecil berlebih<br>
                • <b>Polydipsia</b>         : Rasa haus berlebih<br>
                • <b>partial paresis</b>    : Kelemahan otot sebagian<br>
                • <b>sudden weight loss</b> : Penurunan berat badan mendadak<br>
                • <b>Itching</b>            : Gatal-gatal<br>
                • <b>Alopecia</b>           : Kerontokan rambut<br>
                • <b>Polyphagia</b>         : Nafsu makan berlebih<br>
                • <b>Gender</b>             : Jenis kelamin (1=Male, 0=Female)
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-green">✂️ Split Data Train & Test</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("📦 Total Data",  f"{len(df_qda_enc)}")
        c2.metric("🏋️ Data Train", f"{len(X_tr_qda)} ({100-test_size_qda}%)")
        c3.metric("🧪 Data Test",  f"{len(X_te_qda)} ({test_size_qda}%)")
        st.dataframe(X_tr_qda.head(), use_container_width=True)
        st.success("✅ Data siap digunakan untuk QDA!")

    with qt3:
        st.markdown('<div class="section-title-green">📐 Model QDA (Quadratic Discriminant Analysis)</div>', unsafe_allow_html=True)
        if model_src_qda == "pkl":
            st.markdown('<div class="pkl-box-green">✅ Model dimuat dari file: <b>qda_model_8_features.pkl</b><br>🔒 Model tidak ditraining ulang</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pkl-box-green">⚠️ Model ditraining ulang karena PKL belum diupload</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("📐 Model",    "QDA Classifier")
        c2.metric("🔧 Fitur",   "8 fitur terpilih")
        c3.metric("🎯 Akurasi", f"{acc_qda*100:.2f}%")

        st.markdown("""
        <div class="info-box">
            <b>📖 Cara Kerja QDA (Quadratic Discriminant Analysis):</b><br><br>
            1. QDA adalah generalisasi dari LDA yang <b>tidak mengasumsikan matriks kovarians sama</b> antar kelas<br>
            2. Setiap kelas memiliki matriks kovarians <b>sendiri-sendiri</b> → batas keputusan berbentuk <b>kuadratik (kurva)</b><br>
            3. Menghitung <b>mean vektor (μₖ) dan matriks kovarians (Σₖ)</b> untuk setiap kelas k<br>
            4. Prediksi menggunakan <b>discriminant score</b> berbasis distribusi Gaussian multivariate<br>
            5. Lebih fleksibel dari LDA tapi membutuhkan lebih banyak data untuk estimasi kovarians yang stabil
        </div>""", unsafe_allow_html=True)

        # Mean per kelas
        st.markdown('<div class="section-title-green">📊 Mean Fitur per Kelas</div>', unsafe_allow_html=True)
        try:
            mean_df = pd.DataFrame(qda_model.means_, columns=QDA_FEATS, index=['Negative (0)','Positive (1)'])
            st.dataframe(mean_df.round(3), use_container_width=True)

            # Visualisasi mean
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            x = np.arange(len(QDA_FEATS))
            width = 0.35
            ax.bar(x - width/2, mean_df.iloc[0], width, label='Negative (0)', color='#2e7d32', edgecolor='white', alpha=0.85)
            ax.bar(x + width/2, mean_df.iloc[1], width, label='Positive (1)', color='#81c784', edgecolor='white', alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(QDA_FEATS, rotation=25, ha='right', color='white', fontsize=8)
            ax.set_title('Mean Fitur per Kelas — QDA', color='white', fontweight='bold')
            ax.legend(facecolor='#2d3250', labelcolor='white')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#444')
            plt.tight_layout(); st.pyplot(fig)
        except Exception as e:
            st.info(f"Visualisasi mean tidak tersedia: {e}")

    with qt4:
        st.markdown('<div class="section-title-green">📊 Hasil Evaluasi QDA</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col, label, val, color in [
            (c1,"Accuracy",acc_qda,"#81c784"),(c2,"Precision",prec_qda,"#26c6da"),
            (c3,"Recall",rec_qda,"#f39c12"),(c4,"F1-Score",f1_qda,"#e63946")]:
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-green">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
            cm_qda = confusion_matrix(y_te_qda, y_pred_qda)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130')
            sns.heatmap(cm_qda, annot=True, fmt='d', cmap='Greens',
                        xticklabels=['Negative','Positive'],
                        yticklabels=['Negative','Positive'],
                        linewidths=1, linecolor='#333', ax=ax,
                        annot_kws={"size":16,"weight":"bold"})
            ax.set_title('Confusion Matrix - QDA', color='white', fontweight='bold')
            ax.set_xlabel('Prediksi', color='white'); ax.set_ylabel('Aktual', color='white')
            ax.tick_params(colors='white'); st.pyplot(fig)
        with c2:
            st.markdown('<div class="section-title-green">📈 Grafik Metrik Evaluasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Accuracy','Precision','Recall','F1-Score'],
                          [acc_qda,prec_qda,rec_qda,f1_qda],
                          color=['#81c784','#26c6da','#f39c12','#e63946'],
                          edgecolor='white', linewidth=1, width=0.5)
            for bar, val in zip(bars, [acc_qda,prec_qda,rec_qda,f1_qda]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val*100:.2f}%', ha='center', color='white', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_title('Metrik Evaluasi - QDA', color='white', fontweight='bold')
            ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2)
            for sp in ax.spines.values(): sp.set_color('#444')
            st.pyplot(fig)

        # Error analysis
        st.markdown('<div class="section-title-green">❌ Analisis Kesalahan Model</div>', unsafe_allow_html=True)
        tn_q, fp_q, fn_q, tp_q = cm_qda.ravel()
        error_rate_qda = 1 - acc_qda
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_qda*100:.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_q}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_q}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#27ae60"><div class="label">Total Benar</div><div class="value" style="color:#27ae60">{tp_q+tn_q}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <b>📌 Interpretasi Kesalahan:</b><br>
            • <b>False Positive (FP = {fp_q})</b>: Model memprediksi <i>Positif Diabetes</i> padahal sebenarnya <i>Negatif</i><br>
            • <b>False Negative (FN = {fn_q})</b>: Model memprediksi <i>Negatif</i> padahal sebenarnya <i>Positif Diabetes</i> (lebih berbahaya)<br>
            • <b>Error Rate = {error_rate_qda*100:.2f}%</b> → dari {len(y_te_qda)} data test, {fp_q+fn_q} prediksi salah
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-green">📋 Classification Report</div>', unsafe_allow_html=True)
        report_qda = classification_report(y_te_qda, y_pred_qda,
                                            target_names=['Negative (0)','Positive (1)'],
                                            output_dict=True)
        st.dataframe(pd.DataFrame(report_qda).transpose().round(4), use_container_width=True)

        st.markdown('<div class="section-title-green">📝 Kesimpulan</div>', unsafe_allow_html=True)
        grade_qda, color_qda = ("BAIK 🟢","#2ecc71") if acc_qda>=0.85 else ("CUKUP 🟡","#f39c12") if acc_qda>=0.70 else ("KURANG 🔴","#e63946")
        st.markdown(f"""
        <div class="info-box">
            <b>Oleh</b>        : Bian<br>
            <b>Dataset</b>     : Diabetes Symptoms (Gejala Klinis) &nbsp;|&nbsp;
            <b>Total Data</b>  : {len(df_qda_enc)} baris &nbsp;|&nbsp;
            <b>Fitur</b>       : {len(QDA_FEATS)} fitur (feature selection)<br><br>
            <b>Model</b>       : QDA Classifier &nbsp;|&nbsp;
            <b>Sumber</b>      : {'📦 qda_model_8_features.pkl' if model_src_qda=='pkl' else '🔄 Retrain'} &nbsp;|&nbsp;
            <b>Split</b>       : {100-test_size_qda}% Train / {test_size_qda}% Test<br><br>
            <b>Accuracy</b>    : <span style="color:{color_qda}; font-weight:800">{acc_qda*100:.2f}%</span> &nbsp;→&nbsp;
            Performa: <span style="color:{color_qda}; font-weight:800">{grade_qda}</span>
        </div>""", unsafe_allow_html=True)

    with qt5:
        st.markdown('<div class="section-title-green">🔮 Prediksi Risiko Diabetes — QDA</div>', unsafe_allow_html=True)
        st.markdown("Masukkan gejala pasien (pilih **Yes/No**) untuk memprediksi risiko diabetes.")

        c1,c2 = st.columns(2)
        with c1:
            qda_polyuria  = st.selectbox("Polyuria (Sering BAK berlebih)",       ['No','Yes'], key="qda_poly")
            qda_polydip   = st.selectbox("Polydipsia (Haus berlebih)",            ['No','Yes'], key="qda_polyd")
            qda_paresis   = st.selectbox("Partial Paresis (Kelemahan otot)",      ['No','Yes'], key="qda_par")
            qda_swl       = st.selectbox("Sudden Weight Loss (Penurunan BB tiba-tiba)", ['No','Yes'], key="qda_swl")
        with c2:
            qda_itching   = st.selectbox("Itching (Gatal-gatal)",                ['No','Yes'], key="qda_itch")
            qda_alopecia  = st.selectbox("Alopecia (Kerontokan rambut)",          ['No','Yes'], key="qda_alop")
            qda_polyphagia = st.selectbox("Polyphagia (Nafsu makan berlebih)",   ['No','Yes'], key="qda_polyph")
            qda_gender    = st.selectbox("Gender (Jenis Kelamin)",               ['Male','Female'], key="qda_gender")

        yn = lambda x: 1 if x=='Yes' else 0
        gn = lambda x: 1 if x=='Male' else 0

        if st.button("🔮 Prediksi Diabetes! (QDA)", key="qda_predict"):
            input_qda = np.array([[yn(qda_polyuria), yn(qda_polydip), yn(qda_paresis),
                                    yn(qda_swl), yn(qda_itching), yn(qda_alopecia),
                                    yn(qda_polyphagia), gn(qda_gender)]])
            pred_qda_val  = qda_model.predict(input_qda)[0]
            proba_qda_val = qda_model.predict_proba(input_qda)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if pred_qda_val == 1:
                st.markdown('<div class="pred-box pred-diabetes2">🩸 TERDETEKSI RISIKO DIABETES (POSITIF)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-box pred-sehat">💚 TIDAK TERDETEKSI RISIKO DIABETES (NEGATIF)</div>', unsafe_allow_html=True)

            c1,c2 = st.columns(2)
            c1.metric("Probabilitas Negatif", f"{proba_qda_val[0]*100:.2f}%")
            c2.metric("Probabilitas Positif", f"{proba_qda_val[1]*100:.2f}%")

            # Ringkasan gejala
            gejala_masuk = []
            if yn(qda_polyuria):  gejala_masuk.append("Polyuria")
            if yn(qda_polydip):   gejala_masuk.append("Polydipsia")
            if yn(qda_paresis):   gejala_masuk.append("Partial Paresis")
            if yn(qda_swl):       gejala_masuk.append("Sudden Weight Loss")
            if yn(qda_itching):   gejala_masuk.append("Itching")
            if yn(qda_alopecia):  gejala_masuk.append("Alopecia")
            if yn(qda_polyphagia): gejala_masuk.append("Polyphagia")

            st.markdown(f"""
            <div class="info-box">
                <b>📋 Gejala yang ditemukan ({len(gejala_masuk)}/7):</b><br>
                {', '.join(gejala_masuk) if gejala_masuk else 'Tidak ada gejala positif'}
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-title-green">📊 Performa Model QDA pada Data Test</div>', unsafe_allow_html=True)
            ca1,ca2,ca3,ca4 = st.columns(4)
            ca1.markdown(f'<div class="metric-card" style="border-left-color:#81c784"><div class="label">Akurasi Model</div><div class="value" style="color:#81c784">{acc_qda*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca2.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_qda*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca3.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_q}</div></div>', unsafe_allow_html=True)
            ca4.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_q}</div></div>', unsafe_allow_html=True)


# ================================
# ============================================================
#   BAGIAN BAGGING — Rifki (Penyakit Ginjal)
# ============================================================
# ================================
elif st.session_state.active_model == "Bagging":

    st.markdown("""
    <div class="header-orange">
        <h1>🫘 Prediksi Penyakit Ginjal</h1>
        <p>Metode: Klasifikasi &nbsp;|&nbsp; Model: Bagging Classifier &nbsp;|&nbsp; Dataset: Kidney Disease &nbsp;|&nbsp; Oleh: Muh.Rifki Fitriansyah G.</p>
    </div>
    """, unsafe_allow_html=True)

    if 'uploaded_file_bg' not in dir():
        uploaded_file_bg = None
        uploaded_pkl_bg  = None
        test_size_bg     = 20
        sep_option_bg    = ","

    if uploaded_file_bg is None:
        st.markdown("""
        <div class="empty-state">
            <h2>📂 Belum Ada Dataset</h2>
            <p>Upload file CSV penyakit ginjal dan model PKL Bagging melalui sidebar kiri</p>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="step-box"><h4>① Upload Dataset CSV</h4><p>Upload file dataset_ginjal_preprocessed.csv melalui sidebar</p></div>
            <div class="step-box"><h4>② Upload Model PKL</h4><p>Upload file model_bagging_final.pkl yang sudah ditraining</p></div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="step-box"><h4>③ Lihat Hasil Analisis</h4><p>Tab Data, Preprocessing, Training & Evaluasi Bagging</p></div>
            <div class="step-box"><h4>④ Coba Prediksi</h4><p>Masukkan data pasien untuk prediksi risiko penyakit ginjal</p></div>
            """, unsafe_allow_html=True)
        st.stop()

    # LOAD DATA
    df_bg = pd.read_csv(uploaded_file_bg, sep=sep_option_bg)
    df_bg = df_bg.apply(lambda col: col.astype(int) if col.dtype == bool else col)
    st.success(f"✅ Dataset Penyakit Ginjal berhasil dimuat! **{df_bg.shape[0]} baris** dan **{df_bg.shape[1]} kolom**")

    BG_FEATS    = ['Age','BMI','SystolicBP','FastingBloodSugar','BUNLevels','GFR','Edema','FatigueLevels']
    TARGET_BG   = 'Diagnosis'

    X_bg = df_bg[BG_FEATS]
    y_bg = df_bg[TARGET_BG]

    X_tr_bg, X_te_bg, y_tr_bg, y_te_bg = train_test_split(
        X_bg, y_bg, test_size=test_size_bg/100, random_state=42, stratify=y_bg
    )

    # LOAD MODEL
    bg_model = None
    model_src_bg = "retrain"
    if uploaded_pkl_bg is not None:
        try:
            bg_model = joblib.load(uploaded_pkl_bg)
            st.success("✅ Model **model_bagging_final.pkl** berhasil dimuat!")
            model_src_bg = "pkl"
        except Exception as e:
            try:
                uploaded_pkl_bg.seek(0)
                bg_model = pickle.load(uploaded_pkl_bg)
                model_src_bg = "pkl"
            except Exception as e2:
                st.warning(f"⚠️ Gagal memuat PKL. Model Bagging ditraining ulang.")

    if bg_model is None:
        st.warning("⚠️ PKL belum diupload. Model Bagging ditraining ulang dari data.")
        bg_model = BaggingClassifier(n_estimators=100, random_state=42)
        bg_model.fit(X_tr_bg, y_tr_bg)
        model_src_bg = "retrain"

    y_pred_bg = bg_model.predict(X_te_bg)
    acc_bg  = accuracy_score(y_te_bg, y_pred_bg)
    prec_bg = precision_score(y_te_bg, y_pred_bg, zero_division=0)
    rec_bg  = recall_score(y_te_bg,   y_pred_bg, zero_division=0)
    f1_bg   = f1_score(y_te_bg,       y_pred_bg, zero_division=0)

    # TABS
    bt1, bt2, bt3, bt4, bt5 = st.tabs([
        "📂 Data", "🔧 Preprocessing", "📐 Training Bagging", "📊 Evaluasi", "🔮 Prediksi"
    ])

    with bt1:
        st.markdown('<div class="section-title-orange">📂 Eksplorasi Dataset Penyakit Ginjal</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📋 Total Data",  f"{df_bg.shape[0]} baris")
        c2.metric("📊 Kolom",       f"{df_bg.shape[1]} kolom")
        c3.metric("🎯 Fitur Model", f"{len(BG_FEATS)} fitur")
        c4.metric("❓ Missing",     "0 ✅")

        st.markdown('<div class="section-title-orange">👀 Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(df_bg.head(10), use_container_width=True)

        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-orange">📈 Statistik Deskriptif</div>', unsafe_allow_html=True)
            st.dataframe(df_bg.describe().round(3), use_container_width=True)
        with c2:
            st.markdown('<div class="section-title-orange">🎯 Distribusi Target (Diagnosis)</div>', unsafe_allow_html=True)
            tc_bg = df_bg[TARGET_BG].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Tidak Sakit (0)','Penyakit Ginjal (1)'],
                          [tc_bg.get(0,0), tc_bg.get(1,0)],
                          color=['#e65100','#ffb74d'], edgecolor='white', linewidth=1.2)
            for bar, val in zip(bars, [tc_bg.get(0,0), tc_bg.get(1,0)]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        str(val), ha='center', color='white', fontweight='bold')
            ax.set_title('Distribusi Target — Penyakit Ginjal', color='white', fontweight='bold')
            ax.tick_params(colors='white')
            for sp in ax.spines.values(): sp.set_color('#444')
            st.pyplot(fig)

        st.markdown('<div class="section-title-orange">🔥 Heatmap Korelasi</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e2130')
        sns.heatmap(df_bg.corr(), annot=True, fmt='.2f', cmap='YlOrRd',
                    linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title('Korelasi Antar Fitur — Penyakit Ginjal', color='white', fontweight='bold')
        ax.tick_params(colors='white'); st.pyplot(fig)

    with bt2:
        st.markdown('<div class="section-title-orange">🔧 Preprocessing Pipeline — Penyakit Ginjal</div>', unsafe_allow_html=True)
        st.info("ℹ️ Dataset sudah melalui preprocessing lengkap sebelum diupload.")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="info-box">
                <b>✅ Preprocessing yang Dilakukan:</b><br>
                1. <b>Pembersihan data</b> — penanganan missing values dan duplikat<br>
                2. <b>Feature engineering</b> — seleksi 8 fitur klinis paling relevan<br>
                3. <b>Encoding</b> — fitur kategorik (Edema) dikonversi ke numerik<br>
                4. Dataset siap pakai untuk Bagging Classifier
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="info-box">
                <b>📋 8 Fitur Dataset:</b><br>
                • <b>Age</b>               : Usia pasien (tahun)<br>
                • <b>BMI</b>               : Indeks massa tubuh (kg/m²)<br>
                • <b>SystolicBP</b>        : Tekanan darah sistolik (mmHg)<br>
                • <b>FastingBloodSugar</b> : Gula darah puasa (mg/dL)<br>
                • <b>BUNLevels</b>         : Blood Urea Nitrogen (mg/dL)<br>
                • <b>GFR</b>              : Laju Filtrasi Glomerulus (mL/min)<br>
                • <b>Edema</b>             : Edema/pembengkakan (0/1)<br>
                • <b>FatigueLevels</b>     : Tingkat kelelahan (skala)<br>
                • <b>Target → Diagnosis</b>: 0 = Tidak Sakit, 1 = Penyakit Ginjal
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-orange">✂️ Split Data Train & Test</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("📦 Total Data",  f"{len(df_bg)}")
        c2.metric("🏋️ Data Train", f"{len(X_tr_bg)} ({100-test_size_bg}%)")
        c3.metric("🧪 Data Test",  f"{len(X_te_bg)} ({test_size_bg}%)")
        st.dataframe(X_tr_bg.head(), use_container_width=True)

        st.markdown('<div class="section-title-orange">📊 Distribusi Fitur per Kelas</div>', unsafe_allow_html=True)
        num_bg = ['Age','BMI','SystolicBP','FastingBloodSugar','BUNLevels','GFR','FatigueLevels']
        fig, axes = plt.subplots(1, 7, figsize=(18, 4))
        fig.patch.set_facecolor('#1e2130')
        for i, col in enumerate(num_bg):
            axes[i].set_facecolor('#1e2130')
            for cls_val, color, lbl in [(0,'#e65100','Tidak Sakit'),(1,'#ffb74d','Penyakit Ginjal')]:
                subset = df_bg[df_bg[TARGET_BG]==cls_val][col].dropna()
                axes[i].hist(subset, bins=15, alpha=0.6, color=color, label=lbl, edgecolor='none')
            axes[i].set_title(col, color='white', fontsize=8, fontweight='bold')
            axes[i].tick_params(colors='white', labelsize=6)
            if i == 0: axes[i].legend(fontsize=6, facecolor='#2d3250', labelcolor='white')
            for sp in axes[i].spines.values(): sp.set_color('#444')
        plt.tight_layout(); st.pyplot(fig)
        st.success("✅ Data siap digunakan untuk Bagging Classifier!")

    with bt3:
        st.markdown('<div class="section-title-orange">📐 Model Bagging Classifier</div>', unsafe_allow_html=True)
        if model_src_bg == "pkl":
            st.markdown('<div class="pkl-box-orange">✅ Model dimuat dari file: <b>model_bagging_final.pkl</b><br>🔒 Model tidak ditraining ulang</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="pkl-box-orange">⚠️ Model ditraining ulang karena PKL belum diupload</div>', unsafe_allow_html=True)

        n_est = bg_model.n_estimators if hasattr(bg_model, 'n_estimators') else '?'
        c1,c2,c3 = st.columns(3)
        c1.metric("📐 Model",        "Bagging Classifier")
        c2.metric("🔧 N Estimators", str(n_est))
        c3.metric("🎯 Akurasi",      f"{acc_bg*100:.2f}%")

        st.markdown("""
        <div class="info-box">
            <b>📖 Cara Kerja Bagging Classifier:</b><br><br>
            1. <b>Bootstrap Aggregating (Bagging)</b> — melatih banyak base classifier pada subset data yang berbeda<br>
            2. Setiap subset dibuat dengan <b>sampling dengan pengembalian (bootstrap)</b> dari data training<br>
            3. Setiap base model ditraining secara <b>independen dan paralel</b><br>
            4. Final prediksi ditentukan melalui <b>majority voting</b> dari semua base classifier<br>
            5. Mengurangi <b>variance</b> dan overfitting — cocok untuk dataset dengan noise tinggi
        </div>""", unsafe_allow_html=True)

        # Estimator info
        st.markdown('<div class="section-title-orange">📊 Distribusi Prediksi Per Estimator (Sampel)</div>', unsafe_allow_html=True)
        try:
            sample_idx = np.random.choice(len(X_te_bg), min(50, len(X_te_bg)), replace=False)
            X_sample = X_te_bg.iloc[sample_idx]
            votes_0 = []
            votes_1 = []
            for est in bg_model.estimators_[:min(20, len(bg_model.estimators_))]:
                preds = est.predict(X_sample)
                votes_0.append(sum(preds == 0))
                votes_1.append(sum(preds == 1))

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            x_est = np.arange(len(votes_0))
            ax.plot(x_est, votes_0, 'o-', color='#e65100', label='Vote: Tidak Sakit (0)', linewidth=1.5)
            ax.plot(x_est, votes_1, 's-', color='#ffb74d', label='Vote: Penyakit Ginjal Kronis (1)', linewidth=1.5)
            ax.set_title('Voting per Estimator (20 estimator pertama, 50 sampel)', color='white', fontweight='bold')
            ax.set_xlabel('Estimator ke-', color='white')
            ax.set_ylabel('Jumlah Vote', color='white')
            ax.legend(facecolor='#2d3250', labelcolor='white')
            ax.tick_params(colors='white'); ax.grid(alpha=0.2)
            for sp in ax.spines.values(): sp.set_color('#444')
            plt.tight_layout(); st.pyplot(fig)
        except Exception as e:
            st.info(f"Visualisasi estimator tidak tersedia: {e}")

    with bt4:
        st.markdown('<div class="section-title-orange">📊 Hasil Evaluasi Bagging Classifier</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col, label, val, color in [
            (c1,"Accuracy",acc_bg,"#ffb74d"),(c2,"Precision",prec_bg,"#26c6da"),
            (c3,"Recall",rec_bg,"#f39c12"),(c4,"F1-Score",f1_bg,"#e63946")]:
            col.markdown(f"""
            <div class="metric-card" style="border-left-color:{color}">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title-orange">🔲 Confusion Matrix</div>', unsafe_allow_html=True)
            cm_bg = confusion_matrix(y_te_bg, y_pred_bg)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130')
            sns.heatmap(cm_bg, annot=True, fmt='d', cmap='YlOrRd',
                        xticklabels=['Tidak Sakit','Penyakit Ginjal'],
                        yticklabels=['Tidak Sakit','Penyakit Ginjal'],
                        linewidths=1, linecolor='#333', ax=ax,
                        annot_kws={"size":16,"weight":"bold"})
            ax.set_title('Confusion Matrix - Bagging', color='white', fontweight='bold')
            ax.set_xlabel('Prediksi', color='white'); ax.set_ylabel('Aktual', color='white')
            ax.tick_params(colors='white'); st.pyplot(fig)
        with c2:
            st.markdown('<div class="section-title-orange">📈 Grafik Metrik Evaluasi</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
            bars = ax.bar(['Accuracy','Precision','Recall','F1-Score'],
                          [acc_bg,prec_bg,rec_bg,f1_bg],
                          color=['#ffb74d','#26c6da','#f39c12','#e63946'],
                          edgecolor='white', linewidth=1, width=0.5)
            for bar, val in zip(bars, [acc_bg,prec_bg,rec_bg,f1_bg]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f'{val*100:.2f}%', ha='center', color='white', fontweight='bold', fontsize=11)
            ax.set_ylim(0, 1.15)
            ax.set_title('Metrik Evaluasi - Bagging', color='white', fontweight='bold')
            ax.tick_params(colors='white'); ax.grid(axis='y', alpha=0.2)
            for sp in ax.spines.values(): sp.set_color('#444')
            st.pyplot(fig)

        # Error analysis
        st.markdown('<div class="section-title-orange">❌ Analisis Kesalahan Model</div>', unsafe_allow_html=True)
        tn_b, fp_b, fn_b, tp_b = cm_bg.ravel()
        error_rate_bg = 1 - acc_bg
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_bg*100:.2f}%</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_b}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_b}</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card" style="border-left-color:#27ae60"><div class="label">Total Benar</div><div class="value" style="color:#27ae60">{tp_b+tn_b}</div></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-box">
            <b>📌 Interpretasi Kesalahan:</b><br>
            • <b>False Positive (FP = {fp_b})</b>: Model memprediksi <i>Penyakit Ginjal</i> padahal sebenarnya <i>Sehat</i><br>
            • <b>False Negative (FN = {fn_b})</b>: Model memprediksi <i>Sehat</i> padahal sebenarnya <i>Penyakit Ginjal</i> (lebih berbahaya)<br>
            • <b>Error Rate = {error_rate_bg*100:.2f}%</b> → dari {len(y_te_bg)} data test, {fp_b+fn_b} prediksi salah
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title-orange">📋 Classification Report</div>', unsafe_allow_html=True)
        report_bg = classification_report(y_te_bg, y_pred_bg,
                                          target_names=['Tidak Sakit (0)','Penyakit Ginjal (1)'],
                                          output_dict=True)
        st.dataframe(pd.DataFrame(report_bg).transpose().round(4), use_container_width=True)

        st.markdown('<div class="section-title-orange">📝 Kesimpulan</div>', unsafe_allow_html=True)
        grade_bg, color_bg = ("BAIK 🟢","#2ecc71") if acc_bg>=0.85 else ("CUKUP 🟡","#f39c12") if acc_bg>=0.70 else ("KURANG 🔴","#e63946")
        st.markdown(f"""
        <div class="info-box">
            <b>Oleh</b>        : Rifki<br>
            <b>Dataset</b>     : Kidney Disease (Penyakit Ginjal Preprocessed) &nbsp;|&nbsp;
            <b>Total Data</b>  : {len(df_bg)} baris &nbsp;|&nbsp;
            <b>Fitur</b>       : {len(BG_FEATS)} fitur<br><br>
            <b>Model</b>       : Bagging Classifier &nbsp;|&nbsp;
            <b>Sumber</b>      : {'📦 model_bagging_final.pkl' if model_src_bg=='pkl' else '🔄 Retrain'} &nbsp;|&nbsp;
            <b>Split</b>       : {100-test_size_bg}% Train / {test_size_bg}% Test<br><br>
            <b>Accuracy</b>    : <span style="color:{color_bg}; font-weight:800">{acc_bg*100:.2f}%</span> &nbsp;→&nbsp;
            Performa: <span style="color:{color_bg}; font-weight:800">{grade_bg}</span>
        </div>""", unsafe_allow_html=True)

    with bt5:
        st.markdown('<div class="section-title-orange">🔮 Prediksi Penyakit Ginjal — Bagging Classifier</div>', unsafe_allow_html=True)
        st.markdown("Masukkan data klinis pasien untuk memprediksi risiko penyakit ginjal.")

        c1,c2 = st.columns(2)
        with c1:
            bg_age  = st.number_input("Age (Usia, tahun)",                 min_value=1,   max_value=100,  value=50,   step=1,   key="bg_age")
            bg_bmi  = st.number_input("BMI (kg/m²)",                       min_value=10.0, max_value=60.0, value=25.0, step=0.1, key="bg_bmi")
            bg_sbp  = st.number_input("SystolicBP (Tek. darah sistolik, mmHg)", min_value=80, max_value=200, value=120, step=1, key="bg_sbp")
            bg_fbs  = st.number_input("FastingBloodSugar (Gula darah puasa, mg/dL)", min_value=50.0, max_value=400.0, value=100.0, step=0.5, key="bg_fbs")
        with c2:
            bg_bun  = st.number_input("BUNLevels (Blood Urea Nitrogen, mg/dL)", min_value=1.0, max_value=100.0, value=15.0, step=0.5, key="bg_bun")
            bg_gfr  = st.number_input("GFR (Laju Filtrasi Glomerulus, mL/min)", min_value=1.0, max_value=150.0, value=80.0, step=0.5, key="bg_gfr")
            bg_edema = st.selectbox("Edema (Pembengkakan)",               [0, 1], format_func=lambda x: "Tidak Ada (0)" if x==0 else "Ada (1)", key="bg_edema")
            bg_fatigue = st.number_input("FatigueLevels (Tingkat Kelelahan, 1-10)", min_value=0.0, max_value=10.0, value=3.0, step=0.1, key="bg_fat")

        if st.button("🔮 Prediksi Penyakit Ginjal! (Bagging)", key="bg_predict"):
            input_bg = np.array([[bg_age, bg_bmi, bg_sbp, bg_fbs, bg_bun, bg_gfr, bg_edema, bg_fatigue]])
            pred_bg_val  = bg_model.predict(input_bg)[0]
            proba_bg_val = bg_model.predict_proba(input_bg)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            if pred_bg_val == 1:
                st.markdown('<div class="pred-box pred-ginjal">🫘 TERDETEKSI RISIKO PENYAKIT GINJAL</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pred-box pred-ginjal-ok">💚 TIDAK TERDETEKSI PENYAKIT GINJAL</div>', unsafe_allow_html=True)

            c1,c2 = st.columns(2)
            c1.metric("Probabilitas Tidak Sakit", f"{proba_bg_val[0]*100:.2f}%")
            c2.metric("Probabilitas Penyakit Ginjal", f"{proba_bg_val[1]*100:.2f}%")

            st.markdown("---")
            st.markdown('<div class="section-title-orange">📊 Performa Model Bagging pada Data Test</div>', unsafe_allow_html=True)
            ca1,ca2,ca3,ca4 = st.columns(4)
            ca1.markdown(f'<div class="metric-card" style="border-left-color:#ffb74d"><div class="label">Akurasi Model</div><div class="value" style="color:#ffb74d">{acc_bg*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca2.markdown(f'<div class="metric-card" style="border-left-color:#e63946"><div class="label">Error Rate</div><div class="value" style="color:#e63946">{error_rate_bg*100:.2f}%</div></div>', unsafe_allow_html=True)
            ca3.markdown(f'<div class="metric-card" style="border-left-color:#f39c12"><div class="label">False Positive</div><div class="value" style="color:#f39c12">{fp_b}</div></div>', unsafe_allow_html=True)
            ca4.markdown(f'<div class="metric-card" style="border-left-color:#e74c3c"><div class="label">False Negative</div><div class="value" style="color:#e74c3c">{fn_b}</div></div>', unsafe_allow_html=True)