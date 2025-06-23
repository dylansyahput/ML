import streamlit as st
import pandas as pd
import joblib
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import re

# -------------------------------------------------------------------
# Konfigurasi Halaman dan Judul
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Komentar Instagram",
    page_icon="😊",
    layout="wide"
)

st.title("Analisis Sentimen Komentar Instagram 🇮🇩")
st.markdown("Aplikasi ini menggunakan model *Machine Learning* untuk memprediksi sentimen (positif atau negatif) dari teks komentar berbahasa Indonesia.")

# -------------------------------------------------------------------
# Fungsi-fungsi Preprocessing (Sama seperti di notebook)
# -------------------------------------------------------------------
# Inisialisasi Sastrawi Stemmer
try:
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
except Exception as e:
    st.error(f"Gagal menginisialisasi Sastrawi Stemmer: {e}")
    st.stop()


# Fungsi untuk membersihkan dan memproses teks
@st.cache_resource
def preprocess_text(text):
    """
    Fungsi ini membersihkan dan memproses teks input:
    1. Mengubah ke lowercase
    2. Menghapus angka
    3. Menghapus tanda baca
    4. Melakukan stemming
    """
    # 1. Mengubah ke lowercase
    text = text.lower()
    # 2. Menghapus angka
    text = re.sub(r"\d+", "", text)
    # 3. Menghapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 4. Melakukan stemming
    stemmed_text = stemmer.stem(text)
    
    return stemmed_text

# -------------------------------------------------------------------
# Memuat Model dan Vectorizer yang Telah Disimpan
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    """
    Memuat model dan TF-IDF vectorizer yang telah dilatih.
    """
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("File model atau vectorizer tidak ditemukan. Pastikan 'sentiment_model.pkl' dan 'tfidf_vectorizer.pkl' berada di folder yang sama.")
        return None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

# -------------------------------------------------------------------
# Tampilan Antarmuka Streamlit
# -------------------------------------------------------------------
if model and vectorizer:
    # Kolom untuk input dan hasil
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Masukkan Komentar untuk Dianalisis")
        # Text area untuk input pengguna
        user_input = st.text_area("Tulis atau tempel komentar di sini:", height=150, placeholder="Contoh: Filmnya bagus banget, saya suka sekali!")

        # Tombol untuk melakukan analisis
        if st.button("Analisis Sentimen", type="primary"):
            if user_input:
                # 1. Preprocess teks input
                cleaned_text = preprocess_text(user_input)
                
                # 2. Transformasi teks menggunakan TF-IDF vectorizer yang sudah di-fit
                vectorized_text = vectorizer.transform([cleaned_text])
                
                # 3. Lakukan prediksi menggunakan model
                prediction = model.predict(vectorized_text)
                prediction_proba = model.predict_proba(vectorized_text)
                
                # Tampilkan hasil
                sentiment = prediction[0]
                confidence_score = prediction_proba[0].max() * 100
                
                with col2:
                    st.subheader("Hasil Analisis")
                    if sentiment == 'positive':
                        st.markdown(f"**Sentimen: <span style='color:green;'>Positif 😊</span>**", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Sentimen: <span style='color:red;'>Negatif 😠</span>**", unsafe_allow_html=True)             
                    st.metric(label="Tingkat Keyakinan", value=f"{confidence_score:.2f}%")
                    
                    with st.expander("Lihat Detail Proses"):
                        st.write("**Teks Asli:**")
                        st.text(user_input)
                        st.write("**Teks Setelah Preprocessing (Stemming):**")
                        st.text(cleaned_text)

            else:
                st.warning("Mohon masukkan teks komentar terlebih dahulu.")
else:
    st.warning("Aplikasi tidak dapat berjalan karena model gagal dimuat.")

# Menambahkan sedikit informasi tentang proyek
st.markdown("---")