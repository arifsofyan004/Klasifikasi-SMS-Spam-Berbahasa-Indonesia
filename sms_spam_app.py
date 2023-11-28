import streamlit as st
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle

# Unduh data yang diperlukan
nltk.download("stopwords")
nltk.download("punkt")

# Inisialisasi factory stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk langkah stemming bahasa Indonesia
def stemming(text):
    # Melakukan proses stemming pada teks
    return stemmer.stem(text)

# Fungsi untuk mengubah teks SMS
def transform_text(text):
    # 1. Mengubah teks menjadi huruf kecil
    text = text.lower()

    # 2. Tokenisasi teks dan menghapus karakter non-alphanumeric
    tokens = [i for i in nltk.word_tokenize(text) if i.isalnum()]

    # 3. Menghapus stopwords (kata-kata umum) dalam bahasa Indonesia dan tanda baca
    tokens = [i for i in tokens if i not in stopwords.words('indonesian') + list(string.punctuation)]

    # 4. Stemming (menghilangkan imbuhan) dari kata-kata dalam teks
    tokens = [stemming(i) for i in tokens]

    # 5. Menggabungkan token-token yang sudah diproses menjadi teks kembali
    return " ".join(tokens)

# Load model dari file
with open('/content/model-sms_spam.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit setup
st.set_page_config(
    page_title="Deteksi SMS Spam",
    page_icon="/content/logoikmi.jpg",
    layout="wide"
)

# Warna Latar Belakang
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: #f0f0f0;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Warna Header
st.markdown(
    f"""
    <style>
    .stApp header {{
        background-color: #0075b8;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Logo STMIK IKMI Cirebon (ditempatkan di header)
st.image("/content/logoikmi.jpg", use_column_width=False, width=200, caption="STMIK IKMI Cirebon")

# Tulisan "Aplikasi Deteksi SMS Spam" di sebelah kiri
st.markdown("<div style='text-align: left;'>"
         "<h1 style='color: #0075b8;'>Aplikasi Deteksi SMS Spam</h1>"
         "</div>", unsafe_allow_html=True)  # Menambahkan tulisan dengan HTML

# Input teks SMS
sms_input = st.text_area("Masukkan SMS yang ingin Anda deteksi:")

if st.button("Deteksi"):
    # Preprocess user input (lowercasing, stemming, stopwords)
    text = transform_text(sms_input)
    # Perform the prediction
    prediction = model.predict([text])[0]

    # Display the result
    st.subheader('Hasil Deteksi:')
    if prediction == 1:
        st.error('Ini adalah SMS Spam!')
    else:
        st.success('Ini adalah SMS Normal.')

    # Add a confidence score (probability) if your model provides it
    if hasattr(model, 'predict_proba'):
        confidence_score = model.predict_proba([text])[:, 1][0]
        st.write(f'Confidence Score (Probability): {confidence_score:.2%}')

# Tombol "Reset" untuk mengosongkan input teks
if st.button("Reset"):
    sms_input = ""

# Footer
st.markdown("<div style='background-color: #0075b8; color: white; padding: 10px; text-align: center;'>"
            "<p style='font-size: 18px;'>STMIK IKMI CIREBON</p>"
            "<p style='font-size: 14px;'>@TeknikInformatika | arifsofyan004@gmail.com</p>"
            "<p style='font-size: 12px;'>Copyright © 2023 @arf.sof</p>"
            "</div>", unsafe_allow_html=True)
