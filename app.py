import streamlit as st
import json
import numpy as np
import random
import nltk
from nltk.stem import LancasterStemmer
from sklearn.neural_network import MLPClassifier

# ===============================
# CONFIG HALAMAN
# ===============================
st.set_page_config(
    page_title="Sahabat Donor AI",
    page_icon="🩸",
    layout="centered"
)

# ===============================
# LOAD & TRAIN MODEL (FIXED NAMES)
# ===============================
@st.cache_resource
def init_model(): # Nama fungsi diseragamkan jadi init_model
    # Downloader untuk server Cloud
    nltk.download('punkt')
    nltk.download('punkt_tab') 
    
    stemmer = LancasterStemmer()

    # Membuka dataset intent sesuai ketentuan UAS 
    with open("intents.json", encoding="utf-8") as f:
        data = json.load(f)

    words, labels, docs_x, docs_y = [], [], [], []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            # Preprocessing: Tokenization 
            tokens = nltk.word_tokenize(pattern)
            words.extend(tokens)
            docs_x.append(tokens)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # Preprocessing: Stemming & Lowercase 
    words = sorted(list(set(stemmer.stem(w.lower()) for w in words if w != "?")))
    labels = sorted(labels)

    training, output = [], []
    for i, doc in enumerate(docs_x):
        bag = []
        stemmed = [stemmer.stem(w.lower()) for w in doc]
        # Feature Extraction: Bag of Words 
        for w in words:
            bag.append(1 if w in stemmed else 0)
        training.append(bag)
        output.append(labels.index(docs_y[i]))

    # Model: Supervised Learning (MLP) 
    model = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        max_iter=1000,
        random_state=42
    )
    model.fit(np.array(training), np.array(output))

    return words, labels, model, data, stemmer

# Memanggil fungsi yang benar
words, labels, model, data, stemmer = init_model()

# ===============================
# FUNGSI RESPON BOT (FIX JADWAL)
# ===============================
def get_response(text):
    text_lower = text.lower()

    # Rule untuk akurasi jadwal
    if "jadwal" in text_lower or "hari ini" in text_lower:
        return (
            "Jadwal donor darah biasanya tersedia setiap hari kerja di kantor PMI. "
            "Untuk jadwal donor darah hari ini, silakan cek media sosial PMI setempat."
        )

    bag = [0] * len(words)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(w.lower()) for w in tokens]

    for token in tokens:
        for i, w in enumerate(words):
            if w == token:
                bag[i] = 1

    probs = model.predict_proba([bag])[0]
    idx = np.argmax(probs)

    # Threshold klasifikasi intent [cite: 5]
    if probs[idx] > 0.35:
        tag = labels[idx]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    return "Maaf, saya hanya bisa membantu seputar donor darah 🩸"

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("### 👩‍💻 Developer")
    st.success(f"**Kayla Az Zahra**\n\nProject UAS Machine Learning")

    st.markdown("### ℹ️ Info Sistem")
    st.info("Chatbot ini menggunakan Supervised Learning (MLP) sesuai kriteria UAS.")

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align:center;color:#e63946;'>🩸 Sahabat Donor AI</h1>", unsafe_allow_html=True)
st.divider()

# ===============================
# SESSION STATE & UI
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya Sahabat Donor. Ada yang bisa saya bantu?"}]

for msg in st.session_state.messages:
    align = "flex-end" if msg["role"] == "user" else "flex-start"
    bg = "#ffe3e3" if msg["role"] == "user" else "#f1f3f6"
    radius = "18px 18px 4px 18px" if msg["role"] == "user" else "18px 18px 18px 4px"
    
    st.markdown(f"""
    <div style="display:flex;justify-content:{align};margin:10px 0;">
        <div style="background:{bg};padding:10px 14px;border-radius:{radius};max-width:70%;">
            {msg["content"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

# INPUT CHAT
if prompt := st.chat_input("Tanya sesuatu..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    res = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": res})
    st.rerun()