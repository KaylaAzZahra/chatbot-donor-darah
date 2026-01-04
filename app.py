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
# LOAD & TRAIN MODEL
# ===============================
@st.cache_resource
def init_all():
    # TAMBAHKAN DUA BARIS INI:
    nltk.download('punkt')
    nltk.download('punkt_tab') # Penting untuk versi NLTK terbaru di server
    
    stemmer = LancasterStemmer()
    # ... sisa kode kamu yang lain ...

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

    words = sorted(set(stemmer.stem(w.lower()) for w in words if w != "?"))
    labels = sorted(labels)

    training, output = [], []
    for i, doc in enumerate(docs_x):
        bag = []
        stemmed = [stemmer.stem(w.lower()) for w in doc]
        for w in words:
            bag.append(1 if w in stemmed else 0)
        training.append(bag)
        output.append(labels.index(docs_y[i]))

    model = MLPClassifier(
        hidden_layer_sizes=(16, 16),
        max_iter=1000,
        random_state=42
    )
    model.fit(np.array(training), np.array(output))

    return words, labels, model, data, stemmer

words, labels, model, data, stemmer = init_model()

# ===============================
# FUNGSI RESPON BOT (FIX JADWAL)
# ===============================
def get_response(text):
    text_lower = text.lower()

    # HARD RULE biar "jadwal" PASTI kebaca
    if "jadwal" in text_lower or "hari ini" in text_lower:
        return (
            "Jadwal donor darah biasanya tersedia setiap hari kerja di kantor PMI. "
            "Untuk jadwal donor darah hari ini, silakan cek media sosial PMI setempat "
            "atau datang langsung ke Unit Donor Darah (UDD) terdekat."
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

    # THRESHOLD DITURUNKAN (PENTING)
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
    st.success(
        "**Kayla Az Zahra**\n\n"
        "Project UAS Machine Learning"
    )

    st.markdown("### ℹ️ Info Sistem")
    st.info(
        "Chatbot ini menggunakan algoritma Neural Network "
        "(Multi-layer Perceptron) dengan pendekatan Supervised Learning."
    )

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# ===============================
# HEADER
# ===============================
st.markdown(
    "<h1 style='text-align:center;color:#e63946;'>🩸 Sahabat Donor AI</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Asisten Cerdas Informasi Donor Darah</p>",
    unsafe_allow_html=True
)
st.caption(
    "<p style='text-align:center;'>Engine: Supervised Learning (MLP)</p>",
    unsafe_allow_html=True
)
st.divider()

# ===============================
# SESSION STATE
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya Sahabat Donor. Ada yang bisa saya bantu?"
        }
    ]

# ===============================
# CHAT UI MANUAL (KANAN–KIRI)
# ===============================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-end;margin:10px 0;">
            <div style="
                background:#ffe3e3;
                padding:10px 14px;
                border-radius:18px 18px 4px 18px;
                max-width:70%;
            ">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;justify-content:flex-start;margin:10px 0;">
            <div style="
                background:#f1f3f6;
                padding:10px 14px;
                border-radius:18px 18px 18px 4px;
                max-width:70%;
            ">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ===============================
# INPUT CHAT
# ===============================
prompt = st.chat_input("Tanya syarat, manfaat, jadwal donor darah...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    response = get_response(prompt)
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    st.rerun()