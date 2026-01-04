import streamlit as st
from groq import Groq

# 1. KONFIGURASI API GROQ
# Masukkan API Key kamu di sini
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# 2. UI CUSTOM CSS (STRICT THEME & CHAT BUBBLES)
st.set_page_config(page_title="Sahabat Donor AI", page_icon="🩸", layout="centered")

st.markdown("""
    <style>
    /* Background Dasar */
    .stApp { background-color: #f7f9fb; }
    
    /* Header Box Merah */
    .main-header {
        background: linear-gradient(135deg, #e53935 0%, #b71c1c 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .main-header h1 { color: white !important; font-size: 32px !important; margin-bottom: 5px; }
    .main-header p { font-size: 16px; opacity: 0.9; }

    /* Container Chat */
    .chat-container { width: 100%; display: flex; flex-direction: column; gap: 15px; }

    /* Bubble Chat USER (RATA KANAN) */
    .user-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        width: 100%;
        margin-bottom: 10px;
    }
    .user-bubble {
        background-color: #d32f2f;
        color: white;
        padding: 12px 18px;
        border-radius: 20px 20px 0px 20px;
        width: fit-content;
        max-width: 80%;
        box-shadow: 0 4px 10px rgba(211, 47, 47, 0.2);
    }

    /* Bubble Chat BOT (RATA KIRI) */
    .bot-wrapper {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        width: 100%;
        margin-bottom: 10px;
    }
    .bot-bubble {
        background-color: white;
        color: #333;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 0px;
        width: fit-content;
        max-width: 80%;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }

    /* Label Nama & Ikon */
    .label { font-size: 12px; font-weight: bold; margin-bottom: 4px; }
    .label-user { color: #d32f2f; text-align: right; }
    .label-bot { color: #555; text-align: left; }
    </style>
    
    <div class="main-header">
        <h1>🩸 SAHABAT DONOR AI</h1>
        <p>Asisten Khusus Informasi Donor Darah</p>
    </div>
    """, unsafe_allow_html=True)

# 3. INSTRUKSI SISTEM (DIPERKETAT AGAR TIDAK JAWAB LUAR TOPIK)
# Ini adalah kunci agar AI menolak pertanyaan Xiaomi/Jam/Lainnya
system_prompt = {
    "role": "system",
    "content": """
    STRICT RULES / ATURAN KETAT:
    1. Nama kamu adalah 'Sahabat Donor'. Kamu adalah asisten KHUSUS donor darah.
    2. Kamu HANYA boleh menjawab pertanyaan seputar donor darah (syarat, manfaat, prosedur, dll).
    3. Jika user bertanya hal di luar donor darah (seperti: gadget, HP Xiaomi, waktu/jam, cuaca, masak, dll), kamu DILARANG memberikan informasi tersebut.
    4. Respon wajib jika di luar topik: "Maaf Kak, sebagai Sahabat Donor, saya hanya bisa membantu menjawab pertanyaan seputar donor darah. Silakan tanya hal terkait donor darah ya! 😊"
    5. Gunakan bahasa Indonesia yang ramah dan sopan.
    """
}

# 4. SIDEBAR INFORMASI
with st.sidebar:
    st.markdown("### 🏥 **Informasi Sistem**")
    st.info("Chatbot ini menggunakan model **Llama-3 (70B)** yang sudah dikunci untuk topik Donor Darah.")
    if st.button("Hapus Riwayat Chat"):
        st.session_state.messages = [system_prompt]
        st.rerun()
    st.divider()
    st.write("📌 **Developer:** Kayla Az Zahra")
    st.write("🎓 **Project:** UAS Machine Learning")

# Logika Chat Session
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Tampilkan Riwayat Chat
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''<div class="user-wrapper"><span class="label label-user">Kamu 👤</span><div class="user-bubble">{message["content"]}</div></div>''', unsafe_allow_html=True)
    elif message["role"] == "assistant":
        st.markdown(f'''<div class="bot-wrapper"><span class="label label-bot">🤖 Sahabat Donor</span><div class="bot-bubble">{message["content"]}</div></div>''', unsafe_allow_html=True)

# 5. INPUT USER & RESPONS AI
if prompt := st.chat_input("Tanya soal donor darah..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Cek jika pesan terakhir dari user, maka minta jawaban AI
if st.session_state.messages[-1]["role"] == "user":
    with st.spinner("Sedang memverifikasi informasi..."):
        try:
            chat_completion = client.chat.completions.create(
                messages=st.session_state.messages,
                model="llama-3.3-70b-versatile",
                temperature=0.3, # Temperature rendah agar AI lebih patuh/tidak ngaco
            )
            response_text = chat_completion.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()
        except Exception as e:
            st.error(f"Gagal terhubung ke AI: {e}")