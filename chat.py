import json
import numpy as np
import nltk
import pickle
import random
import ssl
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model

# --- FIX SSL (Sama seperti di train.py) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Pastikan punkt sudah ada
nltk.download('punkt')
nltk.download('punkt_tab')

# 1. Load Model dan Data Pendukung
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_up_sentence(sentence):
    # Tokenisasi & Stemming input user
    sentence_words = nltk.word_tokenize(sentence.lower())
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    # Ubah kalimat jadi angka (Bag of Words)
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

print("\n" + "="*40)
print("CHATBOT LAYANAN DONOR DARAH SIAP!")
print("(Ketik 'keluar' untuk berhenti)")
print("="*40)

while True:
    message = input("Anda: ")
    if message.lower() == "keluar":
        print("Bot: Terima kasih, semoga harimu menyenangkan!")
        break
    
    # 2. Prediksi Intent
    p = bow(message, words)
    res = model.predict(np.array([p]), verbose=0)[0]
    
    # Ambil hasil dengan probabilitas di atas threshold
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 3. Berikan Respon
    if results:
        tag = classes[results[0][0]]
        for i in intents['intents']:
            if i['tag'] == tag:
                print("Bot: " + random.choice(i['responses']))
    else:
        print("Bot: Maaf, saya kurang mengerti. Bisa ditanyakan dengan kalimat lain?")