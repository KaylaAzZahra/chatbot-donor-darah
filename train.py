import json
import numpy as np
import nltk
import pickle
import ssl
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# --- FIX DOWNLOAD NLTK ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("Mengecek data bahasa...")
nltk.download('punkt')
nltk.download('punkt_tab')
# -------------------------

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# 1. Load Data
try:
    with open('intents.json') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: File 'intents.json' tidak ditemukan di folder yang sama!")
    exit()

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

print("Sedang memproses data (Stemming Bahasa Indonesia)...")

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenisasi
        w = nltk.word_tokenize(pattern.lower())
        # Stemming (Nilai Tambah UAS)
        w = [stemmer.stem(i) for i in w if i not in ignore_words]
        
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Simpan metadata untuk chat.py
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 2. Persiapan Data Training (Bag of Words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

training = np.array(training, dtype=object)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# 3. Arsitektur Model Deep Learning (Nilai Tambah UAS)
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training
print("Memulai Training Model...")
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Simpan Model
model.save('chatbot_model.h5')
print("\n" + "="*30)
print("TRAINING SELESAI!")
print("File berikut telah dibuat:")
print("1. chatbot_model.h5 (Otak AI)")
print("2. words.pkl (Daftar Kata)")
print("3. classes.pkl (Daftar Kategori)")
print("="*30)