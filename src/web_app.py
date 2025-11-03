# src/web_app.py - KODE LENGKAP DAN FINAL (Versi Aman & Siap Pakai)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from google import genai
from google.genai.errors import APIError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- BAGIAN 0: PENENTUAN PATH ABSOLUT ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'fruit_veg_cnn_model.h5')
LABELS_PATH = os.path.join(ROOT_DIR, 'models', 'class_labels.txt')
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'static', 'uploads')

# --- BAGIAN 1: KONFIGURASI GLOBAL ---
IMAGE_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Pastikan folder uploads bisa ditulis
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
try:
    os.chmod(UPLOAD_FOLDER, 0o777)
except Exception:
    pass  # kalau gagal ubah permission, biarkan saja di Windows

# --- BAGIAN 2 & 3: FUNGSI PEMBANGUN MODEL & MUAT MODEL ---
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

app = Flask(__name__,
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cnn_model = None
CLASS_LABELS = []
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Model ({os.path.basename(MODEL_PATH)}) atau Label ({os.path.basename(LABELS_PATH)}) belum ditemukan. Jalankan train.py dulu."
        )

    cnn_model = tf.keras.models.load_model(MODEL_PATH)

    with open(LABELS_PATH, 'r') as f:
        CLASS_LABELS = [line.strip() for line in f]

    print(f"✅ Model '{os.path.basename(MODEL_PATH)}' berhasil dimuat. Label: {CLASS_LABELS}")

except Exception as e:
    print(f"❌ Gagal memuat model/label. Error: {e}")

# --- BAGIAN 4: FUNGSI UTILITY & LLM GEMINI ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_llm_output(raw_text):
    cleaned_text = raw_text.replace('**', '<b>').replace('</b>', '</b>')
    cleaned_text = cleaned_text.replace('*', '').replace('•', '').replace('-', '').strip()
    for i in range(1, 6):
        cleaned_text = cleaned_text.replace(f"{i}. ", "")
    cleaned_text = cleaned_text.replace('\r\n\r\n', '</p><p>')
    cleaned_text = cleaned_text.replace('\n\n', '</p><p>')
    cleaned_text = cleaned_text.replace('\n', '<br>')
    if not cleaned_text.startswith('<p>'):
        cleaned_text = f"<p>{cleaned_text}"
    if not cleaned_text.endswith('</p>'):
        cleaned_text = f"{cleaned_text}</p>"
    cleaned_text = cleaned_text.replace('<p><br>', '<p>').replace('<br></p>', '</p>')
    return cleaned_text

def get_llm_info_gemini(fruit_veg_name):
    YOUR_GEMINI_API_KEY = "AIzaSyArpr6cR1l6VO7EiuYJSZ_PO3jKEm3zn-o"
    if not YOUR_GEMINI_API_KEY:
        return "[LLM ERROR] Kunci API belum diset."

    os.environ['GEMINI_API_KEY'] = YOUR_GEMINI_API_KEY
    try:
        client = genai.Client()
        prompt = (f"Berikan analisis lengkap untuk buah **{fruit_veg_name}**. "
                  "Gunakan format paragraf dan tebal untuk nama buah. "
                  "Bagian: Ringkasan, Manfaat, dan Vitamin.")
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return clean_llm_output(response.text)
    except Exception as e:
        return f"[LLM ERROR] {e}"

def predict_and_analyze(img_path):
    if cnn_model is None or not CLASS_LABELS:
        return None, None, None, "Model atau Label belum dimuat."

    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = cnn_model.predict(img_array, verbose=0)
        idx = np.argmax(predictions[0])
        confidence = f"{predictions[0][idx] * 100:.2f}"
        label = CLASS_LABELS[idx]
        llm_response = get_llm_info_gemini(label)
        return label, confidence, llm_response, None
    except Exception as e:
        return None, None, None, f"Kesalahan prediksi: {e}"

# --- BAGIAN 5: RUTE FLASK (Antarmuka Web) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Tidak ada file di request.')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Tidak ada file yang dipilih.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # bersihkan folder lama tanpa ganggu permission
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
                except Exception:
                    pass

            # Simpan file baru
            file.save(filepath)

            prediction, confidence, llm_response, error = predict_and_analyze(filepath)
            return render_template('index.html',
                                   filename=filename,
                                   prediction=prediction,
                                   confidence=confidence,
                                   llm_response=llm_response,
                                   error=error)
        else:
            return render_template('index.html', error='Format file tidak diizinkan.')

    return render_template('index.html')


# --- BAGIAN 6: LAMAN TENTANG ---
@app.route('/tentang')
def tentang():
    return render_template('tentang.html')


if __name__ == '__main__':
    print("\n[INFO] Jalankan: python src/train.py sebelum ini.")
    print("Aplikasi web berjalan di http://127.0.0.1:5000")
    app.run(debug=True)
