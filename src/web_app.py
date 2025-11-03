# src/web_app.py - KODE LENGKAP DALAM SATU FILE (Perbaikan Path & Sinkronisasi Buah)

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

# Path Model dan Label harus sinkron dengan yang disimpan oleh train.py
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'fruit_veg_cnn_model.h5')
LABELS_PATH = os.path.join(ROOT_DIR, 'models', 'class_labels.txt')
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'static', 'uploads') 

# --- BAGIAN 1: KONFIGURASI GLOBAL ---
IMAGE_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- BAGIAN 2: FUNGSI PEMBANGUN MODEL (hanya untuk referensi) ---

def build_cnn_model(input_shape, num_classes):
    """Membangun arsitektur Sequential CNN."""
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

# --- BAGIAN 3: KONFIGURASI FLASK & MUAT MODEL ---

app = Flask(__name__, 
            template_folder=os.path.join(ROOT_DIR, 'templates'),
            static_folder=os.path.join(ROOT_DIR, 'static'))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cnn_model = None
CLASS_LABELS = []
try:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        # Pesan error akan menampilkan nama file yang hilang
        raise FileNotFoundError(f"Model ({os.path.basename(MODEL_PATH)}) atau Label ({os.path.basename(LABELS_PATH)}) belum ditemukan. Jalankan train.py dulu.")
    
    # 1. Muat Model
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Muat Label Kelas secara Dinamis
    with open(LABELS_PATH, 'r') as f:
        CLASS_LABELS = [line.strip() for line in f]

    print(f"✅ Model '{os.path.basename(MODEL_PATH)}' berhasil dimuat. Label: {CLASS_LABELS}")

except Exception as e:
    print(f"❌ Gagal memuat model/label. Error: {e}")

# --- BAGIAN 4: FUNGSI UTILITY & LLM GEMINI ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_llm_info_gemini(fruit_veg_name):
    """Memanggil Gemini API untuk mendapatkan manfaat dan nutrisi."""
    
    # *** LANGKAH ANTI-GAGAL: SISIPKAN KUNCI API ANDA DI BAWAH INI ***
    # CATATAN: Kunci di bawah ini hanya placeholder.
    YOUR_GEMINI_API_KEY = "AIzaSyArpr6cR1l6VO7EiuYJSZ_PO3jKEm3zn-o" 
    
    if YOUR_GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not YOUR_GEMINI_API_KEY:
         return "[LLM ERROR] Kunci API belum diganti di web_app.py. Harap segera perbaiki."

    # Menyetel kunci secara eksplisit untuk Python sesi ini
    os.environ['GEMINI_API_KEY'] = YOUR_GEMINI_API_KEY

    try:
        client = genai.Client() 
        
        prompt = (f"Berikan rangkuman manfaat kesehatan dan 3 vitamin terpenting dari **{fruit_veg_name}**. "
                  "Fokus pada nutrisi yang relevan untuk buah-buahan. Gunakan format daftar poin. Jawab dalam Bahasa Indonesia.")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
        
    except APIError as e:
        return f"[LLM API ERROR] Kesalahan API Gemini: {e}"
    except Exception as e:
        return f"[LLM ERROR] Kesalahan umum: {e}"


def predict_and_analyze(img_path):
    """Melakukan prediksi CNN dan analisis LLM."""
    if cnn_model is None or not CLASS_LABELS:
        return None, None, None, "Model atau Label belum dimuat. Pastikan train.py sudah dijalankan!"

    try:
        # 1. Preprocessing CNN
        print(f"\n[DEBUG] 1. Memuat dan mengolah gambar: {img_path}")
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 

        # 2. Prediksi CNN
        print(" [DEBUG] 2. Memulai prediksi CNN...")
        predictions = cnn_model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]
        predicted_label = CLASS_LABELS[predicted_index]
        
        confidence_percent = f"{confidence*100:.2f}"

        # 3. Analisis LLM
        print(f" [DEBUG] 3. Memulai panggilan Gemini untuk {predicted_label}...")
        llm_response = get_llm_info_gemini(predicted_label)
        
        print(f" [DEBUG] 4. Analisis selesai.")
        
        return predicted_label, confidence_percent, llm_response, None

    except Exception as e:
        print(f" [DEBUG] ERROR PRED/LLM: {e}")
        return None, None, None, f"Terjadi kesalahan saat klasifikasi: {e}"


# --- BAGIAN 5: RUTE FLASK (Antarmuka Web) ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Tidak ada file di request.')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='Tidak ada file yang dipilih.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Hapus file lama di folder uploads
            for f in os.listdir(app.config['UPLOAD_FOLDER']):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))

            file.save(filepath)

            # Lakukan prediksi dan analisis
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

if __name__ == '__main__':
    print("\n[INFO] 1. Pastikan folder data_buah/training HANYA berisi folder buah.")
    print("[INFO] 2. Jalankan 'python src/train.py' DULU sebelum ini.")
    print("Aplikasi web berjalan di http://127.0.0.1:5000")
    app.run(debug=True)