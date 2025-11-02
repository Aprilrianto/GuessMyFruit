# src/train.py - KODE LENGKAP DALAM SATU FILE (Perbaikan Final)

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- BAGIAN 0: PENENTUAN PATH ABSOLUT ---

# Menghitung path absolut untuk ROOT proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.dirname(BASE_DIR) 

# Definisikan semua path absolut
DATA_DIR = os.path.join(ROOT_DIR, 'data_buah') 
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_veg_cnn_model.h5')
LABELS_PATH = os.path.join(MODEL_DIR, 'class_labels.txt') # <-- TAMBAHKAN PATH LABEL INI

# --- BAGIAN 1: KONFIGURASI DAN PEMUATAN DATA ---

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
INPUT_SHAPE = IMAGE_SIZE + (3,)


def get_data_generators(data_dir=DATA_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """Memuat data training dan validation."""
    
    print(f"DEBUG: Mencari data di: {data_dir}")
    print("Mempersiapkan Data Generators...")
    
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, horizontal_flip=True, fill_mode='nearest',
        width_shift_range=0.2, height_shift_range=0.2
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'training'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'validation'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = train_generator.num_classes
    class_labels = list(train_generator.class_indices.keys())
    
    return train_generator, validation_generator, num_classes, class_labels

# --- BAGIAN 2: ARSITEKTUR CNN (Tidak berubah) ---
def build_cnn_model(input_shape, num_classes):
    """Membangun arsitektur Sequential CNN."""
    print("Membangun Arsitektur CNN...")
    
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
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- BAGIAN 3: PELATIHAN MODEL & PENYIMPANAN LABEL ---

def save_labels(labels):
    """Fungsi BARU: Menyimpan daftar label kelas ke file teks."""
    # Pastikan direktori models ada
    if not os.path.exists(MODEL_DIR): 
        os.makedirs(MODEL_DIR)
        
    with open(LABELS_PATH, 'w') as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"✅ [LABEL] Label kelas disimpan di: {LABELS_PATH}")


def train_and_save_model(model, train_gen, val_gen):
    """Melatih model dan menyimpannya ke folder models/."""
    
    if not os.path.exists(MODEL_DIR): 
        os.makedirs(MODEL_DIR)
        
    EPOCHS = 20
    print(f"\n[PELATIHAN] Memulai Pelatihan Model selama {EPOCHS} Epochs...")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
    ]
    
    model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        callbacks=callbacks
    )
    
    print(f"\n✅ Model terbaik berhasil disimpan di: {MODEL_PATH}")

if __name__ == '__main__':
    print("--- MEMULAI PROSES PELATIHAN MODEL CNN (Path Dibenarkan) ---")
    
    # 1. Muat Data
    train_generator, validation_generator, NUM_CLASSES, CLASS_LABELS = get_data_generators()
    
    if NUM_CLASSES < 2:
        print("❌ ERROR: Minimal diperlukan 2 kelas buah untuk klasifikasi. Harap cek folder data.")
    else:
        # 2. Bangun Model
        cnn_model = build_cnn_model(INPUT_SHAPE, NUM_CLASSES)
        
        # 3. Latih Model
        train_and_save_model(cnn_model, train_generator, validation_generator)

        # 4. SIMPAN LABEL UNTUK WEB_APP (INI YANG HILANG)
        save_labels(CLASS_LABELS)
        
        print("\n✅ Proses Train Selesai. Lanjutkan menjalankan web_app.py")