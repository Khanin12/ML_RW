#!/bin/bash

# --- Bagian Konfigurasi ---
# Nama folder Virtual Environment Anda (ganti jika berbeda)
VENV_NAME="venv" 
# Nama file Python utama Anda
APP_FILE="app.py" 
# Port default (ganti jika Anda perlu port lain)
PORT=5000

# --- Bagian Otomasi ---

# 1. Pastikan Virtual Environment ada dan aktif
if [ -d "$VENV_NAME" ]; then
    echo "✅ Mengaktifkan Virtual Environment..."
    source "$VENV_NAME/bin/activate"
else
    echo "❌ Virtual Environment ($VENV_NAME) tidak ditemukan."
    echo "Silakan buat dengan 'python3 -m venv $VENV_NAME' dan coba lagi."
    exit 1
fi

# 2. Instal dependensi (hanya jika requirements.txt ada)
if [ -f "requirements.txt" ]; then
    echo "📦 Menginstal/Memperbarui dependensi..."
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt tidak ditemukan. Melewatkan instalasi dependensi."
fi

# 3. Jalankan Aplikasi Flask

# Gunakan Gunicorn untuk Production (sesuai Procfile)
# web: gunicorn app:app
echo "▶️ Menjalankan aplikasi menggunakan Gunicorn..."
gunicorn $APP_FILE:app -b 0.0.0.0:$PORT

# --- Jika Anda ingin menggunakan server development Flask: ---
# echo "▶️ Menjalankan aplikasi menggunakan server development Flask..."
# export FLASK_APP=$APP_FILE
# export FLASK_ENV=development
# flask run --host=0.0.0.0 --port=$PORT

echo "Stoppped."