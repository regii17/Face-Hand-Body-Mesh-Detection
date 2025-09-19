# Face, Hand & Body Mesh Detection

Project ini adalah aplikasi **deteksi mesh wajah, tangan, dan tubuh** menggunakan kombinasi:
- [face_recognition](https://github.com/ageitgey/face_recognition) untuk landmark wajah,
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) untuk deteksi tangan,
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) untuk deteksi tubuh.

Output berupa **visualisasi real-time** dengan mesh (garis penghubung antar landmark) yang ditampilkan pada kamera.

---

## ✨ Fitur
- Deteksi wajah menggunakan `face_recognition`
- Menggambar **mesh wajah lengkap** (kontur wajah, alis, mata, bibir, hidung)
- Deteksi tangan kiri & kanan menggunakan MediaPipe
- Menggambar **mesh tangan** + detail ujung jari dengan label
- Deteksi tubuh menggunakan MediaPipe Pose
- Hanya menggambar **titik & garis tubuh utama** (tidak menggambar wajah & tangan dari pose agar tidak bertumpuk)
- Tampilan **FPS**, jumlah wajah, jumlah tangan, dan status tubuh
- Real-time menggunakan kamera laptop/webcam

---

## 📦 Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/username/face-hand-body-mesh.git
cd face-hand-body-mesh
```

### 2. Buat Virtual Environment (Opsional tapi direkomendasikan)
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
### 3. Install Dependencies
Pastikan sudah punya Python 3.8+
Lalu install requirements:
```bash
pip install -r requirements.txt
```
Atau manual:
```bash
pip install opencv-python mediapipe face-recognition numpy
```

⚠️ Untuk face_recognition, pastikan CMake dan dlib sudah terinstall di sistem.
Windows:
```bash
pip install cmake dlib
```
Ubuntu/Debian:
```bash
sudo apt-get install cmake libdlib-dev
```

## ▶️ Cara Menjalankan
```bash
python main.py
```

Tampilan window akan muncul:
- Mesh wajah berwarna cyan (biru muda)
- Mesh tangan hijau dengan garis biru, ujung jari merah dengan label
- Mesh tubuh kuning/magenta (tanpa wajah & tangan agar tidak bertumpuk)
- Informasi FPS, jumlah wajah, jumlah tangan, status tubuh ditampilkan di pojok kiri atas

## 📂 Struktur Project
```bash
face-hand-body-mesh/
│── main.py              # Main program (kamera + deteksi mesh)
│── requirements.txt     # Dependencies
│── README.md            # Dokumentasi
```
