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
pip install opencv-python mediapipe face-recognition numpy
