import cv2
import face_recognition
import pickle
import numpy as np
import os
from datetime import datetime

# ==== Konstanta ====
ENCODING_FILE = "encodings_knn.pkl"
UNKNOWN_DIR = "unknown_faces"
ALIAS_MAPPING = {
    "diri–sendiri": "Fathir",
    "diri sendiri": "Fathir",
    "sendiri": "Fathir"
}

# ==== Fungsi Load ====
def load_known_faces(path):
    if not os.path.exists(path):
        print(f"[❌] File encoding tidak ditemukan: {path}")
        return [], []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("encodings", []), data.get("names", [])

# ==== Nama Resolver ====
def resolve_name(raw):
    name = raw.strip().lower()
    return ALIAS_MAPPING.get(name, name.capitalize())

# ==== Inisialisasi ====
known_encodings, known_names = load_known_faces(ENCODING_FILE)
print(f"📥 Memuat wajah dari file...\n✅ Total wajah dikenal: {len(known_encodings)}")
os.makedirs(UNKNOWN_DIR, exist_ok=True)

if len(known_encodings) == 0:
    print("⚠️ Tidak ada data wajah dikenal. Jalankan dulu script encoding.")
    exit()

video_capture = cv2.VideoCapture(0)
print("[INFO] Mulai deteksi wajah... Tekan 'q' untuk keluar.")

# ==== Loop Kamera ====
while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[❌] Gagal mengambil frame dari kamera.")
        break

    # Resize & konversi
    small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        name = "Orang Tidak Dikenal"
        similarity = None

        if len(distances) > 0:
            best_index = np.argmin(distances)
            if distances[best_index] < 0.5:
                matched = known_names[best_index]
                similarity = round((1 - distances[best_index]) * 100, 2)
                name = resolve_name(matched)

        # Lokasi kotak (kembali ke ukuran asli)
        top, right, bottom, left = [v * 4 for v in location]

        is_known = name != "Orang Tidak Dikenal"
        color_box = (0, 255, 0) if is_known else (0, 0, 255)
        text_color = (0, 0, 0) if is_known else (255, 255, 255)
        label = f"{name} ({similarity}%)" if similarity else name

        # Kotak dan teks
        cv2.rectangle(frame, (left, top), (right, bottom), color_box, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color_box, cv2.FILLED)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Simpan wajah tak dikenal
        if not is_known:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop = frame[top:bottom, left:right]
            if crop.size > 0:
                filename = os.path.join(UNKNOWN_DIR, f"unknown_{timestamp}.jpg")
                cv2.imwrite(filename, crop)

    # Tampilkan
    cv2.imshow("Deteksi Wajah", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==== Cleanup ====
video_capture.release()
cv2.destroyAllWindows()
