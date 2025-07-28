import os
import face_recognition
import pickle

DATASET_DIR = "faces"
encodings = []
names = []

print("[INFO] Mulai proses encoding wajah berdasarkan nama file...")

for filename in os.listdir(DATASET_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue  # Lewati file non-gambar

    image_path = os.path.join(DATASET_DIR, filename)

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"[ERROR] Gagal membuka {image_path}: {e}")
        continue

    face_locations = face_recognition.face_locations(image)
    if len(face_locations) != 1:
        print(f"[WARNING] {filename}: Ditemukan {len(face_locations)} wajah. Lewati.")
        continue

    face_encoding = face_recognition.face_encodings(image, face_locations)[0]

    # Ambil nama dari awal filename, misal 'apip_1.jpg' → 'apip'
    raw_name = os.path.splitext(filename)[0].split("_")[0].lower()

    encodings.append(face_encoding)
    names.append(raw_name)
    print(f"[INFO] Terekam {filename} sebagai '{raw_name}'")

# Simpan ke file pickle
data = {"encodings": encodings, "names": names}
with open("encodings_knn.pkl", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encoding selesai dan disimpan di 'encodings_knn.pkl'")
