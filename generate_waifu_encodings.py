import os
import face_recognition
import pickle
import json
from tqdm import tqdm
from PIL import UnidentifiedImageError

WAIFU_DIR = "waifus"
ENCODINGS_FILE = "waifu_encodings.pkl"
METADATA_FILE = "waifu_metadata.json"

waifu_data = []
metadata_list = []

print("🚀 Memproses waifu...")

# Telusuri folder waifus secara rekursif
for root, dirs, files in os.walk(WAIFU_DIR):
    for filename in tqdm(sorted(files), desc="🚀 Memproses waifu", unit="file"):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(root, filename)

        # Lewati file terlalu kecil (<10KB)
        if os.path.getsize(image_path) < 10 * 1024:
            print(f"⚠️ File too small, skipped: {image_path}")
            continue

        try:
            image = face_recognition.load_image_file(image_path)
        except UnidentifiedImageError:
            print(f"❌ Unidentified image file: {image_path}")
            continue
        except Exception as e:
            print(f"❌ Error reading {image_path}: {e}")
            continue

        # Deteksi wajah
        face_locations = face_recognition.face_locations(image)
        if not face_locations:
            print(f"❌ No face found in {image_path}")
            continue

        # Ambil encoding wajah
        encodings = face_recognition.face_encodings(image, face_locations)
        if not encodings:
            print(f"❌ No face encoding in {image_path}")
            continue

        encoding = encodings[0]
        folder_name = os.path.basename(root)
        label = f"{folder_name}/{filename}"

        # Simpan data encoding dan metadata
        waifu_data.append({
            "label": label,
            "encoding": encoding
        })

        metadata_list.append({
            "label": label,
            "name": folder_name,
            "filename": filename,
            "image_path": image_path
        })

# Simpan ke file
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(waifu_data, f)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata_list, f, indent=2, ensure_ascii=False)

print(f"\n✅ Total waifu yang berhasil diproses: {len(waifu_data)}")
print(f"📦 Encoding disimpan di: {ENCODINGS_FILE}")
print(f"📝 Metadata disimpan di: {METADATA_FILE}")
