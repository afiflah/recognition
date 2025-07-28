import cv2
import face_recognition
import pickle
import numpy as np
import os

def load_encodings(file_path):
    """Load encoding file (user atau waifu) dari pickle."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def match_user(user_encodings, current_encoding):
    """Cocokkan wajah pengguna ke model encoding pengguna."""
    distances = face_recognition.face_distance(user_encodings["encodings"], current_encoding)
    best_idx = np.argmin(distances)
    return user_encodings["names"][best_idx], distances[best_idx]

def match_waifu(waifu_data, current_encoding):
    """Cocokkan wajah pengguna ke encoding waifu."""
    enc_list = [w["encoding"] for w in waifu_data]
    labels = [w["label"] for w in waifu_data]
    distances = face_recognition.face_distance(enc_list, current_encoding)
    best_idx = np.argmin(distances)
    return labels[best_idx], distances[best_idx]

def main():
    user_data = load_encodings("encodings_knn.pkl")
    waifu_data = load_encodings("waifu_encodings.pkl")
    print(f"✅ {len(waifu_data)} waifu berhasil dimuat.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera tidak tersedia.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Gagal membaca frame dari kamera.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            try:
                user_name, user_dist = match_user(user_data, face_encoding)
                waifu_label, waifu_dist = match_waifu(waifu_data, face_encoding)

                # Format nama file waifu
                waifu_name = os.path.basename(waifu_label)
                confidence = (1 - waifu_dist) * 100

                # Gambar kotak wajah
                top *= 2; right *= 2; bottom *= 2; left *= 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Teks user di atas kotak
                user_text = f"{user_name}"
                cv2.putText(frame, user_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Teks waifu di bawah kotak
                waifu_text = f"Waifu: {waifu_name} ({confidence:.2f}%)"
                cv2.putText(frame, waifu_text, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            except Exception as e:
                print(f"❌ Error saat mencocokkan wajah: {e}")

        cv2.imshow("🧠 Real-time Matcher", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
