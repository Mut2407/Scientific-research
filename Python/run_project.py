import cv2
import numpy as np
import time
import pyttsx3
from keras.models import load_model
from collections import deque, Counter
from play_emotion import play_playlist

# --- Cấu hình giọng nói ---
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 0.9)

# --- Nạp model ---
model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- Nhãn cảm xúc ---
labels_dict = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

labels_vn = {
    'angry': 'Tức giận', 'disgust': 'Ghê sợ', 'fear': 'Sợ hãi',
    'happy': 'Vui vẻ', 'neutral': 'Bình thường', 'sad': 'Buồn bã', 'surprise': 'Ngạc nhiên'
}

# --- Gán giá trị Valence–Arousal ---
emotion_va = {
    'angry': (-0.6, 0.8),
    'disgust': (-0.7, 0.6),
    'fear': (-0.8, 0.9),
    'happy': (0.8, 0.8),
    'neutral': (0.0, 0.3),
    'sad': (-0.7, 0.2),
    'surprise': (0.5, 0.9)
}

# --- Khởi tạo biến ---
video = cv2.VideoCapture(0)
emotion_history = deque(maxlen=30)  # lưu cảm xúc trong 30 khung hình (~10s)
current_emotion = None
last_change_time = time.time()
min_confidence = 0.6  # ngưỡng tin cậy tối thiểu
stable_seconds = 8    # yêu cầu cảm xúc mới ổn định ≥ 8 giây trước khi đổi nhạc

print("🧠 Hệ thống nhận diện cảm xúc nâng cao đang khởi động... (nhấn Q để thoát)")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)

        confidence = np.max(result)
        label = np.argmax(result, axis=1)[0]
        predicted_emotion = labels_dict[label]

        # --- Áp dụng ngưỡng tin cậy ---
        if confidence >= min_confidence:
            emotion_history.append(predicted_emotion)

        # --- Tính cảm xúc ổn định ---
        if len(emotion_history) >= 10:
            dominant_emotion = Counter(emotion_history).most_common(1)[0][0]
        else:
            dominant_emotion = predicted_emotion

        # --- Tính trung bình mức cảm xúc (Valence, Arousal) ---
        valence_vals = [emotion_va[e][0] for e in emotion_history if e in emotion_va]
        arousal_vals = [emotion_va[e][1] for e in emotion_history if e in emotion_va]
        avg_valence = np.mean(valence_vals) if valence_vals else 0
        avg_arousal = np.mean(arousal_vals) if arousal_vals else 0

        # --- Hiển thị lên khung hình ---
        cv2.putText(frame, f"{dominant_emotion} ({confidence:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"V:{avg_valence:.2f} A:{avg_arousal:.2f}", (x, y+h+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # --- Quy tắc đổi nhạc: chỉ khi cảm xúc khác và ổn định lâu ---
        current_time = time.time()
        if dominant_emotion != current_emotion:
            if current_time - last_change_time >= stable_seconds:
                current_emotion = dominant_emotion
                vn_name = labels_vn[current_emotion]
                print(f"🧩 Cảm xúc ổn định: {vn_name} (V={avg_valence:.2f}, A={avg_arousal:.2f})")
                engine.say(f"Tôi cảm nhận bạn đang {vn_name}")
                engine.runAndWait()

                if current_emotion in ["happy", "sad", "angry", "neutral"]:
                    play_playlist(current_emotion)
                    print(f"🎵 Đang phát nhạc phù hợp với cảm xúc: {vn_name}")

                last_change_time = current_time
        else:
            # reset thời gian nếu vẫn cùng cảm xúc
            last_change_time = current_time

    cv2.imshow("Emotion Detection - Smoothed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
