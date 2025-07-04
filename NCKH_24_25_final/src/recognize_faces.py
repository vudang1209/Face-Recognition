import os
import sys
import cv2
import numpy as np
import pandas as pd
import mtcnn
import pickle
from datetime import datetime
from architecture import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine

# Tắt GPU 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Định nghĩa hàm resource_path để lấy đúng đường dẫn khi chạy exe hoặc script
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # Khi chạy script, lấy từ thư mục cha của src
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)

required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights(resource_path(os.path.join("src", "facenet_keras_weights.h5")))
detector = mtcnn.MTCNN()
l2_normalizer = Normalizer('l2')

def load_encoding_dict(path=None):
    if path is None:
        path = resource_path(os.path.join("src", "encodings", "encodings.pkl"))
    else:
        path = resource_path(path)
    with open(path, 'rb') as f:
        return pickle.load(f)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    return img[y1:y2, x1:x2], (x1, y1), (x2, y2)

def recognize_and_log():
    encoding_dict = load_encoding_dict()
    cap = cv2.VideoCapture(0)
    attendance = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        for res in results:
            if res['confidence'] < 0.99:
                continue

            face, pt1, pt2 = get_face(img_rgb, res['box'])
            face = normalize(face)
            face = cv2.resize(face, required_shape)
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

            name = "unknown"
            min_dist = float("inf")
            similarity_score = 0.0
            
            for db_name, db_enc in encoding_dict.items():
                dist = cosine(db_enc, encode)
                if dist < 0.2 and dist < min_dist:
                    name = db_name
                    min_dist = dist
                    similarity_score = (1 - dist) * 100  

            if name != "unknown" and name not in [a[0] for a in attendance]:
                attendance.append((
                    name,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    round(similarity_score, 2)
                ))

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            label = f"{name} ({round(similarity_score, 2)}%)" if name != "unknown" else name
            cv2.rectangle(frame, pt1, pt2, color, 2)
            cv2.putText(frame, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Nhận diện khuôn mặt", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if attendance:
        df = pd.DataFrame(attendance, columns=["Họ tên", "Thời gian", "% Chính xác"])
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        print(f"✅ Lưu điểm danh vào: {filename}")

if __name__ == "__main__":
    recognize_and_log()