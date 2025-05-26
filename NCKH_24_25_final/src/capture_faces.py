import os
import cv2
import numpy as np
from architecture import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from database import connect_db, create_table, insert_encoding
from custom_mtcnn_ThongTin import CustomMTCNN 

required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights("./src/facenet_keras_weights.h5")
detector = CustomMTCNN() 
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def capture_and_save():
    ma_sv = input("Nhập mã sinh viên: ")
    ho_ten = input("Nhập họ tên: ")
    label = f"{ma_sv}_{ho_ten.replace(' ', '_')}"

    cap = cv2.VideoCapture(0)
    count = 0

    conn = connect_db()
    create_table(conn)

    while cap.isOpened() and count < 50:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        if results:
            x, y, w, h = results[0]['box']
            x, y = abs(x), abs(y)
            face = img_rgb[y:y+h, x:x+w]

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

            insert_encoding(conn, label, encode.tobytes())
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Capturing Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    conn.close()
    print("Hoàn tất ghi lại khuôn mặt.")

if __name__ == "__main__":
    capture_and_save()
