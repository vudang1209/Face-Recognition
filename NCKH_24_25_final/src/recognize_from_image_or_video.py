import os
import sys
import cv2
import numpy as np
import pickle
from architecture import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine
from custom_mtcnn_ThongTin import CustomMTCNN

# Định nghĩa hàm resource_path để lấy đúng đường dẫn khi chạy exe hoặc script
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # Khi chạy script, lấy từ thư mục cha của src
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)

required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights(resource_path(os.path.join("src", "facenet_keras_weights.h5")))
detector = CustomMTCNN()
l2_normalizer = Normalizer('l2')

with open(resource_path(os.path.join("src", "encodings", "encodings.pkl")), 'rb') as f:
    encoding_dict = pickle.load(f)

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def recognize_from_image(image_path, encoding_dict_override=None):
    enc_dict = encoding_dict_override if encoding_dict_override is not None else encoding_dict
    img = cv2.imread(image_path)
    if img is None:
        print("Không thể đọc ảnh.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    for res in results:
        if res['confidence'] < 0.99:
            continue

        x, y, w, h = res['box']
        x, y = abs(x), abs(y)
        face = img_rgb[y:y+h, x:x+w]

        face = normalize(face)
        face = cv2.resize(face, required_shape)
        encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

        name = "unknown"
        min_dist = float("inf")
        for db_name, db_enc in enc_dict.items():
            dist = cosine(db_enc, encode)
            if dist < 0.2 and dist < min_dist:
                name = db_name
                min_dist = dist

        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 5)
        cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow("Nhan Dien khuon mat MTCNN va Facenet", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def recognize_from_video(video_path=None, encoding_dict_override=None):
    enc_dict = encoding_dict_override if encoding_dict_override is not None else encoding_dict
    cap = cv2.VideoCapture(0 if video_path is None else video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        for res in results:
            if res['confidence'] < 0.99:
                continue

            x, y, w, h = res['box']
            x, y = abs(x), abs(y)
            face = img_rgb[y:y+h, x:x+w]

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

            name = "unknown"
            min_dist = float("inf")
            for db_name, db_enc in enc_dict.items():
                dist = cosine(db_enc, encode)
                if dist < 0.2 and dist < min_dist:
                    name = db_name
                    min_dist = dist

            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        scale_percent = 50
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Nhận diện từ video", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

__all__ = ["recognize_from_image", "recognize_from_video"]

if __name__ == "__main__":
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    Tk().withdraw()
    file_path = askopenfilename(title="Chọn file", filetypes=[("Image/Video Files", "*.jpg;*.jpeg;*.png;*.mp4;*.avi;*.mov")])

    if not file_path:
        print("Không có file nào được chọn.")
    else:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            recognize_from_image(file_path)
        elif ext in [".mp4", ".avi", ".mov"]:
            recognize_from_video(file_path)
        else:
            print("File không được hỗ trợ.")