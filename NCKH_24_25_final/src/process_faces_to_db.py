import os
import sys
import cv2
import numpy as np
from architecture import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from database import connect_db, create_table, insert_encoding
from custom_mtcnn_ThongTin import CustomMTCNN

# Tự định nghĩa hàm resource_path để lấy đúng đường dẫn khi chạy exe hoặc script
def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    # Khi chạy script, lấy từ thư mục cha của src
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_path)

# Cấu hình mô hình và thông số
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights(resource_path(os.path.join("src", "facenet_keras_weights.h5")))
detector = CustomMTCNN()
l2_normalizer = Normalizer('l2')

def normalize(img):
    """Chuẩn hóa ảnh dựa trên trung bình và độ lệch chuẩn."""
    mean, std = img.mean(), img.std()
    if std == 0:
        return img - mean
    return (img - mean) / std

def process_image(image_path):
    """Xử lý một ảnh để tạo embedding."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    if not results:
        print(f"Không phát hiện khuôn mặt trong ảnh: {image_path}")
        return None

    x, y, w, h = results[0]['box']
    x, y = abs(x), abs(y)
    if w <= 0 or h <= 0:
        print(f"Kích thước khuôn mặt không hợp lệ trong ảnh: {image_path}")
        return None

    face = img_rgb[y:y+h, x:x+w]
    face = normalize(face)
    face = cv2.resize(face, required_shape)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
    return encode

def process_folders_to_db(root_folder, db_path='database/face_recognition.db'):
    """Duyệt qua các folder con và lưu embedding vào cơ sở dữ liệu."""
    conn = None
    try:
        # Kết nối cơ sở dữ liệu
        conn = connect_db(resource_path(db_path))
        create_table(conn)

        # Duyệt qua các folder con
        for person_folder in os.listdir(root_folder):
            person_path = os.path.join(root_folder, person_folder)
            if not os.path.isdir(person_path):
                continue

            print(f"Đang xử lý folder: {person_folder}")
            # Giả sử tên folder là định dạng: "ma_sv_ho_ten"
            label = person_folder  # Có thể tùy chỉnh định dạng label

            # Duyệt qua các ảnh trong folder
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                if not os.path.isfile(image_path) or not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                # Xử lý ảnh và lấy embedding
                encode = process_image(image_path)
                if encode is not None:
                    # Lưu embedding vào cơ sở dữ liệu
                    insert_encoding(conn, label, encode.tobytes())
                    print(f"Đã lưu embedding cho ảnh: {image_path}")

        print("Hoàn tất xử lý và lưu vào cơ sở dữ liệu.")

    except Exception as e:
        print(f"Lỗi xảy ra: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    root_folder = "D:\\NCKH_24_25\\NCKH_24_25_final\\VN-celeb"
    process_folders_to_db(root_folder)