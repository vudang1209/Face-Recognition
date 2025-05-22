import os
import cv2
import numpy as np
from architecture import InceptionResNetV2
from sklearn.preprocessing import Normalizer
from database import connect_db, create_table, insert_encoding
from custom_mtcnn import CustomMTCNN
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import matplotlib.pyplot as plt

required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights("./src/facenet_keras_weights.h5")
detector = CustomMTCNN()
l2_normalizer = Normalizer('l2')

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def display_intermediate_results(img_rgb, stages_data, image_path):
    plt.figure(figsize=(15, 5))

    # Gốc
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Ảnh gốc")
    plt.axis('off')

    stages = ['P-Net', 'R-Net', 'O-Net']
    for i, stage_name in enumerate(stages):
        stage_data = stages_data.get(stage_name)
        if stage_data is not None:
            plt.subplot(1, 4, i + 2)
            plt.imshow(cv2.cvtColor(stage_data, cv2.COLOR_BGR2RGB))
            plt.title(stage_name)
            plt.axis('off')

    plt.suptitle(f"Kết quả MTCNN: {os.path.basename(image_path)}")
    plt.tight_layout()
    plt.show()

def collect_faces_from_images(image_paths, ma_sv, ho_ten):
    label = f"{ma_sv}_{ho_ten.replace(' ', '_')}"
    conn = connect_db()
    create_table(conn)

    for image_path in image_paths:
        print(f"🔍 Đang xử lý ảnh: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không thể đọc ảnh: {image_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            # Lấy kết quả và ảnh trung gian từ các giai đoạn của MTCNN
            results, pnet_img, rnet_img, onet_img = detector.detect_and_visualize(img_rgb)
        except Exception as e:
            print(f"Lỗi khi xử lý MTCNN: {e}")
            continue

        if results is None or len(results) == 0:
            print(f"Không phát hiện khuôn mặt trong ảnh: {image_path}")
            continue

        # Hiển thị pipeline trung gian
        stages_data = {
            "P-Net": pnet_img,
            "R-Net": rnet_img,
            "O-Net": onet_img
        }
        display_intermediate_results(img_rgb, stages_data, image_path)

        for i, res in enumerate(results):
            if isinstance(res, dict) and 'box' in res:
                x, y, w, h = res['box']
            elif isinstance(res, (list, tuple, np.ndarray)) and len(res) == 4:
                x, y, w, h = res
            else:
                print(f"Không thể phân tích res: {res}")
                continue

            x, y = abs(int(x)), abs(int(y))
            face = img_rgb[y:y + int(h), x:x + int(w)]

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]

            insert_encoding(conn, label, encode.tobytes())
            print(f"Đã lưu khuôn mặt {i+1} từ ảnh: {image_path}")

    conn.close()
    print("Hoàn tất thu thập khuôn mặt từ các ảnh.")

if __name__ == "__main__":
    Tk().withdraw()
    image_paths = askopenfilenames(title="Chọn ảnh", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

    if not image_paths:
        print("Không có ảnh nào được chọn.")
    else:
        ma_sv = input("Nhập mã sinh viên: ")
        ho_ten = input("Nhập họ tên: ")
        collect_faces_from_images(image_paths, ma_sv, ho_ten)
