# Hệ Thống Nhận Diện Khuôn Mặt (Face Recognition System)
file đầy đủ: https://drive.google.com/file/d/10u9n1V1thW8N2tzHrQJSNRkCeaqC3MwG/view?usp=sharing
## Mô tả
Đây là hệ thống nhận diện khuôn mặt sử dụng MTCNN để phát hiện khuôn mặt và FaceNet (InceptionResNetV2) để sinh vector đặc trưng (embedding). Hệ thống hỗ trợ:
- Thu thập khuôn mặt từ webcam hoặc ảnh.
- Huấn luyện và lưu trữ embedding vào database.
- Nhận diện khuôn mặt từ webcam, ảnh hoặc video.
- Giao diện đồ họa bằng PyQt5.

## Cấu trúc thư mục
```
NCKH_24_25_final/
│
├── gui/
│   ├── main_app.py         # File chạy giao diện chính
│   ├── QtGui.ui            # File giao diện PyQt5
│   └── main_app.spec       # File cấu hình build PyInstaller
│
├── src/
│   ├── architecture.py
│   ├── capture_faces.py
│   ├── collect_faces_from_image.py
│   ├── custom_mtcnn.py
│   ├── custom_mtcnn_ThongTin.py
│   ├── database.py
│   ├── process_faces_to_db.py
│   ├── recognize_faces.py
│   ├── recognize_from_image_or_video.py
│   ├── train_faces.py
│   └── encodings/
│       └── encodings.pkl
│   └── facenet_keras_weights.h5
│
├── database/
│   └── face_recognition.db
│
└── dist/
    └── main_app/
        └── main_app.exe    # File chạy đã build
```

## Yêu cầu môi trường
- Python >= 3.8
- Các thư viện: `tensorflow`, `keras`, `opencv-python`, `numpy`, `scikit-learn`, `mtcnn`, `PyQt5`, `matplotlib`, `pandas`, `h5py`, `sqlite3`, `tkinter`
- File trọng số: `facenet_keras_weights.h5`
- File dữ liệu: `encodings.pkl`, `face_recognition.db`
- File MTCNN: `mtcnn_weights.npy` (được đóng gói khi build)

## Hướng dẫn sử dụng

### 1. Chạy trực tiếp bằng Python (dành cho dev)
```bash
cd gui
python main_app.py
```

### 2. Build file .exe (dành cho người dùng cuối)
- Đảm bảo đã cài PyInstaller:  
  `pip install pyinstaller`
- Build:
  ```bash
  pyinstaller main_app.spec
  ```
- File chạy sẽ nằm ở `dist/main_app/main_app.exe`

### 3. Gửi cho người khác
- Gửi toàn bộ thư mục `dist/main_app/` (bao gồm `.exe` và các thư mục dữ liệu đi kèm như `src`, `database`, `QtGui.ui`...)
- Người nhận chỉ cần chạy `main_app.exe`

### 4. Các chức năng chính
- **Thu thập khuôn mặt:**  
  Chụp từ webcam hoặc chọn ảnh để lưu embedding vào database.
- **Huấn luyện:**  
  Sinh lại file `encodings.pkl` từ database.
- **Nhận diện:**  
  Nhận diện từ webcam, ảnh hoặc video, xuất file điểm danh `.xlsx`.
- **Xóa sinh viên:**  
  Xóa dữ liệu khuôn mặt theo mã sinh viên.

## Lưu ý khi đóng gói (.exe)
- Đảm bảo file `mtcnn_weights.npy` được đóng gói vào thư mục `mtcnn/data` (đã cấu hình trong file `.spec`).
- Luôn sử dụng hàm `resource_path` để truy cập file dữ liệu khi chạy `.exe`.

## Liên hệ
- Tác giả: Lê Đăng Vũ
- Email: ledangvu1209@gmail.com

---

**Nếu gặp lỗi về thiếu file dữ liệu, hãy kiểm tra lại cấu trúc thư mục và các đường dẫn trong file `.spec` cũng như hàm `resource_path` trong code.**
