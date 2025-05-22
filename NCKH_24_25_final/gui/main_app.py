
import sys
import os
import subprocess
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMessageBox

# Xác định đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
ui_file = os.path.join(current_dir, "QtGui.ui")

class FaceRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceRecognitionApp, self).__init__()
        uic.loadUi(ui_file, self)

        # Tìm các widget
        self.text_inputs = self.findChildren(QtWidgets.QLineEdit)
        self.buttons = self.findChildren(QtWidgets.QPushButton)

        try:
            # Kết nối nút Lấy Ảnh
            lay_anh_buttons = [btn for btn in self.buttons if "lay" in btn.objectName().lower() or "anh" in btn.objectName().lower()]
            if lay_anh_buttons:
                lay_anh_buttons[0].clicked.connect(self.capture_images)

            # Kết nối nút Cập Nhật
            cap_nhat_buttons = [btn for btn in self.buttons if "cap" in btn.objectName().lower() or "nhat" in btn.objectName().lower()]
            if cap_nhat_buttons:
                cap_nhat_buttons[0].clicked.connect(self.train_model)

            # Kết nối nút Nhận Diện
            nhan_dang_buttons = [btn for btn in self.buttons if btn.objectName().lower() == "btnnhandang"]
            if nhan_dang_buttons:
                nhan_dang_buttons[0].clicked.connect(self.recognize_faces)

            # Kết nối nút Thu Khuôn Mặt Bằng Ảnh
            thu_anh_buttons = [btn for btn in self.buttons if "thu" in btn.objectName().lower()]
            if thu_anh_buttons:
                thu_anh_buttons[0].clicked.connect(self.collect_faces_from_image)

            # Kết nối nút Nhận Diện Khuôn Mặt Bằng Ảnh/Video
            nhan_dien_anh_video_buttons = [btn for btn in self.buttons if "video" in btn.objectName().lower()]
            if nhan_dien_anh_video_buttons:
                nhan_dien_anh_video_buttons[0].clicked.connect(self.recognize_faces_from_image_or_video)

        except Exception as e:
            print(f"Lỗi khi kết nối nút: {str(e)}")

    def capture_images(self):
        try:
            ma_sv_inputs = [i for i in self.text_inputs if "ma" in i.objectName().lower()]
            ho_ten_inputs = [i for i in self.text_inputs if "ho" in i.objectName().lower() or "ten" in i.objectName().lower()]

            if ma_sv_inputs and ho_ten_inputs:
                student_id = ma_sv_inputs[0].text().strip()
                name = ho_ten_inputs[0].text().strip()
            else:
                student_id, ok1 = QtWidgets.QInputDialog.getText(self, "Nhập thông tin", "Mã sinh viên:")
                if not ok1 or not student_id.strip():
                    return
                name, ok2 = QtWidgets.QInputDialog.getText(self, "Nhập thông tin", "Họ và tên:")
                if not ok2 or not name.strip():
                    return

            if not student_id or not name:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập đầy đủ mã sinh viên và họ tên!")
                return

            script_path = os.path.join(parent_dir, "src", "capture_faces.py")
            process = subprocess.Popen(["python", script_path], stdin=subprocess.PIPE, text=True)
            process.communicate(input=f"{student_id}\n{name}\n50\n")

            QMessageBox.information(self, "Thành công", "Đã lưu trữ dữ liệu khuôn mặt!")

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi chụp khuôn mặt: {str(e)}")

    def train_model(self):
        try:
            script_path = os.path.join(parent_dir, "src", "train_faces.py")
            encodings_dir = os.path.join(parent_dir, "src", "encodings")
            os.makedirs(encodings_dir, exist_ok=True)
            subprocess.Popen(["python", script_path]).wait()
            QMessageBox.information(self, "Thành công", "Đã cập nhật dữ liệu khuôn mặt vào hệ thống!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi cập nhật dữ liệu: {str(e)}")

    def recognize_faces(self):
        try:
            encodings_path = os.path.join(parent_dir, "src", "encodings", "encodings.pkl")
            if not os.path.exists(encodings_path):
                QMessageBox.warning(self, "Cảnh báo", "Chưa có dữ liệu khuôn mặt! Vui lòng nhấn nút Cập Nhật trước.")
                return
            script_path = os.path.join(parent_dir, "src", "recognize_faces.py")
            subprocess.Popen(["python", script_path])
            QMessageBox.information(self, "Thông báo", "Đã mở chương trình nhận diện khuôn mặt.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi chạy chức năng nhận diện: {str(e)}")

    def collect_faces_from_image(self):
        try:
            script_path = os.path.join(parent_dir, "src", "collect_faces_from_image.py")
            subprocess.Popen(["python", script_path])
            # QMessageBox.information(self, "Thành công", "Đã thu khuôn mặt từ ảnh thành công!")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi thu khuôn mặt từ ảnh: {str(e)}")

    def recognize_faces_from_image_or_video(self):
        try:
            script_path = os.path.join(parent_dir, "src", "recognize_from_image_or_video.py")
            subprocess.Popen(["python", script_path])
            # QMessageBox.information(self, "Thông báo", "Đã mở chương trình nhận diện từ ảnh/video.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi nhận diện ảnh/video: {str(e)}")

# Chạy ứng dụng
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
