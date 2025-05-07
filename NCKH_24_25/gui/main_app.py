import sys
import os
import subprocess
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

# Xác định đường dẫn
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
ui_file = os.path.join(current_dir, "QtGui.ui")

class FaceRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceRecognitionApp, self).__init__()
        
        # Load trực tiếp từ file .ui
        uic.loadUi(ui_file, self)
        
        # In ra tên của tất cả các widget để xác định tên chính xác
        print("\nDanh sách các widget trong UI:")
        for child in self.findChildren(QtWidgets.QWidget):
            print(f"- {child.objectName()}")
        
        # Tìm các trường nhập liệu bằng kiểu widget
        self.text_inputs = self.findChildren(QtWidgets.QLineEdit)
        print("\nCác trường nhập liệu:")
        for text_input in self.text_inputs:
            print(f"- {text_input.objectName()}")
        
        # Tìm các nút bằng kiểu widget
        self.buttons = self.findChildren(QtWidgets.QPushButton)
        print("\nCác nút:")
        for button in self.buttons:
            print(f"- {button.objectName()}")
        
        # Kết nối các nút bằng objectName
        try:
            # Tìm và kết nối các nút
            lay_anh_buttons = [btn for btn in self.buttons if "lay" in btn.objectName().lower() or "anh" in btn.objectName().lower()]
            cap_nhat_buttons = [btn for btn in self.buttons if "cap" in btn.objectName().lower() or "nhat" in btn.objectName().lower()]
            nhan_dang_buttons = [btn for btn in self.buttons if "nhan" in btn.objectName().lower() or "dang" in btn.objectName().lower()]
            
            if lay_anh_buttons:
                lay_anh_buttons[0].clicked.connect(self.capture_images)
                print(f"Đã kết nối nút Lấy Ảnh: {lay_anh_buttons[0].objectName()}")
            
            if cap_nhat_buttons:
                cap_nhat_buttons[0].clicked.connect(self.train_model)
                print(f"Đã kết nối nút Cập Nhật: {cap_nhat_buttons[0].objectName()}")
            
            if nhan_dang_buttons:
                nhan_dang_buttons[0].clicked.connect(self.recognize_faces)
                print(f"Đã kết nối nút Nhận Dạng: {nhan_dang_buttons[0].objectName()}")
                
            print("Kết nối các nút thành công")
        except Exception as e:
            print(f"Lỗi khi kết nối nút: {str(e)}")
    
    def capture_images(self):
        """Gọi script capture_faces.py để chụp khuôn mặt"""
        try:
            # Tìm các trường nhập liệu cho mã sinh viên và họ tên
            ma_sv_inputs = [input for input in self.text_inputs if "ma" in input.objectName().lower() or "msv" in input.objectName().lower()]
            ho_ten_inputs = [input for input in self.text_inputs if "ho" in input.objectName().lower() or "ten" in input.objectName().lower() or "ht" in input.objectName().lower()]
            
            # Sử dụng các trường tìm được
            if ma_sv_inputs and ho_ten_inputs:
                student_id = ma_sv_inputs[0].text().strip()
                name = ho_ten_inputs[0].text().strip()
                
                print(f"Tìm thấy mã SV: {student_id} từ {ma_sv_inputs[0].objectName()}")
                print(f"Tìm thấy họ tên: {name} từ {ho_ten_inputs[0].objectName()}")
            else:
                # Trường hợp không tìm thấy, hiển thị dialog yêu cầu nhập
                student_id, ok1 = QtWidgets.QInputDialog.getText(self, "Nhập thông tin", "Mã sinh viên:")
                if not ok1 or not student_id.strip():
                    return
                    
                name, ok2 = QtWidgets.QInputDialog.getText(self, "Nhập thông tin", "Họ và tên:")
                if not ok2 or not name.strip():
                    return
            
            # Kiểm tra dữ liệu nhập vào
            if not student_id or not name:
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập đầy đủ mã sinh viên và họ tên!")
                return
            
            print(f"Bắt đầu chụp khuôn mặt cho: {student_id} - {name}")
            
            # Sử dụng subprocess với đối số cho script con
            script_path = os.path.join(parent_dir, "src", "capture_faces.py")
            print(f"Đường dẫn script: {script_path}")
            
            # Tạo quá trình con và cung cấp đầu vào qua stdin
            process = subprocess.Popen(
                ["python", script_path],
                stdin=subprocess.PIPE,
                text=True
            )
            
            # Gửi các đầu vào mà script có thể yêu cầu
            process.communicate(input=f"{student_id}\n{name}\n50\n")
            
            QMessageBox.information(self, "Thành công", "Đã lưu trữ dữ liệu khuôn mặt!")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi chụp khuôn mặt: {str(e)}")
            print(f"Lỗi khi chụp khuôn mặt: {str(e)}")
    
    def train_model(self):
        """Gọi script train_faces.py để cập nhật model"""
        try:
            print("Bắt đầu cập nhật dữ liệu khuôn mặt")
            
            # Gọi script train_faces.py
            script_path = os.path.join(parent_dir, "src", "train_faces.py")
            
            # Đảm bảo thư mục encodings tồn tại
            encodings_dir = os.path.join(parent_dir, "src", "encodings")
            os.makedirs(encodings_dir, exist_ok=True)
            
            print(f"Chạy script: {script_path}")
            process = subprocess.Popen(["python", script_path])
            process.wait()
            
            QMessageBox.information(self, "Thành công", "Đã cập nhật dữ liệu khuôn mặt vào hệ thống!")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi cập nhật dữ liệu: {str(e)}")
            print(f"Lỗi khi cập nhật dữ liệu: {str(e)}")
    
    def recognize_faces(self):
        """Gọi script recognize_faces.py để nhận diện khuôn mặt"""
        try:
            print("Bắt đầu nhận diện khuôn mặt")
            
            # Kiểm tra file encodings đã tồn tại chưa
            encodings_path = os.path.join(parent_dir, "src", "encodings", "encodings.pkl")
            if not os.path.exists(encodings_path):
                QMessageBox.warning(self, "Cảnh báo", 
                                  "Chưa có dữ liệu khuôn mặt! Vui lòng nhấn nút Cập Nhật trước.")
                return
            
            # Gọi script recognize_faces.py
            script_path = os.path.join(parent_dir, "src", "recognize_faces.py")
            print(f"Chạy script: {script_path}")
            
            subprocess.Popen(["python", script_path])
            
            QMessageBox.information(self, "Thông báo", 
                              "Đã mở chương trình nhận diện khuôn mặt. Nhấn phím 'q' trên cửa sổ camera để thoát.")
            
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Lỗi khi chạy chức năng nhận diện: {str(e)}")
            print(f"Lỗi khi chạy chức năng nhận diện: {str(e)}")

# Chạy ứng dụng
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())