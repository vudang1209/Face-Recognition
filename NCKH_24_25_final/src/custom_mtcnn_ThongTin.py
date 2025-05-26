import mtcnn
import numpy as np
import cv2

class CustomMTCNN(mtcnn.MTCNN):
    def __init__(self, *args, **kwargs):
        # Khởi tạo lớp MTCNN từ thư viện mtcnn
        super().__init__(*args, **kwargs)

    def detect_faces(self, image):
        """
        Ghi đè phương thức detect_faces để in ra thông tin của các lớp mạng P-Net, R-Net và O-Net.
        """
        # Phát hiện khuôn mặt sử dụng MTCNN (đã tự động khởi tạo các mạng)
        faces = super().detect_faces(image)
        
        # In ra thông tin về các khuôn mặt được phát hiện
        print("Faces Detected:")
        for i, face in enumerate(faces):
            print(f"Face {i+1}:")
            print("Bounding Box: ", face['box'])
            print("Keypoints: ", face['keypoints'])
            print("Confidence: ", face['confidence'])
        
        return faces
