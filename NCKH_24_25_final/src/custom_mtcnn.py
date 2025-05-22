import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

class CustomMTCNN:
    def __init__(self):
        """
        Khởi tạo MTCNN và các thuộc tính để lưu bounding box từ từng giai đoạn.
        """
        self.detector = MTCNN()

    def detect_and_visualize(self, img_rgb):
        """
        Phát hiện khuôn mặt và hiển thị bounding box từ từng giai đoạn của MTCNN.
        """
        img = img_rgb.copy()
        h, w = img.shape[:2]
        scale = 1.0
        max_dim = max(h, w)

        # Thay đổi kích thước ảnh nếu quá lớn
        if max_dim > 1000:
            print(f"Ảnh lớn ({h}x{w}), thay đổi kích thước xuống tối đa 1000 pixel.")
            scale = 1000 / max_dim
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Phát hiện khuôn mặt
        results = self.detector.detect_faces(img)

        # Tạo bản sao ảnh cho từng giai đoạn
        pnet_img = img.copy()
        rnet_img = img.copy()
        onet_img = img.copy()

        # Nếu không phát hiện khuôn mặt, trả về ảnh gốc
        if not results:
            print("Không phát hiện khuôn mặt.")
            return img, pnet_img, rnet_img, onet_img

        # --- P-Net ---
        if hasattr(self.detector, 'pnet_proposals'):
            print(f"[+] Số bounding box từ P-Net: {len(self.detector.pnet_proposals)}")
            for proposal in self.detector.pnet_proposals:
                px, py, pw, ph = proposal[:4]
                cv2.rectangle(pnet_img, (int(px), int(py)), (int(px + pw), int(py + ph)), (255, 0, 0), 1)

        # --- R-Net ---
        if hasattr(self.detector, 'rnet_proposals'):
            print(f"[+] Số bounding box từ R-Net: {len(self.detector.rnet_proposals)}")
            for refined in self.detector.rnet_proposals:
                rx, ry, rw, rh = refined[:4]
                cv2.rectangle(rnet_img, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)), (0, 255, 0), 1)

        # --- O-Net ---
        print(f"[+] Số khuôn mặt được phát hiện sau O-Net: {len(results)}")
        for i, face in enumerate(results):
            x, y, w, h = face['box']
            x, y = abs(int(x)), abs(int(y))
            w, h = int(w), int(h)
            confidence = face['confidence']
            cv2.rectangle(onet_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(onet_img, f"{confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Vẽ các điểm mốc (facial landmarks)
            for key, point in face['keypoints'].items():
                cv2.circle(onet_img, point, 3, (0, 255, 255), -1)

        return img, pnet_img, rnet_img, onet_img

    def show_pipeline(self, img_rgb):
        """
        Hiển thị pipeline xử lý của MTCNN với bounding box từ từng giai đoạn.
        """
        goc, pnet_img, rnet_img, onet_img = self.detect_and_visualize(img_rgb)

        # Hiển thị ảnh từ từng giai đoạn
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        axs[0, 0].imshow(cv2.cvtColor(goc, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Ảnh gốc")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(cv2.cvtColor(pnet_img, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("P-Net: Đề xuất vùng")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(cv2.cvtColor(rnet_img, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("R-Net: Tinh chỉnh")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(cv2.cvtColor(onet_img, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("O-Net: Kết quả cuối + điểm mốc")
        axs[1, 1].axis("off")

        plt.tight_layout()
        plt.show()