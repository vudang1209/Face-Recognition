import cv2
from custom_mtcnn import CustomMTCNN

img_bgr = cv2.imread('D:/NCKH_24_25/z6595386735132_1b3f5a64d4ca916993bced4ee8455e0b.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

mtcnn = CustomMTCNN()
mtcnn.show_pipeline(img_rgb)
