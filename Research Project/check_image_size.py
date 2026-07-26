import cv2
import os

folder = r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data\fold_0\all"

for file in os.listdir(folder):
    if file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".png"):
        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is not None:
            print("File Name :", file)
            print("Image Shape :", image.shape)
            break