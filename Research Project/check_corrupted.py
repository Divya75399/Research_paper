import cv2
import os

folders = [
    r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data\fold_0\all",
    r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data\fold_0\hem"
]

corrupted = 0
total = 0

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".bmp"):
            total += 1
            path = os.path.join(folder, file)
            img = cv2.imread(path)

            if img is None:
                print("Corrupted:", file)
                corrupted += 1

print("Total Images Checked:", total)
print("Corrupted Images:", corrupted)

if corrupted == 0:
    print("✅ No corrupted images found.")