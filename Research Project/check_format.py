import os

folder = r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data\fold_0\all"

formats = set()

for file in os.listdir(folder):
    extension = os.path.splitext(file)[1]
    formats.add(extension)

print("Image Formats:", formats)