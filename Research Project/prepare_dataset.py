import os
import shutil

# Source dataset
source = r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data"

# Destination dataset
destination = "dataset_classification"

# Create folders
for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(destination, folder, "Leukemia"), exist_ok=True)
    os.makedirs(os.path.join(destination, folder, "Normal"), exist_ok=True)

# Fold mapping
folds = {
    "train": ["fold_0", "fold_1"],
    "val": ["fold_2"]
}

for split in folds:
    for fold in folds[split]:

        leukemia = os.path.join(source, fold, "all")
        normal = os.path.join(source, fold, "hem")

        for file in os.listdir(leukemia):
            shutil.copy(
                os.path.join(leukemia, file),
                os.path.join(destination, split, "Leukemia", file)
            )

        for file in os.listdir(normal):
            shutil.copy(
                os.path.join(normal, file),
                os.path.join(destination, split, "Normal", file)
            )

print("Dataset Prepared Successfully!")