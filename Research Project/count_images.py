import os

all_folder = r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data\fold_0\all"
hem_folder = r"dataset\C-NMC 2019 (PKG)\C-NMC_training_data\fold_0\hem"

all_count = len([f for f in os.listdir(all_folder) if f.endswith(".bmp")])
hem_count = len([f for f in os.listdir(hem_folder) if f.endswith(".bmp")])

print("Leukemia Images (all):", all_count)
print("Normal Images (hem):", hem_count)
print("Total Images:", all_count + hem_count)