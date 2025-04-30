import os
import shutil
import random
from tqdm import tqdm

random.seed(42)

def split_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    assert train_ratio + val_ratio < 1.0

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    for class_name in tqdm(os.listdir(input_dir), desc="Splitting dataset"):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        all_files = os.listdir(class_dir)
        random.shuffle(all_files)

        train_split = int(len(all_files) * train_ratio)
        val_split = int(len(all_files) * (train_ratio + val_ratio))

        train_files = all_files[:train_split]
        val_files = all_files[train_split:val_split]
        test_files = all_files[val_split:]

        for split, split_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            split_class_dir = os.path.join(output_dir, split, class_name)
            if not os.path.exists(split_class_dir):
                os.makedirs(split_class_dir)

            for file_name in split_files:
                src_path = os.path.join(class_dir, file_name)
                dst_path = os.path.join(split_class_dir, file_name)
                shutil.copy(src_path, dst_path)

input_directory = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/data/data with aug"
output_directory = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/split_dataset"

split_dataset(input_dir=input_directory, output_dir=output_directory, train_ratio=0.7, val_ratio=0.15)

print("Dataset splitting completed! Split data saved in:", output_directory)