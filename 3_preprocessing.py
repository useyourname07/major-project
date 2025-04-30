import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import plotly.express as px

IMAGE_PATH = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/split_dataset/train/"
OUTPUT_PATH = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/preprocessed_dataset/"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def normalize_image(image):
    return image / 255.0

def adjust_brightness_contrast(image, alpha=1.2, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def preprocess_and_save_images(input_dir, output_dir):
    for category in tqdm(os.listdir(input_dir), desc="Preprocessing images"):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for file_name in os.listdir(category_path):
            input_path = os.path.join(category_path, file_name)
            output_path = os.path.join(output_category_path, file_name)

            image = load_image(input_path)
            image = normalize_image(image)
            image = adjust_brightness_contrast((image * 255).astype(np.uint8))

            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def channel_analysis(image):
    mean_red = np.mean(image[:, :, 0])
    mean_green = np.mean(image[:, :, 1])
    mean_blue = np.mean(image[:, :, 2])
    print(f"Mean Red: {mean_red:.2f}, Mean Green: {mean_green:.2f}, Mean Blue: {mean_blue:.2f}")

def visualize_augmentations(image_path):
    image = load_image(image_path)
    normalized_image = normalize_image(image)
    adjusted_image = adjust_brightness_contrast((normalized_image * 255).astype(np.uint8))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(normalized_image)
    plt.title("Normalized Image")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(adjusted_image)
    plt.title("Brightness/Contrast Adjusted")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preprocess_and_save_images(IMAGE_PATH, OUTPUT_PATH)
    print("Preprocessing complete! Saved to:", OUTPUT_PATH)

    sample_category = os.listdir(IMAGE_PATH)[0]
    sample_image_name = os.listdir(os.path.join(IMAGE_PATH, sample_category))[0]
    sample_image_path = os.path.join(IMAGE_PATH, sample_category, sample_image_name)
    sample_image = load_image(sample_image_path)

    channel_analysis(sample_image)
    visualize_augmentations(sample_image_path)