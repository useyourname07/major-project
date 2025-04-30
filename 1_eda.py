import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from collections import Counter

IMAGE_PATH = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/split_dataset/train/"

def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(os.path.join(IMAGE_PATH, file_path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def visualize_one_leaf(image):
    resized_image = cv2.resize(image, (205, 136))
    fig = px.imshow(resized_image)
    fig.update_layout(coloraxis_showscale=False, title="Leaf Image")
    fig.show()

def visualize_channel_distributions(image):
    channels = {
        "Red": image[:, :, 0].flatten(),
        "Green": image[:, :, 1].flatten(),
        "Blue": image[:, :, 2].flatten()
    }

    plt.figure(figsize=(12, 6))
    for channel, values in channels.items():
        sns.histplot(values, bins=50, kde=True, label=channel, color=channel.lower())
    plt.title("Channel Intensity Distributions")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def visualize_sample_leaves(image_path, categories, num_samples=5):
    plt.figure(figsize=(15, len(categories) * 3))
    for i, category in enumerate(categories):
        category_path = os.path.join(image_path, category)
        images = os.listdir(category_path)[:num_samples]

        for j, img_name in enumerate(images):
            img = cv2.imread(os.path.join(category_path, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(categories), num_samples, i * num_samples + j + 1)
            plt.imshow(img)
            plt.axis("off")
            if j == 0:
                plt.ylabel(category, fontsize=12)
    plt.suptitle("Sample Leaves from Each Category", fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_targets(image_path):
    categories = [
        category for category in os.listdir(image_path)
        if os.path.isdir(os.path.join(image_path, category))
    ]
    category_counts = {
        category: len(os.listdir(os.path.join(image_path, category)))
        for category in categories
    }

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
    plt.xticks(rotation=90)
    plt.title("Target Distribution")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    categories = [
        category for category in os.listdir(IMAGE_PATH)
        if os.path.isdir(os.path.join(IMAGE_PATH, category))
    ]
    sample_category = categories[0]
    sample_image_path = os.path.join(
        IMAGE_PATH, sample_category, os.listdir(os.path.join(IMAGE_PATH, sample_category))[0]
    )
    sample_image = cv2.imread(sample_image_path)
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

    visualize_one_leaf(sample_image)
    visualize_channel_distributions(sample_image)
    visualize_sample_leaves(IMAGE_PATH, categories, num_samples=5)
    visualize_targets(IMAGE_PATH)