import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
from tqdm import tqdm

MODEL_SAVE_PATH = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/saved_model/leaf_classifier_v1.h5"
IMG_SIZE = (224, 224)
CLASS_LABELS = ['Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy', 
    'Blueberry__healthy', 'Cherry__healthy', 'Cherry__Powdery_mildew',
    'Corn__Cercospora_leaf_spot Grat_leaf_spot', 'Corn__Common_rust', 'Corn__healthy', 
    'Corn__Northern_Leaf_Blight', 'Grape__Black_rot', 'Grape__Esca_(Black_Measles)', 
    'Grape__healthy', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)', 
    'Orange__Haunglongbing_(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach__healthy',
    'Pepper,_bell__Bacterial_spot', 'Pepper,_bell__healthy', 'Potato__Early_blight', 
    'Potato__healthy', 'Potato__Late_blight', 'Raspberry__healthy', 'Soybean__healthy', 
    'Squash__Powdery_mildew', 'Strawberry__healthy', 'Strawberry__Leaf_scorch', 
    'Tomato__Bacterial_spot', 'Tomato__Early_blight', 'Tomato__healthy', 'Tomato__Late_blight',
    'Tomato__Leaf_Mold', 'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = cv2.convertScaleAbs((img * 255).astype(np.uint8), alpha=1.2, beta=20)
    img_array = np.expand_dims(img, axis=0)
    return img_array

def load_trained_model(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
    return model

def launch_gui(model):
    def show_image(image_path):
        img = Image.open(image_path)
        img = img.resize((250, 250))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

    def predict_image():
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if image_path:
            try:
                img_array = preprocess_image(image_path)
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions)
                confidence_score = predictions[0][predicted_class_index]
                predicted_class_label = CLASS_LABELS[predicted_class_index]
                plant_species, disease = predicted_class_label.split("__", 1)
                show_image(image_path)
                species_label.config(text=f"Plant Species: {plant_species}")
                result_label.config(text=f"Leaf Disease: {disease}")
                confidence_label.config(text=f"Confidence: {confidence_score:.4f}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    window = tk.Tk()
    window.title("Plant Disease Detection")
    window.geometry("600x600")

    img_label = tk.Label(window)
    img_label.pack(pady=10)

    species_label = tk.Label(window, text="Plant Species: ", font=('Helvetica', 14, 'bold'), fg="green")
    species_label.pack(pady=10)

    result_label = tk.Label(window, text="Leaf Disease: ", font=('Helvetica', 14, 'bold'), fg="red")
    result_label.pack(pady=10)

    confidence_label = tk.Label(window, text="Confidence: ", font=('Helvetica', 12, 'italic'), fg="blue")
    confidence_label.pack(pady=10)

    predict_button = tk.Button(window, text="Select Image to Predict", command=predict_image, font=('Helvetica', 14))
    predict_button.pack(pady=20)

    window.mainloop()

if __name__ == "__main__":
    model = load_trained_model(MODEL_SAVE_PATH)
    launch_gui(model)