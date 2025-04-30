import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from prettytable import PrettyTable

# Parameters
img_height, img_width = 128, 128 # Same size used for training
class_labels  = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Load the saved model
model = load_model('models/model_tanbir1_.h5')

# Function to preprocess a single image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))  # Load and resize the image
    img_array = img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Create GUI window
window = tk.Tk()
window.title("Plant Disease Classifier")
window.geometry("600x600")

# Display image function
def show_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250,250))  # Resize image for display
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img

# Function to extract plant species and disease from class label
def get_plant_and_disease(class_label):
    # Special case for 'Pepper,_bell'
    if 'Pepper,_bell' in class_label:
        plant_species, disease = class_label.split('__', 1)
    else:
        plant_species, disease = class_label.split('__', 1)
    
    return plant_species, disease

# Predict function
def predict_image():
    # Open file dialog to select an image
    image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    
    if image_path:
        # Preprocess and predict the image
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        confidence_scores = predictions[0]
        
        # Extract plant species and disease from the label
        plant_species, disease = get_plant_and_disease(predicted_class_label)
        
        # Update GUI with the result
        show_image(image_path)
        species_label.config(text=f"Plant Species: {plant_species}")
        result_label.config(text=f"Leaf Disease: {disease}")
        confidence_label.config(text=f"Confidence: {confidence_scores[predicted_class_index]:.4f}")
        
# Add components to the window
img_label = tk.Label(window)
img_label.pack(pady=10)

species_label = tk.Label(window, text="Plant Species: ", font=('Helvetica', 14, 'bold'), fg="green")  # Updated font and color for species
species_label.pack(pady=10)

result_label = tk.Label(window, text="Leaf Disease: ", font=('Helvetica', 14, 'bold'), fg="red")  # Updated font and color for disease
result_label.pack(pady=10)

confidence_label = tk.Label(window, text="Confidence: ", font=('Helvetica', 12, 'italic'), fg="blue")  # Different font style and color for confidence
confidence_label.pack(pady=10)

predict_button = tk.Button(window, text="Select Image to Predict", command=predict_image, font=('Helvetica', 14))
predict_button.pack(pady=20)

# Run the GUI event loop
window.mainloop()

# Test the model on a batch of labeled test samples (optional)
# Assuming test_dir contains directories for each class
test_dir = 'datasets/archive2/test'  # Replace with the path to your labeled test dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Print metrics using PrettyTable
print("Classification Report:")
report = classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys()), output_dict=True)

# Create PrettyTable for the classification report
table = PrettyTable()
table.field_names = ["Class", "Precision", "Recall", "F1-Score", "Support"]

for class_name, metrics in report.items():
    if class_name not in ["accuracy", "macro avg", "weighted avg"]:  # Skip aggregate metrics
        table.add_row([
            class_name,
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            int(metrics['support'])
        ])

# Add overall metrics
table.add_row(["Overall (Accuracy)", "-", "-", f"{report['accuracy']:.2f}", sum(y_true)])
table.add_row(["Macro Avg", f"{report['macro avg']['precision']:.2f}", f"{report['macro avg']['recall']:.2f}", 
               f"{report['macro avg']['f1-score']:.2f}", "-"])
table.add_row(["Weighted Avg", f"{report['weighted avg']['precision']:.2f}", f"{report['weighted avg']['recall']:.2f}", 
               f"{report['weighted avg']['f1-score']:.2f}", "-"])

print(table)

# Assuming `y_true` and `y_pred` are already defined
conf_matrix = confusion_matrix(y_true, y_pred)

# Shorten class names for display
short_class_labels = [
    label.split("__")[0][:3] + "__" + label.split("__")[1][:6] if "__" in label else label[:10]
    for label in test_generator.class_indices.keys()
]

# Plot the confusion matrix as a heatmap

plt.figure(figsize=(12, 10))  # Adjust figure size
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=short_class_labels, 
            yticklabels=short_class_labels)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()  # Automatically adjust layout to avoid cutoff
plt.show()