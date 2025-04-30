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

# Paths and parameters
data_dir = "0 Plant Village Dataset/data with aug"
img_height, img_width = 224, 224
batch_size=32
model_path = "0 list of models/model_resnet50.h5"

"""Performs EDA and visualizations on the dataset."""
# Data Generator for EDA
data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Load training data for EDA
train_gen = data_gen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Analyze class distribution
class_counts = pd.DataFrame.from_dict(train_gen.class_indices, orient='index', columns=['Class Index'])
class_counts['Frequency'] = [
    len(os.listdir(os.path.join(data_dir, cls))) for cls in train_gen.class_indices
]
class_counts = class_counts.sort_values(by='Frequency', ascending=False)

# Plot class distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=class_counts.index, y=class_counts['Frequency'])
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Display sample augmented images
sample_images, _ = next(train_gen)  # Fetch one batch of images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for img, ax in zip(sample_images[:5], axes):  # Show only 5 images
    ax.imshow(img)
    ax.axis('off')
plt.suptitle("Sample Augmented Images")
plt.show()

# Parameters
img_height, img_width = 224, 224 # Same size used for training
class_labels = ['Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy', 
                'Background_without_leaves', 'Blueberry__healthy', 'Cherry__healthy', 'Cherry__Powdery_mildew',
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

# Load the saved model
model = load_model('0 list of models/model_mobilenetv2.h5')

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
test_dir = '0 model_resnet50/test'  # Replace with the path to your labeled test dataset

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