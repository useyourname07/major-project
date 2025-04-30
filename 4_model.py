import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    DenseNet121, MobileNetV2, EfficientNetB0, ResNet50, InceptionV3
)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
TRAIN_DIR = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/preprocessed_dataset/train"
VAL_DIR = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/preprocessed_dataset/val"
MODEL_SAVE_PATH = "C:/Users/User/Desktop/minor/Detection leaf disease/Datasets/Plant Village dataset/saved_model/leaf_classifier_v1.h5"

# Config
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
BACKBONE = "efficientnetb0"  # Options: 'densenet121', 'mobilenetv2', 'efficientnetb0', 'resnet50', 'inceptionv3'

def prepare_data(train_dir, val_dir, img_size, batch_size):
    train_gen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, rotation_range=20)
    val_gen = ImageDataGenerator(rescale=1. / 255)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    val_data = val_gen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    return train_data, val_data

def build_model(backbone, input_shape=(224, 224, 3), num_classes=38):
    if backbone == "densenet121":
        base = DenseNet121
    elif backbone == "mobilenetv2":
        base = MobileNetV2
    elif backbone == "efficientnetb0":
        base = EfficientNetB0
    elif backbone == "resnet50":
        base = ResNet50
    elif backbone == "inceptionv3":
        base = InceptionV3
    else:
        raise ValueError("Unsupported backbone")

    base_model = base(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=output)

def train_model(model, train_data, val_data, epochs, learning_rate, save_path):
    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )

    model.save(save_path)
    print(f"Model saved at: {save_path}")
    return history

if __name__ == "__main__":
    train_gen, val_gen = prepare_data(TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE)
    model = build_model(BACKBONE, input_shape=(224, 224, 3), num_classes=train_gen.num_classes)
    model.summary()
    history = train_model(model, train_gen, val_gen, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH)
