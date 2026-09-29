import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json

# ========================
# PATHS
# ========================
data_dir = r"D:\celebA\celebA-dataset"
img_dir = os.path.join(data_dir, "img_align_celeba", "img_align_celeba")
attr_path = os.path.join(data_dir, "list_attr_celeba.csv")

# ========================
# LOAD ATTRIBUTES
# ========================
attrs = pd.read_csv(attr_path)
print("✅ Attributes loaded:", attrs.shape)

# Convert -1 to 0 (CelebA uses -1/1 format)
attrs = attrs.replace(-1, 0)

# Use a smaller subset (first 5000 images for faster training)
attrs = attrs.iloc[:5000]

# We'll train on the 'Smiling' attribute
target_attr = "Smiling"
print(f"🎯 Training target attribute: {target_attr}")

# Build dataframe for generator
data = attrs[[target_attr]].copy()
data["image_id"] = attrs.index.map(lambda x: f"{x+1:06d}.jpg")
data = data[["image_id", target_attr]]
print("✅ Total images used for training:", len(data))

# ========================
# DATA PREPROCESSING
# ========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_dataframe(
    dataframe=data,
    directory=img_dir,
    x_col='image_id',
    y_col=target_attr,
    target_size=(64, 64),
    batch_size=32,
    class_mode='raw',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=data,
    directory=img_dir,
    x_col='image_id',
    y_col=target_attr,
    target_size=(64, 64),
    batch_size=32,
    class_mode='raw',
    subset='validation'
)

# ========================
# MODEL ARCHITECTURE
# ========================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# ========================
# TRAINING
# ========================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3  # Keep small for a quick test
)

# ========================
# SAVE MODEL
# ========================
model.save("celebrity_face_model_small.h5")
print("✅ Model trained and saved as celebrity_face_model_small.h5")

# ========================
# SAVE CLASS INDICES (CLEAN JSON)
# ========================
class_indices = {
    "info": {
        "description": "Class index mapping for the Smiling detection model.",
        "model_name": "celebrity_face_model_small.h5",
        "type": "binary_classification",
        "attribute": "Smiling"
    },
    "classes": {
        "Smiling": 1,
        "Not_Smiling": 0
    },
    "labels": {
        1: "Smiling 😀",
        0: "Not Smiling 😐"
    }
}

# ✅ Convert all keys (even nested ones) to strings
def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(i) for i in obj]
    else:
        return obj

clean_class_indices = convert_keys_to_str(class_indices)

with open("class_indices.json", "w") as f:
    json.dump(clean_class_indices, f, indent=4)

print("✅ class_indices.json created successfully!")
