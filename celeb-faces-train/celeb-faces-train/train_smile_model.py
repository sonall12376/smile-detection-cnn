import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# === PATHS ===
DATA_DIR = r"D:\celebA\celeba-dataset\img_align_celeba"
ATTR_PATH = r"D:\celebA\celeba-dataset\list_attr_celeba.csv"

# === LOAD ATTRIBUTES ===
attrs = pd.read_csv(ATTR_PATH)
print(f"✅ Attributes loaded: {attrs.shape}")

# Keep only filename and the 'Smiling' column
data = attrs[['image_id', 'Smiling']]
data['Smiling'] = data['Smiling'].apply(lambda x: 1 if x == 1 else 0)

# For speed, we’ll train on a smaller subset first (e.g. 5000 images)
data = data.sample(5000, random_state=42)

# === IMAGE DATA GENERATOR ===
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255
)

train_gen = datagen.flow_from_dataframe(
    dataframe=data,
    directory=DATA_DIR,
    x_col='image_id',
    y_col='Smiling',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_dataframe(
    dataframe=data,
    directory=DATA_DIR,
    x_col='image_id',
    y_col='Smiling',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# === BUILD A SIMPLE CNN ===
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === TRAIN ===
checkpoint = ModelCheckpoint('smile_model.h5', monitor='val_accuracy', save_best_only=True)
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    callbacks=[checkpoint]
)

print("🎉 Training complete! Best model saved as 'smile_model.h5'")
