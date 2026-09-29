# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import json
# import os
# import random
# import matplotlib.pyplot as plt

# # ============================
# # LOAD MODEL
# # ============================
# model = tf.keras.models.load_model("celebrity_face_model_small.h5")
# print("✅ Model loaded successfully!")

# # ============================
# # LOAD LABEL MAPPING
# # ============================
# with open("class_indices.json", "r") as f:
#     class_indices = json.load(f)

# # Extract the label info safely
# label_map = class_indices.get("labels", {})
# label_map = {int(k): v for k, v in label_map.items()}
# print("✅ Loaded label mapping:", label_map)

# # ============================
# # DATASET PATH
# # ============================
# dataset_dir = r"D:\celebA\celebA-dataset\img_align_celeba\img_align_celeba"

# if not os.path.exists(dataset_dir):
#     print("❌ Dataset path not found. Please check the path in your code.")
#     exit()

# # ============================
# # PICK RANDOM IMAGES
# # ============================
# all_images = [f for f in os.listdir(dataset_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
# sample_images = random.sample(all_images, 5)  # pick 5 random images

# print(f"🖼️ Testing on {len(sample_images)} random images...")

# # ============================
# # TEST LOOP
# # ============================
# for img_name in sample_images:
#     img_path = os.path.join(dataset_dir, img_name)
#     img = image.load_img(img_path, target_size=(64, 64))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0) / 255.0

#     # Predict
#     prediction = model.predict(x)[0][0]
#     predicted_class = 1 if prediction >= 0.5 else 0
#     confidence = prediction * 100 if predicted_class == 1 else (100 - prediction * 100)

#     label_text = label_map.get(predicted_class, "Unknown")

#     print(f"📸 {img_name} → {label_text} ({confidence:.2f}% confidence)")

#     # Display the image
#     plt.imshow(image.load_img(img_path))
#     plt.title(f"{label_text} ({confidence:.1f}%)")
#     plt.axis("off")
#     plt.show()
   

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt

# ========================
# LOAD MODEL AND LABELS
# ========================
model = tf.keras.models.load_model("celebrity_face_model_small.h5")
print("✅ Model loaded successfully!")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Load the label mapping properly
label_map = class_indices.get("labels", {})
label_map = {int(k): v for k, v in label_map.items()}
print("✅ Loaded label mapping:", label_map)

# ========================
# IMAGE DIRECTORY
# ========================
img_dir = r"D:\celebA\celebA-dataset\img_align_celeba\img_align_celeba"
all_images = os.listdir(img_dir)
sample_images = random.sample(all_images, 5)
print(f"🎯 Testing on {len(sample_images)} random images...")

# ========================
# TEST EACH IMAGE
# ========================
for img_name in sample_images:
    img_path = os.path.join(img_dir, img_name)

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(x)[0][0]
    predicted_class = 1 if prediction >= 0.5 else 0
    confidence = prediction * 100 if predicted_class == 1 else (100 - prediction * 100)

    print(f"{img_name} → {label_map[predicted_class]} ({confidence:.2f}% confidence)")

    # Show the image with title
    plt.imshow(image.load_img(img_path))
    plt.title(f"{label_map[predicted_class]} ({confidence:.1f}% confidence)")
    plt.axis("off")

    # Show for 3 seconds then close
    plt.show(block=False)
    plt.pause(3)
    plt.close()

print("✅ Done testing random images!")


