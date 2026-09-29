# Smile Detection CNN

A Convolutional Neural Network (CNN) built with Keras/TensorFlow to perform binary classification detecting whether a person is smiling or not in facial images.

## Tech Stack & Dependencies

> **Note**: No `requirements.txt` file is included in this repository. Dependencies identified directly from Python import statements:

- **Language**: Python 3
- **Deep Learning**: TensorFlow / Keras (`Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `Adam`, `ImageDataGenerator`, `ModelCheckpoint`)
- **Data Manipulation**: Pandas (`pd.read_csv`, DataFrame filtering), NumPy
- **Visualization & Utility**: Matplotlib (`pyplot`), JSON, OS, Random

## Dataset

- **Name**: CelebA (CelebFaces Attributes Dataset)
- **Source**: `celeba-dataset` (`list_attr_celeba.csv` and `img_align_celeba` directory)
- **Size**: 
  - Full dataset attribute file contains **202,599** image annotations.
  - The training scripts (`train_model.py` and `train_smile_model.py`) sample a subset of **5,000 images** for speed, split into **80% training** (4,000 images) and **20% validation** (1,000 images).

## Approach & Pipeline

1. **Preprocessing**:
   - Parses `list_attr_celeba.csv` to extract the `Smiling` binary attribute (mapping `-1`/`1` format to `0`/`1`).
   - Normalizes image pixels by scaling by `1/255`.
   - Resizes images to `64x64` (`train_model.py`) or `128x128` (`train_smile_model.py`).
2. **Model Architecture**:
   - Sequential CNN consisting of 2 to 3 Convolutional blocks (`Conv2D` + `MaxPooling2D`).
   - Dense fully connected layer with 128 units and ReLU activation.
   - `Dropout(0.5)` for regularization.
   - Final output node (`Dense(1)`) with Sigmoid activation for binary classification.
3. **Training Configuration**:
   - Loss Function: Binary Cross-Entropy (`binary_crossentropy`)
   - Optimizer: Adam (`lr=0.001` or `0.0001`)
   - Metric: Accuracy (`accuracy`)
   - Epochs: 3 to 5 epochs

## Results

**Results not logged in repo** (No execution logs, evaluation metrics, or training history plots are saved in the repository).

## How to Run

### 1. Installation
Install required dependencies:
```bash
pip install tensorflow pandas numpy matplotlib
```

### 2. Dataset Setup
Ensure the CelebA dataset is unzipped and update the `DATA_DIR` / `img_dir` and `attr_path` variables in the scripts to point to your local dataset directory.

### 3. Training
Train the model:
```bash
python celeb-faces-train/train_smile_model.py
# or
python celeb-faces-train/train_model.py
```

### 4. Inference / Evaluation
Run prediction on random sample images:
```bash
python celeb-faces-train/test_model.py
```

## Project Structure

```text
smile-detection-cnn/
├── celeb-faces-train/
│   ├── train_smile_model.py        # Model training script (128x128 input, 5 epochs)
│   ├── train_model.py              # Model training script (64x64 input, 3 epochs) + metadata generator
│   ├── test_model.py               # Visual inference script on random dataset images
│   ├── smile_model.h5              # Saved model weights
│   ├── celebrity_face_model_small.h5 # Saved model weights
│   ├── class_indices.json          # Class label mapping JSON file
│   └── test_images/                # Sample test images
└── celebA/
    └── celeba-dataset/             # CelebA dataset metadata CSVs & images
```