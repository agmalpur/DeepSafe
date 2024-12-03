import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mtcnn import MTCNN
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from skimage.util import random_noise
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter

# Data Preparation
metadata_path = '/content/metadata.json'
video_folder = '/content/'
train_sample_metadata = pd.read_json(metadata_path).T

# Plotting data distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=train_sample_metadata, x='label')
plt.title('Distribution of Fake vs. Real Videos')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

detector = MTCNN()
train_metadata, val_metadata = train_test_split(train_sample_metadata, test_size=0.2, random_state=42)

class VideoFrameGenerator(Sequence):
    def __init__(self, metadata, batch_size=32, target_size=(224, 224), shuffle=True):
        self.metadata = metadata
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.metadata))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_metadata = self.metadata.iloc[batch_indexes]
        X, y_labels = self.__data_generation(batch_metadata)
        return X, y_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_metadata):
        X, y_labels = [], []
        for video_name, row in batch_metadata.iterrows():
            video_path = os.path.join(video_folder, video_name)
            label = 1 if row['label'] == 'FAKE' else 0
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(frame_rgb)
                for face in faces:
                    x, y, width, height = face['box']
                    face_img = frame_rgb[y:y + height, x:x + width]
                    face_img = cv2.resize(face_img, self.target_size)
                    face_array = face_img / 255.0
                    X.append(face_array)
                    y_labels.append(label)
                    if len(X) >= self.batch_size:
                        cap.release()
                        return np.array(X), np.array(y_labels)
            cap.release()
        while len(X) < self.batch_size:
            X.append(X[0])
            y_labels.append(y_labels[0])
        return np.array(X), np.array(y_labels)

batch_size = 32
train_generator = VideoFrameGenerator(train_metadata, batch_size=batch_size)
val_generator = VideoFrameGenerator(val_metadata, batch_size=batch_size)

# Model Architecture
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freezing the base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  # For demonstration, adjust epochs as needed
    batch_size=batch_size
)

# Evaluation Metrics
val_X, val_y = next(iter(val_generator))
val_predictions = (model.predict(val_X) > 0.5).astype(int)
precision = precision_score(val_y, val_predictions)
recall = recall_score(val_y, val_predictions)
f1 = f1_score(val_y, val_predictions)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save Model
model_save_path = '/content/efficientnet_deepfake_detection.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Adversarial Testing
classifier = KerasClassifier(model=model)

def noise_injection(images, noise_level=0.05):
    return np.clip([random_noise(img, var=noise_level**2) for img in images], 0, 1)

def pixel_modification(images, num_pixels=10):
    modified_images = images.copy()
    for img in modified_images:
        for _ in range(num_pixels):
            x, y = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
            img[x, y] = np.random.rand(3)
    return modified_images

def compression_and_blurring(images, sigma=1):
    compressed_blurred_images = []
    for img in images:
        blurred = gaussian_filter(img, sigma=sigma)
        compressed = img_as_ubyte(blurred)
        compressed_blurred_images.append(compressed / 255.0)
    return np.array(compressed_blurred_images)

def frame_manipulation(images, brightness_offset=50):
    manipulated_images = images.copy()
    for i in range(len(manipulated_images)):
        manipulated_images[i] = np.clip(manipulated_images[i] + (brightness_offset / 255.0), 0, 1)
    return manipulated_images

attacks = {
    "FGSM": FastGradientMethod(estimator=classifier, eps=0.1),
    "Noise Injection": lambda x: noise_injection(x),
    "Pixel Modification": lambda x: pixel_modification(x),
    "Compression & Blurring": lambda x: compression_and_blurring(x),
    "Frame Manipulation": lambda x: frame_manipulation(x),
}

for attack_name, attack in attacks.items():
    if attack_name == "FGSM":
        adversarial_images = attack.generate(X=val_X)
    else:
        adversarial_images = attack(val_X)

    adversarial_predictions = (model.predict(adversarial_images) > 0.5).astype(int)
    adv_accuracy = accuracy_score(val_y, adversarial_predictions)
    adv_precision = precision_score(val_y, adversarial_predictions)
    adv_recall = recall_score(val_y, adversarial_predictions)
    adv_f1 = f1_score(val_y, adversarial_predictions)

    print(f"--- {attack_name} ---")
    print(f"Accuracy: {adv_accuracy:.4f}")
    print(f"Precision: {adv_precision:.4f}")
    print(f"Recall: {adv_recall:.4f}")
    print(f"F1-Score: {adv_f1:.4f}")

# Retraining with adversarial examples
X_train_combined = np.concatenate([train_generator[0][0], attacks["FGSM"].generate(X=train_generator[0][0])])
y_train_combined = np.concatenate([train_generator[0][1], train_generator[0][1]])

model.fit(X_train_combined, y_train_combined, epochs=10, batch_size=32)

# Save Adversarially Robust Model
model_save_path_adv = '/content/efficientnet_deepfake_detection_robust.h5'
model.save(model_save_path_adv)
print(f"Adversarially robust model saved to {model_save_path_adv}")
