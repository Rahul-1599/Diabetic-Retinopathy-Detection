"""Diabetic Retinopathy Classification using DenseNet121.
Dataset: amanneo/diabetic-retinopathy-resized-arranged (KaggleHub)
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import kagglehub
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import load_img, img_to_array

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10

# -----------------------------
# 1. Download dataset
# -----------------------------
path = kagglehub.dataset_download("amanneo/diabetic-retinopathy-resized-arranged")
print("Dataset path:", path)

# -----------------------------
# 2. Find class-folder directory
# -----------------------------
def find_class_directory(root_path):
    root_path = Path(root_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    candidates = []

    for folder in root_path.rglob("*"):
        if not folder.is_dir():
            continue
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if len(subdirs) >= 2:
            image_count = 0
            for subdir in subdirs:
                try:
                    image_count += sum(
                        1 for f in subdir.iterdir()
                        if f.is_file() and f.suffix.lower() in image_extensions
                    )
                except PermissionError:
                    pass
            if image_count > 0:
                candidates.append((image_count, folder))

    if not candidates:
        raise ValueError("Could not locate class folders containing images.")

    candidates.sort(reverse=True, key=lambda x: x[0])
    return str(candidates[0][1])


data_dir = find_class_directory(path)
print("Image directory:", data_dir)

# -----------------------------
# 3. Load dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)
print("Number of classes:", num_classes)

# -----------------------------
# 4. Display sample images
# -----------------------------
plt.figure(figsize=(10, 8))
for images, labels in train_ds.take(1):
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")
plt.tight_layout()
plt.show()

# Performance pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# 5. Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# -----------------------------
# 6. DenseNet121 transfer learning
# DenseNet121 was introduced in 2017.
# -----------------------------
base_model = tf.keras.applications.DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.densenet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

model.summary()

# -----------------------------
# 7. Compile
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_dr_densenet121.keras", monitor="val_loss", save_best_only=True
    ),
]

# -----------------------------
# 8. Initial training
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
)

# -----------------------------
# 9. Plot training curves
# -----------------------------
def plot_history(hist, prefix=""):
    plt.figure(figsize=(7, 5))
    plt.plot(hist.history["accuracy"], label="Training Accuracy")
    plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{prefix}Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(hist.history["loss"], label="Training Loss")
    plt.plot(hist.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{prefix}Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()


plot_history(history)

# -----------------------------
# 10. Evaluation
# -----------------------------
val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# -----------------------------
# 11. Classification report + confusion matrix
# -----------------------------
y_true, y_pred = [], []
for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(num_classes), class_names, rotation=45)
plt.yticks(range(num_classes), class_names)
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.show()

# -----------------------------
# 12. Fine-tuning
# -----------------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks,
)

plot_history(history_fine, prefix="Fine-tuning: ")

# Save final model
model.save("diabetic_retinopathy_densenet121_final.keras")
print("Final model saved as diabetic_retinopathy_densenet121_final.keras")

# -----------------------------
# 13. Single-image prediction function
# -----------------------------
def predict_retina(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))

    print("Predicted class:", class_names[predicted_index])
    print(f"Confidence: {prediction[predicted_index] * 100:.2f}%")

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Prediction: {class_names[predicted_index]}")
    plt.show()

    return prediction

# Example:
# predict_retina("path/to/retinal_image.jpeg")
