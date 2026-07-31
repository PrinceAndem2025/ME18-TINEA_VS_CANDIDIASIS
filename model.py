"""
ME18: Tinea vs Candidiasis
--------------------------------------------
Trains a binary image classifier using MobileNetV2 transfer learning.

Run `python prepare_data.py` FIRST to create me18_train/, me18_val/, me18_test/.

Output files (used by app.py):
  - me18_skin_classifier.keras
  - class_names.json
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16          # small batch size works better for small datasets
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
FINE_TUNE_AT_LAYER = 100  # unfreeze from this layer index onward
MODEL_OUT = "me18_skin_classifier.keras"
CLASS_NAMES_OUT = "class_names.json"

TRAIN_DIR = "me18_train"
VAL_DIR = "me18_val"
TEST_DIR = "me18_test"

# ---------------------------------------------------------------------
# 1. LOAD DATASETS
# ---------------------------------------------------------------------
print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = train_ds.class_names  # e.g. ['Candidiasis', 'Tinea']
print("Classes:", class_names)

with open(CLASS_NAMES_OUT, "w") as f:
    json.dump(class_names, f)

# ---------------------------------------------------------------------
# 2. COMPUTE CLASS WEIGHTS (handles any class imbalance automatically)
# ---------------------------------------------------------------------
train_labels = np.concatenate([y.numpy() for _, y in train_ds]).flatten()
class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
class_weight = {i: w for i, w in enumerate(class_weight_values)}
print("Class weights:", class_weight, "-> maps to", class_names)

# ---------------------------------------------------------------------
# 3. PERFORMANCE: cache + prefetch
# ---------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ---------------------------------------------------------------------
# 4. DATA AUGMENTATION
#    Skin condition photos vary a lot in lighting/angle/zoom in the wild,
#    so augmentation here also helps the model generalise beyond the
#    exact photography conditions of the training set.
# ---------------------------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
    layers.RandomBrightness(0.10),
], name="data_augmentation")

# ---------------------------------------------------------------------
# 5. BUILD THE MODEL (MobileNetV2 transfer learning)
# ---------------------------------------------------------------------
preprocess_input = keras.applications.mobilenet_v2.preprocess_input

base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # freeze for initial training phase

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------
# 6. CALLBACKS
# ---------------------------------------------------------------------
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
)

# ---------------------------------------------------------------------
# 7. PHASE 1: TRAIN WITH FROZEN BASE MODEL
# ---------------------------------------------------------------------
print("\n--- Phase 1: training with frozen base model ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy (before fine-tuning): {test_acc * 100:.2f}%")
acc_before = test_acc

# ---------------------------------------------------------------------
# 8. PHASE 2: FINE-TUNE (unfreeze top layers of the base model)
# ---------------------------------------------------------------------
print("\n--- Phase 2: fine-tuning top layers ---")
base_model.trainable = True

for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # much lower LR
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
)

test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy (after fine-tuning): {test_acc * 100:.2f}%")
print(f"Before fine-tuning was: {acc_before * 100:.2f}%")
print("Fine-tuning helped." if test_acc > acc_before else "Fine-tuning did not help; keeping best weights from EarlyStopping.")

# ---------------------------------------------------------------------
# 9. DETAILED EVALUATION
# ---------------------------------------------------------------------
y_true = np.concatenate([y.numpy() for _, y in test_ds]).flatten()
y_pred_probs = model.predict(test_ds).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

# ---------------------------------------------------------------------
# 10. SAVE MODEL + CLASS NAMES
# ---------------------------------------------------------------------
model.save(MODEL_OUT)
with open(CLASS_NAMES_OUT, "w") as f:
    json.dump(class_names, f)

print(f"\nSaved: {MODEL_OUT} and {CLASS_NAMES_OUT}")
print("Class order:", class_names)
