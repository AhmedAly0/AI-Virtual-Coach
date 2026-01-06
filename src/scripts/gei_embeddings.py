# train.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_loader import load_data_and_groups
from model import build_arcface_model

# Parameters
DATA_PATHS = [
    'preprocessed/GEIs_of_rgb_front/geis/',
    'preprocessed/GEIs_of_rgb_side/geis/'
]
IMG_SIZE = (128, 128)
NUM_CLASSES = 15
BATCH_SIZE = 32
EPOCHS = 50

# Load Data
X, y, groups = load_data_and_groups(DATA_PATHS, IMG_SIZE)

# Convert grayscale → RGB
X = np.expand_dims(X, axis=-1)   # (N, H, W, 1)
X = np.repeat(X, 3, axis=-1)     # (N, H, W, 3)
print("Final dataset shape:", X.shape)

# Split by volunteer → test set
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_val_idx, test_idx = next(gss_test.split(X, y, groups))
X_train_val, y_train_val, groups_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Split train/val
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))
X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# Save train/test splits for evaluation later
np.savez("gei_train_data.npz", X_train=X_train, y_train=y_train)
np.savez("gei_test_data.npz", X_test=X_test, y_test=y_test)

# Build ArcFace model
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
model, embedding_model = build_arcface_model(input_shape, NUM_CLASSES)

# Compile
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=loss_fn,
              metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7)

# Train
history = model.fit(
    [X_train, y_train], y_train,
    validation_data=([X_val, y_val], y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr]
)

# Save embedding model only (for inference)
embedding_model.save("gei_arcface_embedding.h5")
print("\n✅ Training complete. Embedding model saved as gei_arcface_embedding.h5")
