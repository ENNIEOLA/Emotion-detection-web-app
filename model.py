pimport tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Create model
model = keras.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Quick training
X = np.random.rand(100, 48, 48, 1).astype('float32')
y = keras.utils.to_categorical(np.random.randint(0, 7, 100), 7)
model.fit(X, y, epochs=1, verbose=0)

# Save
model.save('emotion_model.keras')
print("Model saved successfully!")