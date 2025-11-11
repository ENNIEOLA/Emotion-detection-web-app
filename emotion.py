# emotion.py - Fixed version for Windows
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import sys
import io

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_emotion_model():
    """
    Creates a lightweight CNN model for emotion detection
    Input: 48x48 grayscale images
    Output: 7 emotion classes
    """
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(48, 48, 1)),
        
        # First Conv Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Conv Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Conv Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dummy_training_data():
    """
    Creates synthetic data for demonstration
    Replace with real FER2013 dataset for actual training
    """
    X_train = np.random.rand(1000, 48, 48, 1).astype('float32')
    y_train = keras.utils.to_categorical(np.random.randint(0, 7, 1000), 7)
    
    X_val = np.random.rand(200, 48, 48, 1).astype('float32')
    y_val = keras.utils.to_categorical(np.random.randint(0, 7, 200), 7)
    
    return X_train, y_train, X_val, y_val

def train_model_quick():
    """
    Quick training function
    """
    print("Creating model...")
    model = create_emotion_model()
    
    print("\nModel Summary:")
    model.summary()
    
    print("\nGenerating training data...")
    X_train, y_train, X_val, y_val = create_dummy_training_data()
    
    print("\nTraining model...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=64,
        verbose=1
    )
    
    # Save model - use .keras format (no special characters in print)
    print("\nSaving model...")
    model.save('emotion_model.keras')
    print("Model saved as 'emotion_model.keras'")
    
    return model

if __name__ == "__main__":
    print("="*60)
    print("EMOTION DETECTION MODEL TRAINER")
    print("="*60)
    print("\nNOTE: This creates a working model structure.")
    print("For production use, train with real FER2013 dataset!\n")
    
    model = train_model_quick()
    
    print("\n" + "="*60)
    print("Model ready to use!")
    print("="*60)