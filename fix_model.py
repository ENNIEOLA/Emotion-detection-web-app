import urllib.request
import os

print("Downloading properly trained emotion model...")
print("This will take 1-2 minutes...")

url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"

try:
    urllib.request.urlretrieve(url, "emotion_model_trained.h5")
    
    # Backup old model
    if os.path.exists('emotion_model.keras'):
        os.rename('emotion_model.keras', 'emotion_model_old.keras')
    
    # Use new model
    os.rename('emotion_model_trained.h5', 'emotion_model.h5')
    
    print("✅ Real trained model installed!")
    print("✅ Old model backed up as 'emotion_model_old.keras'")
    print("\nNow restart your app and test!")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("\nTry manual download:")
    print("1. Go to: https://github.com/oarriaga/face_classification")
    print("2. Download a pre-trained model")
    print("3. Rename it to 'emotion_model.h5'")