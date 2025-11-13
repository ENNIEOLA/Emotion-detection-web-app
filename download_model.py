import urllib.request
import sys
import os

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()

print("Downloading emotion detection model...")
print("This may take 1-2 minutes...\n")

url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

try:
    urllib.request.urlretrieve(url, "emotion_model.h5", download_progress)
    print("\n\n✅ Model downloaded successfully!")
    
    # Check size
    size = os.path.getsize('emotion_model.h5') / (1024*1024)
    print(f"File size: {size:.2f} MB")
    
    if size > 0.5:
        print("✅ Download successful! You can now run: python app.py")
    else:
        print("❌ File too small - download may have failed")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTry Method 2 (requests library)")