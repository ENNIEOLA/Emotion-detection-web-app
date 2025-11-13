import urllib.request
import sys
import os

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()

print("="*60)
print("DOWNLOADING WORKING EMOTION MODEL")
print("="*60)

# Delete corrupted file
if os.path.exists('emotion_model.h5'):
    print("\nRemoving corrupted file...")
    os.remove('emotion_model.h5')

# Try alternative download URL
print("\nDownloading from alternative source...")
url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

try:
    urllib.request.urlretrieve(url, "emotion_model.h5", download_progress)
    print("\n\n✅ Model downloaded successfully!")
    
    # Check file size
    size = os.path.getsize('emotion_model.h5') / (1024*1024)
    print(f"File size: {size:.2f} MB")
    
    if size < 0.5:
        print("⚠️ File seems too small. Download may have failed.")
    else:
        print("✅ File size looks good!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTrying wget method...")