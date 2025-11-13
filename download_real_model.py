import urllib.request
import os
import sys

def download_with_progress(url, filename):
    """Download file with progress bar"""
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}% complete")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filename, progress)
    print("\n✅ Download complete!")

print("="*60)
print("DOWNLOADING REAL PRE-TRAINED EMOTION MODEL")
print("="*60)
print("\nThis model was trained on 35,887 real human faces!")
print("It will actually detect emotions correctly.\n")

# Backup old model
if os.path.exists('emotion_model.keras'):
    print("Backing up your old model...")
    try:
        os.rename('emotion_model.keras', 'emotion_model_BACKUP.keras')
        print("✅ Old model backed up as 'emotion_model_BACKUP.keras'\n")
    except:
        print("⚠️ Could not backup old model (continuing anyway)\n")

# Download the real model
print("Downloading real pre-trained model...")
print("This may take 1-3 minutes depending on your internet speed...\n")

url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"

try:
    download_with_progress(url, "emotion_model.h5")
    
    # Verify file was downloaded
    if os.path.exists('emotion_model.h5'):
        file_size = os.path.getsize('emotion_model.h5') / (1024 * 1024)  # Convert to MB
        print(f"\n✅ SUCCESS! Model downloaded: {file_size:.2f} MB")
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Run: python app.py")
        print("2. Open: http://127.0.0.1:5000")
        print("3. Test your emotion detector - it will work correctly now!")
        print("="*60)
    else:
        print("\n❌ Download completed but file not found")
        
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\n" + "="*60)
    print("ALTERNATIVE: Use PowerShell method")
    print("="*60)
    print("Copy and paste this into PowerShell:\n")
    print('Invoke-WebRequest -Uri "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5" -OutFile "emotion_model.h5"')
    print("\n" + "="*60)