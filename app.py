# app.py - WINDOWS COMPATIBLE VERSION (NO UNICODE)
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import base64
import sys
import io

# Download model if not exists (for Render)
import urllib.request

if not os.path.exists('emotion_model.h5'):
    print("Downloading emotion model for first deployment...")
    try:
        url = "https://github.com/oarriaga/face_classification/raw/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
        urllib.request.urlretrieve(url, "emotion_model.h5")
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Warning: Could not download model - {e}")

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load emotion detection model
print("\n" + "="*60)
print("LOADING EMOTION DETECTION MODEL")
print("="*60)

model = None
try:
    if os.path.exists('emotion_model.h5'):
        model = load_model('emotion_model.h5', compile=False)
        print("SUCCESS: REAL MODEL loaded from emotion_model.h5")
    elif os.path.exists('emotion_model.keras'):
        model = load_model('emotion_model.keras', compile=False)
        print("WARNING: Using basic model from emotion_model.keras")
    else:
        print("ERROR: No model file found!")
except Exception as e:
    print(f"ERROR: Could not load model - {e}")
    model = None

# Load face cascade
print("\nLoading face detection cascade...")
face_cascade = None

try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print(f"WARNING: Failed to load from {cascade_path}")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if not face_cascade.empty():
        print("SUCCESS: Face cascade loaded")
    else:
        print("ERROR: Could not load face cascade!")
        
except Exception as e:
    print(f"ERROR: Face cascade loading failed - {e}")

print("="*60 + "\n")

# Initialize database
def init_db():
    try:
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detections
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      image_path TEXT,
                      emotion TEXT NOT NULL,
                      confidence REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        print("SUCCESS: Database initialized\n")
    except Exception as e:
        print(f"ERROR: Database initialization failed - {e}\n")

init_db()

def detect_emotion(image_path):
    """Detect emotion from image"""
    print(f"\n{'='*60}")
    print(f"DETECTING EMOTION FROM: {image_path}")
    print(f"{'='*60}")
    
    if model is None:
        print("ERROR: Model not loaded!")
        return None, 0
    
    if face_cascade is None or face_cascade.empty():
        print("ERROR: Face cascade not loaded!")
        return None, 0
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Could not read image")
        return None, 0
    
    print(f"SUCCESS: Image loaded - Shape: {img.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    print("SUCCESS: Image preprocessed")
    
    # Try multiple detection attempts
    detection_attempts = [
        (1.3, 5, (30, 30), "Standard"),
        (1.2, 4, (25, 25), "Relaxed"),
        (1.15, 3, (20, 20), "Sensitive"),
        (1.1, 3, (20, 20), "Very sensitive"),
        (1.08, 2, (15, 15), "Aggressive"),
        (1.05, 2, (10, 10), "Maximum"),
    ]
    
    faces = []
    for i, (scale, neighbors, min_size, desc) in enumerate(detection_attempts, 1):
        print(f"Attempt {i} ({desc})...", end=" ")
        
        faces = face_cascade.detectMultiScale(
            gray_equalized, scaleFactor=scale, minNeighbors=neighbors,
            minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            print(f"SUCCESS: Found {len(faces)} face(s)!")
            break
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            print(f"SUCCESS: Found {len(faces)} face(s)!")
            break
        
        print("No faces")
    
    if len(faces) == 0:
        print(f"\n{'='*60}")
        print("FINAL RESULT: No faces detected")
        print(f"{'='*60}\n")
        return None, 0
    
    print(f"\nSUCCESS: Face detection complete!")
    
    # Get largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    print(f"Using face at position ({x},{y}), size {w}x{h}")
    
    # Extract and process face
    face_roi = gray_equalized[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (64, 64))
    face_normalized = face_resized.astype('float32') / 255.0
    face_input = np.reshape(face_normalized, (1, 64, 64, 1))
    
    print(f"Face preprocessed for model: {face_input.shape}")
    
    # Predict emotion
    print("\nPredicting emotion...")
    predictions = model.predict(face_input, verbose=0)
    
    emotion_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][emotion_idx]) * 100
    detected_emotion = emotion_labels[emotion_idx]
    
    # Show probabilities
    print("\nPrediction probabilities:")
    for i, label in enumerate(emotion_labels):
        prob = predictions[0][i] * 100
        print(f"   {label:10s}: {prob:5.2f}%")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULT: {detected_emotion} (Confidence: {confidence:.2f}%)")
    print(f"{'='*60}\n")
    
    return detected_emotion, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        name = request.form.get('name', 'Anonymous')
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\nImage saved to: {filepath}")
        
        emotion, confidence = detect_emotion(filepath)
        
        if emotion is None:
            return jsonify({
                'error': 'No face detected. Please ensure your face is clearly visible with good lighting.'
            }), 400
        

        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute("INSERT INTO detections (name, image_path, emotion, confidence) VALUES (?, ?, ?, ?)",
                  (name, filepath, emotion, confidence))
        conn.commit()
        conn.close()
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'name': name
        })
    
    except Exception as e:
        print(f"\nERROR in upload: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/webcam', methods=['POST'])
def webcam_capture():
    try:
        name = request.form.get('name', 'Anonymous')
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(f"{name}_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        print(f"\nWebcam image saved to: {filepath}")
        
        emotion, confidence = detect_emotion(filepath)
        
        if emotion is None:
            return jsonify({
                'error': 'No face detected. Please position your face clearly in the circle.'
            }), 400
        
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute("INSERT INTO detections (name, image_path, emotion, confidence) VALUES (?, ?, ?, ?)",
                  (name, filepath, emotion, confidence))
        conn.commit()
        conn.close()
        
        return jsonify({
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'name': name
        })
    
    except Exception as e:
        print(f"\nERROR in webcam: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/history')
def history():
    try:
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute("SELECT name, emotion, confidence, timestamp FROM detections ORDER BY timestamp DESC LIMIT 10")
        results = c.fetchall()
        conn.close()
        
        history_data = [
            {'name': r[0], 'emotion': r[1], 'confidence': r[2], 'timestamp': r[3]}
            for r in results
        ]
        
        return jsonify(history_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("\n" + "="*60)
    print("EMOTION DETECTION WEB APP")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop")
    print("="*60 + "\n")

    # Get port from environment variable (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    print(f"Running on port: {port}")
    print("="*60 + "\n")
    
    # Run app (debug=False for production)
    app.run(debug=False, host='0.0.0.0', port=port)
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
        os.environ.get('PORT', 5000)
        if __name__ == '__main__':

              os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("\n" + "="*60)
    print("EMOTION DETECTION WEB APP")
    print("="*60)
    print("\nStarting server...")
    
    # Get port from environment variable (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    print(f"Running on port: {port}")
    print("="*60 + "\n")
    
    # Run app (debug=False for production)
    app.run(debug=False, host='0.0.0.0', port=port)