# app.py - Complete Emotion Detection Web App
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load emotion detection model
try:
    if os.path.exists('emotion_model.keras'):
        model = load_model('emotion_model.keras')
        print("Model loaded successfully from emotion_model.keras")
    elif os.path.exists('emotion_model.h5'):
        model = load_model('emotion_model.h5')
        print("Model loaded successfully from emotion_model.h5")
    else:
        print("ERROR: No model file found!")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load face cascade
# Try multiple methods to load face cascade
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Verify it loaded correctly
if face_cascade.empty():
    print("ERROR: Could not load face cascade classifier!")
    print(f"Tried path: {face_cascade_path}")
else:
    print("Face cascade loaded successfully!")


# Initialize database
def init_db():
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
    print("Database initialized successfully")

init_db()

def detect_emotion(image_path):
    """Detect emotion from image with better face detection"""
    if model is None:
        return None, 0
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image at {image_path}")
        return None, 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Try multiple scale factors for better detection
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  # More sensitive
        minNeighbors=3,    # Less strict
        minSize=(30, 30)   # Smaller minimum face size
    )
    
    print(f"Detected {len(faces)} face(s) in image")
    
    if len(faces) == 0:
        # Try even more aggressive detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20)
        )
        print(f"Second attempt: Detected {len(faces)} face(s)")
    
    if len(faces) == 0:
        return None, 0
    
    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
    
    # Extract and process face
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    
    # Predict emotion
    predictions = model.predict(face, verbose=0)
    emotion_idx = np.argmax(predictions)
    confidence = float(predictions[0][emotion_idx]) * 100
    
    print(f"Predicted emotion: {emotion_labels[emotion_idx]} ({confidence:.2f}%)")
    
    return emotion_labels[emotion_idx], confidence

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
            return jsonify({'error': 'No selected file'}), 400
        
        # Create uploads folder if not exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save image
        filename = secure_filename(f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect emotion
        emotion, confidence = detect_emotion(filepath)
        
        if emotion is None:
            return jsonify({'error': 'No face detected in image. Please ensure your face is clearly visible.'}), 400
        
        # Save to database
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
        print(f"Error in upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webcam', methods=['POST'])
def webcam_capture():
    try:
        name = request.form.get('name', 'Anonymous')
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Create uploads folder if not exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(f"{name}_webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        # Detect emotion
        emotion, confidence = detect_emotion(filepath)
        
        if emotion is None:
            return jsonify({'error': 'No face detected. Please position your face clearly in front of the camera.'}), 400
        
        # Save to database
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
        print(f"Error in webcam: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history')
def history():
    """Get detection history"""
    try:
        conn = sqlite3.connect('emotions.db')
        c = conn.cursor()
        c.execute("SELECT name, emotion, confidence, timestamp FROM detections ORDER BY timestamp DESC LIMIT 10")
        results = c.fetchall()
        conn.close()
        
        history_data = [
            {
                'name': r[0],
                'emotion': r[1],
                'confidence': r[2],
                'timestamp': r[3]
            } for r in results
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
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)