import cv2
import sys

# Test if face cascade works
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print("❌ ERROR: Could not load face cascade!")
    print(f"Path tried: {face_cascade_path}")
    sys.exit(1)
else:
    print("✅ Face cascade loaded successfully!")

# Test on an image
image_path = input("Enter path to test image: ")
img = cv2.imread(image_path)

if img is None:
    print(f"❌ Could not read image: {image_path}")
    sys.exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Try detection
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
print(f"Detected {len(faces)} faces with default settings")

if len(faces) == 0:
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(30, 30))
    print(f"Detected {len(faces)} faces with relaxed settings")

if len(faces) > 0:
    print("✅ Face detection working!")
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite('test_output.jpg', img)
    print("Saved result to test_output.jpg")
else:
    print("❌ No faces detected - image might need better lighting or clearer face")