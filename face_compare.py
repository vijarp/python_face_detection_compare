import cv2
import numpy as np
import os
def detect_face(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Load OpenCV pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Extract the face regions
    face_encodings = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        face_encodings.append(face_region)
    return face_encodings

def compare_faces(image_path1, image_path2):
    # Detect faces in both images
    faces1 = detect_face(image_path1)
    faces2 = detect_face(image_path2)
    
    if not faces1 or not faces2:
        print("No faces found in one or both images.")
        return False
    
    # Compare faces using correlation coefficient
    result = []
    for face1 in faces1:
        for face2 in faces2:
            res = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)
            if np.max(res) > 0.6:  # Threshold for matching
                result.append(True)
            else:
                result.append(False)
    
    return result

# Paths to your images
image_path1 = 'images/v1.jpeg'
image_path2 = 'images/v2.jpg'
#image_path2 = 'images/drdilip.png'

# Compare faces
comparison_result = compare_faces(image_path1, image_path2)
print(comparison_result)

if  comparison_result != False:
    print("The faces match!")
else:
    print("The faces do not match.")
