#!/usr/bin/env python3

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os

# Optional: MTCNN for robust face detection (install with `pip install facenet-pytorch`)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

# Define model parameters
NUM_CLASSES = 7  # AffectNet7 has 7 classes
MODEL_PATH = './ckpts/fer.pth.tar'
DEVICE = torch.device("cpu")  # Force CPU (change to "cuda" if GPU desired)

# Expression labels (adjust based on your AffectNet7 dataset ordering)
EXPRESSION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger']

# Load the trained model
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model

# Preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # VGGface2 norm
])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MTCNN face detector (if available)
if MTCNN_AVAILABLE:
    mtcnn = MTCNN(image_size=224, margin=0, min_face_size=10, device=DEVICE)

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = Image.fromarray(face_img)
    face_img = transform(face_img)
    face_img = face_img.unsqueeze(0)  # Add batch dimension
    return face_img

def predict_expression(model, face_tensor):
    with torch.no_grad():
        face_tensor = face_tensor.to(DEVICE)
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.squeeze().cpu().numpy()

def detect_faces_haar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.01,  # Very fine scale step
        minNeighbors=1,    # Very lenient
        minSize=(10, 10)   # Smallest possible face
    )
    return faces

def detect_faces_mtcnn(img):
    if not MTCNN_AVAILABLE:
        return []
    boxes, _ = mtcnn.detect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if boxes is None:
        return []
    return [(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])) for b in boxes]

def main(image_path):
    # Load the trained model
    model = load_model()
    print("Model loaded successfully.")

    # Load and process the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print(f"Image dimensions: {img.shape} (Height, Width, Channels)")

    # Try Haar Cascade first
    faces = detect_faces_haar(img)
    detection_method = "Haar Cascade"

    # Fallback to MTCNN if Haar fails and MTCNN is available
    if len(faces) == 0 and MTCNN_AVAILABLE:
        faces = detect_faces_mtcnn(img)
        detection_method = "MTCNN"
        print("No faces detected with Haar Cascade, trying MTCNN...")

    if len(faces) == 0:
        print("No faces detected in the image with either method.")
        cv2.imwrite('debug_no_faces_' + os.path.basename(image_path), img)
        print(f"Saved original image for debugging as 'debug_no_faces_{os.path.basename(image_path)}'")
        return

    print(f"Faces detected using {detection_method}: {len(faces)}")

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        # Ensure coordinates are within image bounds
        x, y = max(0, x), max(0, y)
        w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)
        if w <= 0 or h <= 0:
            continue

        face_img = img[y:y+h, x:x+w]
        face_tensor = preprocess_face(face_img)

        # Predict expression (get all probabilities)
        probabilities = predict_expression(model, face_tensor)
        
        # Print all labels and their confidence scores
        print(f"Face {i+1}:")
        for label, prob in zip(EXPRESSION_LABELS, probabilities):
            confidence_percent = prob * 100
            print(f"  {label}: {confidence_percent:.2f}%")

        # Get the top prediction for the annotation
        top_idx = np.argmax(probabilities)
        top_expression = EXPRESSION_LABELS[top_idx]
        top_confidence = probabilities[top_idx] * 100

        # Draw rectangle and top prediction on the image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{top_expression}: {top_confidence:.2f}%"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image
    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved as {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict facial expression from an image")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()
    main(args.image)