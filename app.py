from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'car_classifier_model.h5'
model = load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(image_path, img_size=128):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image file")
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# API Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    upload_folder = 'uploads'
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img = preprocess_image(file_path)

        # Make prediction
        prediction = model.predict(img)[0][0]  # Get the raw score

        # Get class label, confidence score, and raw prediction score
        class_label = 'Car' if prediction > 0.5 else 'Not a Car'
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)

        # Return JSON result
        return jsonify({
            'label': class_label,
            'confidence': confidence,
            'score': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
