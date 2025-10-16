from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import time
from pathlib import Path
import google.generativeai as genai

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Increase max upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Configure Gemini
GEMINI_API_KEY = "AIzaSyDvUYaiJzsH3CgYMogcS6BY3kPVzyV69hI"  # Your API key
gemini_model = None

try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Use gemini-2.5-flash (stable, fast, free tier)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✓ Gemini AI configured with gemini-2.5-flash")
except Exception as e:
    print(f"⚠ Gemini configuration failed: {str(e)}")
    print("Chat feature will be disabled.")

# Load YOLOv8 model
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

# Class names from VinBigData dataset
CLASS_NAMES = [
    'Aortic enlargement', 'Atelectasis', 'Calcification',
    'Cardiomegaly', 'Consolidation', 'Edema', 'Emphysema',
    'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
    'Pleural thickening', 'Pneumothorax', 'No finding'
]

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("=" * 50)
    print("Received prediction request")
    try:
        if 'file' not in request.files:
            print("ERROR: No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("ERROR: Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        print(f"Processing file: {file.filename}")

        # Read and process image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("ERROR: Could not decode image")
            return jsonify({'error': 'Invalid image file'}), 400

        print(f"Image shape: {img.shape}")

        # Run inference
        print("Running inference...")
        start_time = time.time()
        results = model(img, conf=0.25)  # Confidence threshold
        inference_time = int((time.time() - start_time) * 1000)
        print(f"Inference completed in {inference_time}ms")

        # Process results
        detections = []
        result = results[0]
        
        # Draw bounding boxes on image
        annotated_img = result.plot()

        # Extract detection information
        if len(result.boxes) > 0:
            print(f"Found {len(result.boxes)} detections")
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence * 100, 1)
                })
                print(f"  - {class_name}: {confidence*100:.1f}%")
        else:
            print("No detections found")

        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        img_data_url = f"data:image/jpeg;base64,{img_base64}"

        print("Response prepared successfully")
        print("=" * 50)

        return jsonify({
            'detections': detections,
            'image': img_data_url,
            'inference_time': inference_time
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Check if Gemini is configured
        if gemini_model is None:
            return jsonify({
                'response': 'Chat is currently unavailable. Please check the Gemini API configuration.'
            }), 503

        data = request.json
        message = data.get('message', '')
        detections = data.get('detections', [])

        if not message:
            return jsonify({'response': 'Please provide a message.'}), 400

        # Build context for Gemini
        context = """You are a medical assistant specialized in chest X-ray analysis. 
You help explain findings from a YOLOv8 model trained on the VinBigData dataset.

Current Analysis Results:
"""
        if detections:
            context += f"Detected {len(detections)} finding(s):\n"
            for det in detections:
                context += f"- {det['class']}: {det['confidence']}% confidence\n"
        else:
            context += "No detections in current image.\n"

        context += """
Available detection classes: Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, 
Consolidation, Edema, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Pleural thickening, 
Pneumothorax, No finding.

Guidelines:
- Keep responses concise and clear (2-3 sentences max)
- Be professional but friendly
- If asked about specific medical conditions, explain them simply
- Always remind users that AI results should be confirmed by healthcare professionals
- Don't make diagnoses, only explain what the findings typically indicate

User question: """

        print(f"Sending to Gemini: {message}")
        
        # Call Gemini with error handling
        response = gemini_model.generate_content(context + message)
        
        # Check if response was blocked
        if not response.text:
            return jsonify({
                'response': 'The response was blocked by safety filters. Please rephrase your question.'
            })
        
        print(f"Gemini response received: {response.text[:100]}...")
        return jsonify({'response': response.text})

    except Exception as e:
        error_msg = str(e)
        print(f"Chat error: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # Provide more specific error messages
        if "API_KEY_INVALID" in error_msg:
            return jsonify({'response': 'API key is invalid. Please check your Gemini API key.'}), 401
        elif "quota" in error_msg.lower():
            return jsonify({'response': 'API quota exceeded. Please try again later.'}), 429
        else:
            return jsonify({'response': f'Error: {error_msg}'}), 500


if __name__ == '__main__':
    print("Starting Insight-Ray Server...")
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Gemini AI: {'✓ Enabled' if gemini_model else '✗ Disabled'}")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)