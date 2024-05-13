import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
model = load_model('final_plant_disease_model.h5')

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        img_path = os.path.join('uploads', filename)
        return detect_disease(img_path)
    else:
        return redirect(request.url)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/camera/live')
def camera_live():
    return render_template('camera_live.html')

@app.route('/camera/detect', methods=['POST'])
def camera_detect():
    # Capture a frame from the camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('uploads/camera.jpg', frame)
        img_path = 'uploads/camera.jpg'
        return detect_disease(img_path)
    else:
        return 'Camera not available'

def detect_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    preds = model.predict(img)
    class_indices = {'class1': 'Disease1', 'class2': 'Disease2', 'class3': 'Disease3'}  # Replace with your class labels
    
    prediction = class_indices[np.argmax(preds)]
    confidence = np.max(preds)
    confidence_percent = "{:.2%}".format(confidence)
    
    return render_template('result.html', result=prediction, confidence_percent=confidence_percent)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
