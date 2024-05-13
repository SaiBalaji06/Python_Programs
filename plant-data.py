# app.py
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
# Update the import statement to import MobileNetV2 specifically
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


app = Flask(__name__)

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

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

    if file:
        image = file.read()
        result = detect_disease(image)
        return render_template('result.html', result=result)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/camera/live')
def live():
    return render_template('live.html')

def detect_disease(image):
    # Preprocess the image
    img = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    label = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]

    return f"Label: {label} ({confidence:.2f})"

if __name__ == '__main__':
    app.run(debug=True)
