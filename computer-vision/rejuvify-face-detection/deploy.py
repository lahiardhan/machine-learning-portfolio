from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from PIL import Image

app = Flask(__name__)

# Predicting Images
model = load_model('../model 1/model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
labels = ["acne", "blackhead", "wrinkles", "enlarged-pores", "redness", "dark-spot"]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    image = Image.open(file)

    # Ensuring appropriate image depth
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    images = np.vstack([image])

    predictions = model.predict(images)
    predicted_labels = []
    for i, pred in enumerate(predictions[0]):
        predicted_labels.append({'label': labels[i], 'accuracy': float(pred)})

    return jsonify({'predictions': predicted_labels})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)