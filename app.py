from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import requests
import os
import re
import json

app = Flask(__name__)
model = load_model('models/skin_disease_model.h5')

# Define dataset folder and extract class names
diseases_folder = 'dataset'
class_labels = sorted([
    name for name in os.listdir(diseases_folder)
    if os.path.isdir(os.path.join(diseases_folder, name))
])

# GEMINI API Configuration
GEMINI_API_KEY = "AIzaSyAzKAz_AJQvLOYgjplRxoF8s1Jc23uy5HU"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

def preprocess_image(image, target_size=224):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, [target_size, target_size])
    img = img / 255.0
    return img.numpy()

def format_markdown(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"^## (.*?)$", r"<h2 class='text-2xl font-bold mt-6 mb-2'>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^### (.*?)$", r"<h3 class='text-xl font-semibold mt-4 mb-2 text-blue-700'>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^- (.*?)$", r"<li class='ml-6 list-disc'>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"(<li.*?>.*?</li>)", r"<ul>\1</ul>", text, flags=re.DOTALL)
    text = re.sub(r"\n{2,}", "<br><br>", text)
    return text

def get_disease_info_from_gemini(disease_name):
    prompt = (
        f"Explain the skin disease '{disease_name}' in user-friendly terms. "
        f"Include:\n"
        f"1. Five key symptoms\n"
        f"2. Five common causes\n"
        f"3. Five effective medical treatments with brief explanations\n"
        f"4. Five dietary suggestions that can help\n"
        f"5. Five exercise or lifestyle tips suitable for managing the disease"
    )

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload)
        if response.status_code == 200:
            gemini_reply = response.json()
            return gemini_reply['candidates'][0]['content']['parts'][0]['text']
        else:
            return f"<p class='text-red-600'>Gemini API Error {response.status_code}: {response.text}</p>"
    except Exception as e:
        return f"<p class='text-red-600'>Error contacting Gemini API: {str(e)}</p>"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image'].read()
        processed_image = preprocess_image(image)
        predictions = model.predict(np.expand_dims(processed_image, axis=0))
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_labels[predicted_class_index]

        # Call Gemini API to get explanation
        explanation = get_disease_info_from_gemini(predicted_class_name)
        formatted_text = format_markdown(explanation)

        return render_template("result.html", disease=predicted_class_name, result_text=formatted_text)

    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('sih.html')

@app.route('/index')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
