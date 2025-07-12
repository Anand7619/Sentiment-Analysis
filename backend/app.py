
from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from flask_cors import CORS
import os
app = Flask(__name__)
CORS(app) 

MODEL_PATH = "./model"
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

def load_model():
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Text not provided'}), 400
    prediction = classifier(data['text'])[0]
    return jsonify({
        "label": prediction['label'].lower(),
        "score": float(prediction['score'])
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
