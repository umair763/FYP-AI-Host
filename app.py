from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app, origins=["https://www.socialsight.me"])

# Load model and vectorizer
MODEL_PATH = os.path.join("model", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize and predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    return jsonify({'prediction': str(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
