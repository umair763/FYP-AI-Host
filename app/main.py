from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('app/model/sentiment_model_logistic_regression.pkl')
vectorizer = joblib.load('app/model/advanced_vectorizer.pkl')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    result = {
        'sentiment': prediction,
        'scores': {
            'negative': float(probs[0]),
            'neutral': float(probs[1]),
            'positive': float(probs[2])
        }
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
