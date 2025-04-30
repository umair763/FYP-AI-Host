from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the saved model and vectorizer using joblib
model_path = os.path.join('model', 'sentiment_model_logistic_regression.pkl')
vectorizer_path = os.path.join('model', 'advanced_vectorizer.pkl')

# Ensure the model and vectorizer are loaded properly
model, vectorizer = None, None
try:
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("Model and vectorizer loaded successfully.")
    else:
        raise FileNotFoundError("Model or vectorizer file not found.")
except Exception as e:
    print(f"Error loading model or vectorizer: {str(e)}")

# Text preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

# Predict sentiment
def predict_sentiment(text):
    try:
        processed_text = preprocess_text(text)
        # Vectorize the processed text
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        proba = model.predict_proba(text_vectorized)[0]
        
        # Prepare the sentiment scores for each class
        sentiment_scores = {str(label): float(prob) for label, prob in zip(model.classes_, proba)}
        
        return {
            'text': text,
            'sentiment': str(prediction),
            'scores': sentiment_scores
        }
    except Exception as e:
        return {
            'text': text,
            'sentiment': 'neutral',
            'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            'error': str(e)
        }

# API to analyze a single text
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        print(f"Received data: {data}")  # Log incoming data to help debug
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        text = data['text']
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'healthy', 'message': 'Service is running'})

# Root endpoint
@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    return jsonify({'message': 'Welcome to the Sentiment Analysis API'})

# Error handler
@app.errorhandler(Exception)
def handle_error(e):
    return jsonify({'error': str(e)}), 500

# Ensure CORS headers after request
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 5000)))
    app.run(debug=False, host='0.0.0.0', port=port)
