from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the saved model and vectorizer using joblib
model_path = os.path.join('model', 'sentiment_model_logistic_regression.pkl')
vectorizer_path = os.path.join('model', 'advanced_vectorizer.pkl')

# Global variables for model and vectorizer
model = None
vectorizer = None

# Load model and vectorizer
def load_models():
    global model, vectorizer
    try:
        logger.info(f"Attempting to load model from: {os.path.abspath(model_path)}")
        logger.info(f"Attempting to load vectorizer from: {os.path.abspath(vectorizer_path)}")
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            logger.info("Model and vectorizer loaded successfully.")
            return True
        else:
            available_files = os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else []
            logger.error(f"Model or vectorizer file not found. Available files in directory: {available_files}")
            return False
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        return False

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
    global model, vectorizer
    
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        if not load_models():
            logger.error("Failed to load models, returning default response")
            return {
                'text': text,
                'sentiment': 'neutral',
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'error': 'Model or vectorizer not available'
            }
    
    try:
        processed_text = preprocess_text(text)
        logger.info(f"Processed text: {processed_text}")
        
        # Vectorize the processed text
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        proba = model.predict_proba(text_vectorized)[0]
        
        # Prepare the sentiment scores for each class
        sentiment_scores = {str(label): float(prob) for label, prob in zip(model.classes_, proba)}
        
        logger.info(f"Prediction: {prediction}, Scores: {sentiment_scores}")
        
        return {
            'text': text,
            'sentiment': str(prediction),
            'scores': sentiment_scores
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
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
        logger.info(f"Received data: {data}")  # Log incoming data to help debug
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        text = data['text']
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    # Check if models are loaded
    model_status = "loaded" if model is not None else "not loaded"
    vectorizer_status = "loaded" if vectorizer is not None else "not loaded"
    
    return jsonify({
        'status': 'healthy' if model is not None and vectorizer is not None else 'unhealthy',
        'message': 'Service is running',
        'model_status': model_status,
        'vectorizer_status': vectorizer_status
    })

# Root endpoint
@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    return jsonify({'message': 'Welcome to the Sentiment Analysis API'})

# Error handler
@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': str(e)}), 500

# Ensure CORS headers after request
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Load models at startup
load_models()

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 5000)))
    # Load models at startup
    if not load_models():
        logger.warning("Starting application without models loaded. Health checks will fail.")
    
    app.run(debug=False, host='0.0.0.0', port=port)