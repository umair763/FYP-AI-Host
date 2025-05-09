from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import os
import logging
import traceback
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import time

# Import config values or use defaults
try:
    from .config import *
except ImportError:
    try:
        import config
    except ImportError:
        # Default config values if config.py is not available
        MODEL_DIR = os.path.join('model')
        MODEL_FILE = os.path.join(MODEL_DIR, 'sentiment_model_logistic_regression_NEW.pkl')
        VECTORIZER_FILE = os.path.join(MODEL_DIR, 'advanced_vectorizer_NEW.pkl')
        LOG_LEVEL = 'INFO'
        LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        DEFAULT_SENTIMENT = 'neutral'
        DEFAULT_SCORES = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
        os.makedirs(MODEL_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables for model and vectorizer
model = None
vectorizer = None

# Download necessary NLTK resources
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        logger.info("NLTK resources downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        return False

# Load model and vectorizer
def load_models():
    global model, vectorizer
    try:
        logger.info(f"Attempting to load model from: {os.path.abspath(MODEL_FILE)}")
        logger.info(f"Attempting to load vectorizer from: {os.path.abspath(VECTORIZER_FILE)}")
        
        if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
            
            logger.info("Model and vectorizer loaded successfully.")
            logger.info(f"Model classes: {model.classes_}")
            logger.info(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
            return True
        else:
            logger.error("Model or vectorizer file not found.")
            return False
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        return False

# Comprehensive text preprocessing function 
def comprehensive_preprocess_text(text):
    """Complete text preprocessing pipeline matching training"""
    if not isinstance(text, str):
        return ""
    
    # 1. Normalization
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Case normalization
    text = text.lower().strip()
    
    # 3. Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    # 4. Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # 5. Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except Exception:
        pass
    
    # 6. Apply stemming
    try:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    except Exception:
        pass
    
    # Join back to string
    return ' '.join(tokens)

# Get descriptive interpretation of sentiment
def get_sentiment_description(sentiment, scores):
    """Generate a descriptive interpretation of the sentiment analysis results"""
    try:
        max_score = max(scores.values())
        confidence_level = ""
        if max_score >= 0.9:
            confidence_level = "very high confidence"
        elif max_score >= 0.75:
            confidence_level = "high confidence"
        elif max_score >= 0.6:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence"
        
        if sentiment == "positive":
            return f"The text expresses a positive sentiment with {confidence_level}. This indicates approval, satisfaction, or optimism."
        elif sentiment == "negative":
            return f"The text expresses a negative sentiment with {confidence_level}. This suggests dissatisfaction, criticism, or pessimism."
        else:  # neutral
            return f"The text expresses a neutral sentiment with {confidence_level}. This indicates an objective, balanced, or impartial tone."
    except Exception as e:
        logger.error(f"Error generating sentiment description: {str(e)}")
        return "Unable to generate detailed sentiment description."

# Predict sentiment
def predict_sentiment(text):
    global model, vectorizer
    
    # Check if model and vectorizer are loaded
    if model is None or vectorizer is None:
        if not load_models():
            logger.error("Failed to load models, returning default response")
            return {
                'text': text,
                'sentiment': DEFAULT_SENTIMENT,
                'scores': DEFAULT_SCORES,
                'description': 'Model or vectorizer not available. Using default response.',
                'error': 'Model or vectorizer not available'
            }
    
    try:
        # Use the comprehensive preprocessing to match training
        processed_text = comprehensive_preprocess_text(text)
        
        # Check if processed text is empty
        if not processed_text.strip():
            processed_text = "empty text"  # Fallback to avoid empty input
        
        # Vectorize the processed text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Get prediction and probabilities
        prediction = model.predict(text_vectorized)[0]
        proba = model.predict_proba(text_vectorized)[0]
        
        # Prepare the sentiment scores for each class
        sentiment_scores = {str(cls): float(prob) for cls, prob in zip(model.classes_, proba)}
        
        # Get descriptive interpretation
        sentiment_description = get_sentiment_description(str(prediction), sentiment_scores)
        
        logger.info(f"Prediction: {prediction}, Top score: {max(sentiment_scores.values()):.4f}")
        
        return {
            'text': text,
            'sentiment': str(prediction),
            'scores': sentiment_scores,
            'description': sentiment_description
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            'text': text,
            'sentiment': DEFAULT_SENTIMENT,
            'scores': DEFAULT_SCORES,
            'description': 'An error occurred during sentiment analysis. Using default response.',
            'error': str(e)
        }

# API to analyze a single text
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'description': 'Please provide a valid JSON request body.'
            }), 400
            
        if 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'description': 'Please provide a text field in your JSON request body.'
            }), 400
        
        # Extract text for analysis    
        text = data['text']
        
        # Validate text input
        if not isinstance(text, str):
            text = str(text)
            
        if not text.strip():
            return jsonify({
                'error': 'Empty text',
                'description': 'Please provide non-empty text for analysis.',
                'sentiment': DEFAULT_SENTIMENT,
                'scores': DEFAULT_SCORES
            }), 400
        
        # Perform sentiment prediction
        result = predict_sentiment(text)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'description': 'An unexpected error occurred while processing your request.',
            'sentiment': DEFAULT_SENTIMENT,
            'scores': DEFAULT_SCORES
        }), 500

# Health check endpoint
@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    
    # Check if models are loaded
    model_status = "loaded" if model is not None else "not loaded"
    vectorizer_status = "loaded" if vectorizer is not None else "not loaded"
    nltk_status = "available" if 'punkt' in nltk.data.path else "not available"
    
    return jsonify({
        'status': 'healthy' if model is not None and vectorizer is not None else 'unhealthy',
        'message': 'Service is running',
        'model_status': model_status,
        'vectorizer_status': vectorizer_status,
        'nltk_resources': nltk_status,
        'version': 'v2.0 (optimized model)'
    })

# Root endpoint
@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    return jsonify({
        'message': 'Welcome to the Advanced Sentiment Analysis API',
        'version': 'v2.0',
        'usage': {
            'endpoint': '/analyze',
            'method': 'POST',
            'body': {
                'text': 'Your text to analyze'
            },
            'example': 'curl -X POST "https://your-api-url/analyze" -H "Content-Type: application/json" -d \'{"text": "I love this product!"}\''
        }
    })

# Error handler
@app.errorhandler(Exception)
def handle_error(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({
        'error': str(e),
        'description': 'An unexpected server error occurred.'
    }), 500

# Ensure CORS headers after request
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Run the app
if __name__ == '__main__':
    # Download NLTK resources at startup
    download_nltk_resources()
    
    # Load models at startup
    if not load_models():
        logger.warning("Starting application without models loaded. Health checks will fail.")
    
    port = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 5000)))
    app.run(debug=False, host='0.0.0.0', port=port)