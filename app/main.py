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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the saved model and vectorizer using joblib
model_path = os.path.join('model', 'sentiment_model_logistic_regression_NEW.pkl')
vectorizer_path = os.path.join('model', 'advanced_vectorizer_NEW.pkl')

# Global variables for model and vectorizer
model = None
vectorizer = None

# Download necessary NLTK resources at startup
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
        logger.error(traceback.format_exc())
        return False

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
            logger.info(f"Model type: {type(model)}")
            logger.info(f"Vectorizer type: {type(vectorizer)}")
            if hasattr(model, 'classes_'):
                logger.info(f"Model classes: {model.classes_}")
            # Try a sample prediction for debugging
            try:
                sample_text = "This is a test."
                processed = comprehensive_preprocess_text(sample_text)
                logger.info(f"Sample processed text: {processed}")
                sample_vec = vectorizer.transform([processed])
                logger.info(f"Sample vectorized shape: {sample_vec.shape}")
                pred = model.predict(sample_vec)
                logger.info(f"Sample prediction: {pred}")
            except Exception as pred_e:
                logger.error(f"Error during sample prediction: {str(pred_e)}")
                logger.error(traceback.format_exc())
            return True
        else:
            available_files = os.listdir(os.path.dirname(model_path)) if os.path.exists(os.path.dirname(model_path)) else []
            logger.error(f"Model or vectorizer file not found. Available files in directory: {available_files}")
            return False
    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Comprehensive text preprocessing function that matches the training preprocessing pipeline
def comprehensive_preprocess_text(text):
    """Complete text preprocessing pipeline following lecture guidelines"""
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
    except:
        tokens = text.split()
    
    # 4. Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # 5. Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    except:
        # Fallback if NLTK resources not available
        tokens = tokens
    
    # 6. Apply stemming
    try:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    except:
        # Fallback if stemming fails
        pass
    
    # Join back to string
    return ' '.join(tokens)

# Fallback simple preprocessing for compatibility
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabet characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text

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
                'sentiment': 'neutral',
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'description': 'Model or vectorizer not available. Using default response.',
                'error': 'Model or vectorizer not available'
            }
    
    try:
        # Use the comprehensive preprocessing to match training
        processed_text = comprehensive_preprocess_text(text)
        logger.info(f"Processed text: {processed_text}")
        
        # Vectorize the processed text
        text_vectorized = vectorizer.transform([processed_text])
        prediction = model.predict(text_vectorized)[0]
        proba = model.predict_proba(text_vectorized)[0]
        
        # Prepare the sentiment scores for each class
        sentiment_scores = {str(label): float(prob) for label, prob in zip(model.classes_, proba)}
        
        # Get descriptive interpretation
        sentiment_description = get_sentiment_description(str(prediction), sentiment_scores)
        
        logger.info(f"Prediction: {prediction}, Scores: {sentiment_scores}")
        logger.info(f"Description: {sentiment_description}")
        
        return {
            'text': text,
            'sentiment': str(prediction),
            'scores': sentiment_scores,
            'description': sentiment_description
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'text': text,
            'sentiment': 'neutral',
            'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
            'description': 'An error occurred during sentiment analysis. Using default response.',
            'error': str(e)
        }

# API to analyze a single text
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")  # Log incoming data to help debug
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'description': 'Please provide a text field in your JSON request body.'
            }), 400
            
        text = data['text']
        result = predict_sentiment(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'description': 'An unexpected error occurred while processing your request.'
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
    logger.error(traceback.format_exc())
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

# Download NLTK resources and load models at startup
download_nltk_resources()
load_models()

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 5000)))
    # Load models at startup
    if not load_models():
        logger.warning("Starting application without models loaded. Health checks will fail.")
    
    app.run(debug=False, host='0.0.0.0', port=port)