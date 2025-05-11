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
import sys
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

# Define default config values
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'vectorizer.pkl')
ENCODER_FILE = os.path.join(MODEL_DIR, 'label_encoder.pkl')
CONFIDENCE_THRESHOLD = 0.6
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_SENTIMENT = 'neutral'
DEFAULT_SCORES = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Try to import custom config values
try:
    from .config import *
except ImportError:
    try:
        import config
        # Update variables from config if available
        if hasattr(config, 'MODEL_FILE'):
            MODEL_FILE = config.MODEL_FILE
        if hasattr(config, 'VECTORIZER_FILE'):
            VECTORIZER_FILE = config.VECTORIZER_FILE
        if hasattr(config, 'ENCODER_FILE'):
            ENCODER_FILE = config.ENCODER_FILE
        if hasattr(config, 'CONFIDENCE_THRESHOLD'):
            CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
        if hasattr(config, 'DEFAULT_SENTIMENT'):
            DEFAULT_SENTIMENT = config.DEFAULT_SENTIMENT
        if hasattr(config, 'DEFAULT_SCORES'):
            DEFAULT_SCORES = config.DEFAULT_SCORES
    except ImportError:
        # Using default values
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Using model file: {MODEL_FILE}")
logger.info(f"Using vectorizer file: {VECTORIZER_FILE}")
logger.info(f"Using encoder file: {ENCODER_FILE}")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables for model components
model = None
vectorizer = None
encoder = None

# Download necessary NLTK resources
def download_nltk_resources():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        logger.info("NLTK resources downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model components
def load_models():
    global model, vectorizer, encoder
    try:
        logger.info(f"Attempting to load model from: {os.path.abspath(MODEL_FILE)}")
        logger.info(f"Attempting to load vectorizer from: {os.path.abspath(VECTORIZER_FILE)}")
        logger.info(f"Attempting to load encoder from: {os.path.abspath(ENCODER_FILE)}")
        
        # Check if files exist
        model_exists = os.path.exists(MODEL_FILE)
        vectorizer_exists = os.path.exists(VECTORIZER_FILE)
        encoder_exists = os.path.exists(ENCODER_FILE)
        
        logger.info(f"Model file exists: {model_exists}")
        logger.info(f"Vectorizer file exists: {vectorizer_exists}")
        logger.info(f"Encoder file exists: {encoder_exists}")
        
        if model_exists and vectorizer_exists and encoder_exists:
            # Load the model, vectorizer and encoder
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
            encoder = joblib.load(ENCODER_FILE)
            
            logger.info("Model components loaded successfully.")
            if hasattr(model, 'classes_'):
                logger.info(f"Model classes: {model.classes_}")
            if hasattr(encoder, 'classes_'):
                logger.info(f"Encoder classes: {encoder.classes_}")
            if hasattr(vectorizer, 'vocabulary_'):
                logger.info(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
            return True
        else:
            # Try alternate paths for DigitalOcean App Platform
            alt_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
            alt_model_file = os.path.join(alt_model_dir, os.path.basename(MODEL_FILE))
            alt_vectorizer_file = os.path.join(alt_model_dir, os.path.basename(VECTORIZER_FILE))
            alt_encoder_file = os.path.join(alt_model_dir, os.path.basename(ENCODER_FILE))
            
            logger.info(f"Trying alternate path for model: {alt_model_file}")
            logger.info(f"Trying alternate path for vectorizer: {alt_vectorizer_file}")
            logger.info(f"Trying alternate path for encoder: {alt_encoder_file}")
            
            if (os.path.exists(alt_model_file) and 
                os.path.exists(alt_vectorizer_file) and 
                os.path.exists(alt_encoder_file)):
                
                model = joblib.load(alt_model_file)
                vectorizer = joblib.load(alt_vectorizer_file)
                encoder = joblib.load(alt_encoder_file)
                logger.info("Model components loaded from alternate path successfully.")
                return True
            
            logger.error("Model files not found in any of the checked locations.")
            return False
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Text preprocessing function matching the training code
def preprocess_text(text):
    """Process text using the same function as during model training"""
    try:
        # Ensure text is a string
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        
        # Remove HTML tags
        text = re.sub(r"<.*?>", '', text)
        
        # Remove punctuation except apostrophes
        punctuation_without_apostrophe = string.punctuation.replace("'", "")
        text = re.sub(f"[{re.escape(punctuation_without_apostrophe)}]", '', text)
        
        # Remove numbers
        text = re.sub(r"\d+", '', text)
        
        # Replace repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
        
        # Join tokens back into text
        return " ".join(tokens)
    except Exception as e:
        logger.error(f"Error in text preprocessing: {str(e)}")
        logger.error(traceback.format_exc())
        return text  # Return original text if processing fails

# Get descriptive interpretation of sentiment
def get_sentiment_description(sentiment, confidence):
    """Generate a descriptive interpretation of the sentiment analysis results"""
    try:
        confidence_level = ""
        if confidence >= 0.9:
            confidence_level = "very high confidence"
        elif confidence >= 0.75:
            confidence_level = "high confidence"
        elif confidence >= 0.6:
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

# Predict sentiment using XGBoost model
def predict_sentiment(text):
    global model, vectorizer, encoder
    
    # Check if model components are loaded
    if model is None or vectorizer is None or encoder is None:
        if not load_models():
            logger.error("Failed to load model components, returning default response")
            return {
                'text': text,
                'sentiment': DEFAULT_SENTIMENT,
                'scores': DEFAULT_SCORES,
                'description': 'Model components not available. Using default response.',
                'error': 'Model components not available'
            }
    
    try:
        # Use the preprocessing function that matches training
        processed_text = preprocess_text(text)
        logger.info(f"Processed text: {processed_text[:100]}...")
        
        # Check if processed text is empty
        if not processed_text.strip():
            processed_text = "empty text"  # Fallback to avoid empty input
        
        # Vectorize the processed text
        text_vectorized = vectorizer.transform([processed_text])
        
        # Get prediction probabilities from the model
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get the highest probability class index and confidence score
        predicted_class_index = np.argmax(probabilities)
        confidence = np.max(probabilities)
        
        # Apply confidence threshold
        if confidence < CONFIDENCE_THRESHOLD:
            # Default to neutral if confidence is below threshold
            neutral_index = np.where(encoder.classes_ == 'neutral')[0]
            if len(neutral_index) > 0:
                prediction = 'neutral'
            else:
                prediction = encoder.inverse_transform([predicted_class_index])[0]
        else:
            # Use predicted class if confidence is sufficient
            prediction = encoder.inverse_transform([predicted_class_index])[0]
        
        # Prepare the sentiment scores for each class
        sentiment_scores = {str(cls): float(prob) for cls, prob in zip(encoder.classes_, probabilities)}
        
        # Get descriptive interpretation
        sentiment_description = get_sentiment_description(prediction, confidence)
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
        
        return {
            'text': text,
            'sentiment': str(prediction),
            'scores': sentiment_scores,
            'confidence': float(confidence),
            'description': sentiment_description
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
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
    encoder_status = "loaded" if encoder is not None else "not loaded"
    nltk_status = "available" if 'punkt' in nltk.data.path else "not available"
    
    return jsonify({
        'status': 'healthy' if model is not None and vectorizer is not None and encoder is not None else 'unhealthy',
        'message': 'Service is running',
        'model_status': model_status,
        'vectorizer_status': vectorizer_status,
        'encoder_status': encoder_status,
        'nltk_resources': nltk_status,
        'version': 'v3.0 (XGBoost model)'
    })

# Root endpoint
@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'})
    return jsonify({
        'message': 'Advanced Sentiment Analysis API with XGBoost Model',
        'version': 'v3.0',
        'usage': {
            'endpoint': '/analyze',
            'method': 'POST',
            'body': {
                'text': 'Your text to analyze'
            },
            'example': 'curl -X POST "https://ai-sentiment-app-o5i3f.ondigitalocean.app/analyze" -H "Content-Type: application/json" -d \'{"text": "I love this product!"}\''
        }
    })

# Diagnostic endpoint
@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    """Returns detailed diagnostic information about the app environment"""
    diagnostics = {
        'environment': {
            'working_directory': os.getcwd(),
            'app_directory': os.path.dirname(os.path.abspath(__file__)),
            'python_version': sys.version,
            'environment_variables': {k: v for k, v in os.environ.items() 
                                     if k.lower() in ('port', 'websites_port', 'path', 'pythonpath', 'app_dir')},
        },
        'model_config': {
            'model_dir': os.path.abspath(MODEL_DIR),
            'model_file': os.path.abspath(MODEL_FILE),
            'vectorizer_file': os.path.abspath(VECTORIZER_FILE),
            'encoder_file': os.path.abspath(ENCODER_FILE),
            'model_dir_exists': os.path.exists(MODEL_DIR),
            'model_file_exists': os.path.exists(MODEL_FILE),
            'vectorizer_file_exists': os.path.exists(VECTORIZER_FILE),
            'encoder_file_exists': os.path.exists(ENCODER_FILE),
            'confidence_threshold': CONFIDENCE_THRESHOLD,
        },
        'model_status': {
            'model_loaded': model is not None,
            'vectorizer_loaded': vectorizer is not None,
            'encoder_loaded': encoder is not None,
            'model_type': str(type(model)) if model is not None else None,
            'vectorizer_type': str(type(vectorizer)) if vectorizer is not None else None,
            'encoder_type': str(type(encoder)) if encoder is not None else None,
            'model_classes': list(model.classes_) if model is not None and hasattr(model, 'classes_') else None,
            'encoder_classes': list(encoder.classes_) if encoder is not None and hasattr(encoder, 'classes_') else None,
        },
        'nltk_status': {
            'nltk_data_path': nltk.data.path,
            'stopwords_available': 'stopwords' in nltk.data.path,
            'punkt_available': 'punkt' in nltk.data.path,
        }
    }
    
    # Try to list files in model directory
    try:
        if os.path.exists(MODEL_DIR):
            diagnostics['model_config']['files_in_model_dir'] = os.listdir(MODEL_DIR)
        
        # Try alternate model directory
        alt_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
        if os.path.exists(alt_model_dir) and alt_model_dir != os.path.abspath(MODEL_DIR):
            diagnostics['model_config']['alt_model_dir'] = alt_model_dir
            diagnostics['model_config']['alt_model_dir_exists'] = True
            diagnostics['model_config']['files_in_alt_model_dir'] = os.listdir(alt_model_dir)
    except Exception as e:
        diagnostics['errors'] = {'directory_listing_error': str(e)}
    
    return jsonify(diagnostics)

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

# Function to create a fallback model if no model is found
def create_fallback_model():
    """Create a simple fallback model if no model files are found"""
    try:
        logger.info("Creating fallback model for basic functionality...")
        
        # Sample data for a very simple XGBoost model
        texts = [
            "I love this product! It's amazing.", 
            "This is great! I'm happy with my purchase.",
            "I'm satisfied with this product.",
            "This is okay, nothing special.",
            "I don't like this very much.",
            "This is terrible. I hate it.",
            "Worst product ever. Completely disappointed."
        ]
        labels = ['positive', 'positive', 'positive', 'neutral', 'neutral', 'negative', 'negative']
        
        # Preprocess the texts
        processed_texts = [preprocess_text(text) for text in texts]
        
        # Create a simple label encoder
        from sklearn.preprocessing import LabelEncoder
        fallback_encoder = LabelEncoder()
        encoded_labels = fallback_encoder.fit_transform(labels)
        
        # Create a simple vectorizer
        fallback_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X = fallback_vectorizer.fit_transform(processed_texts)
        
        # Train a simple XGBoost model
        from xgboost import XGBClassifier
        fallback_model = XGBClassifier(n_estimators=50, max_depth=3)
        fallback_model.fit(X, encoded_labels)
        
        # Save the fallback models
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(fallback_model, MODEL_FILE)
        joblib.dump(fallback_vectorizer, VECTORIZER_FILE)
        joblib.dump(fallback_encoder, ENCODER_FILE)
        
        logger.info("Fallback model created and saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error creating fallback model: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Run the app
if __name__ == '__main__':
    # Download NLTK resources at startup
    download_nltk_resources()
    
    # Log the current working directory to help with debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # List available directories and files for debugging
    try:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"App directory: {app_dir}")
        if os.path.exists(MODEL_DIR):
            logger.info(f"Model directory exists at: {os.path.abspath(MODEL_DIR)}")
            logger.info(f"Files in model directory: {os.listdir(MODEL_DIR)}")
        else:
            logger.warning(f"Model directory does not exist at: {os.path.abspath(MODEL_DIR)}")
            # Try to create it
            try:
                os.makedirs(MODEL_DIR, exist_ok=True)
                logger.info(f"Created model directory at: {os.path.abspath(MODEL_DIR)}")
            except Exception as e:
                logger.error(f"Failed to create model directory: {str(e)}")
    except Exception as e:
        logger.error(f"Error checking directories: {str(e)}")
    
    # Try to load the models
    if not load_models():
        logger.warning("Failed to load models. Attempting to create fallback model...")
        
        # Try to create a fallback model
        if create_fallback_model():
            # Try loading again
            if load_models():
                logger.info("Successfully created and loaded fallback model.")
            else:
                logger.error("Failed to load even the fallback model.")
        else:
            logger.error("Failed to create fallback model.")
            
        logger.warning("Starting application with potential model issues. Some features may not work correctly.")
    
    # Determine port
    port = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 8080)))
    logger.info(f"Starting Flask application on port {port}")
    
    # Start the app
    app.run(debug=False, host='0.0.0.0', port=port)