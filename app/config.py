import os

# Base directories
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

# Model settings
MODEL_DIR = os.path.join(APP_DIR, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'sentiment_model.pkl')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'vectorizer.pkl')
ENCODER_FILE = os.path.join(MODEL_DIR, 'label_encoder.pkl')

# XGBoost specific settings
CONFIDENCE_THRESHOLD = 0.6  # Threshold for prediction confidence

# Logging settings
LOG_DIR = os.path.join(APP_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'sentiment_api.log')
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# API settings
DEBUG_MODE = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
PORT = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 8080)))
HOST = '0.0.0.0'

# Feature settings
ENABLE_DETAILED_LOGGING = True
DEFAULT_SENTIMENT = 'neutral'
DEFAULT_SCORES = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} 