import os

# Base directories
APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

# Model settings
MODEL_DIR = os.path.join(APP_DIR, 'model')
MODEL_FILE = os.path.join(MODEL_DIR, 'sentiment_model_logistic_regression_NEW.pkl')
VECTORIZER_FILE = os.path.join(MODEL_DIR, 'advanced_vectorizer_NEW.pkl')

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
PORT = int(os.environ.get('PORT', os.environ.get('WEBSITES_PORT', 5000)))
HOST = '0.0.0.0'

# Feature settings
ENABLE_DETAILED_LOGGING = True
DEFAULT_SENTIMENT = 'neutral'
DEFAULT_SCORES = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} 