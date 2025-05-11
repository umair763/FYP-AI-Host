# Advanced Sentiment Analysis API with XGBoost

This API provides advanced sentiment analysis capabilities using a pre-trained XGBoost machine learning model.

## Overview

The API analyzes text and classifies it as positive, negative, or neutral, providing confidence scores for each category. The model has been trained on a balanced dataset and optimized for accuracy.

## API Endpoints

### Analyze Text

-  **URL**: `/analyze`
-  **Method**: `POST`
-  **Body**: JSON with a `text` field
-  **Response**: Sentiment analysis results including classification, confidence scores, and description

### Example Request

```bash
curl -X POST "https://ai-sentiment-app-o5i3f.ondigitalocean.app/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product! It works amazingly well."}'
```

### Example Response

```json
{
	"text": "I love this product! It works amazingly well.",
	"sentiment": "positive",
	"scores": {
		"positive": 0.92,
		"neutral": 0.05,
		"negative": 0.03
	},
	"confidence": 0.92,
	"description": "The text expresses a positive sentiment with very high confidence. This indicates approval, satisfaction, or optimism."
}
```

## Health and Diagnostics

-  **Health Check**: `/health` - Returns the status of the API and model components
-  **Diagnostics**: `/diagnostic` - Returns detailed information about the environment and model setup

## Model Information

This API uses a trained XGBoost classifier with the following components:

1. **Text Preprocessing**:

   -  Lowercasing
   -  Removal of URLs, HTML tags, and special characters
   -  Handling of punctuation with apostrophe preservation
   -  Stemming and stopword removal

2. **Feature Engineering**:

   -  TF-IDF vectorization with n-grams (1-3)
   -  Maximum of 15,000 features
   -  Stop words filtering

3. **Model**:
   -  XGBoost classifier trained on balanced data
   -  Confidence threshold of 0.6 for predictions

## Deploying the Model

To deploy this API with the XGBoost model:

1. Place the model files in the `app/model` directory:

   -  `sentiment_model.pkl` - The XGBoost model
   -  `vectorizer.pkl` - The TF-IDF vectorizer
   -  `label_encoder.pkl` - The label encoder

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the API:
   ```bash
   python app/main.py
   ```

## Troubleshooting

If you encounter issues with model loading or API performance:

1. Check the `/diagnostic` endpoint for detailed information about model loading and configuration
2. Ensure all three model files are correctly placed in the `app/model` directory
3. Verify NLTK resources are downloaded (stopwords and punkt)
4. Check that the confidence threshold is appropriate for your use case (default: 0.6)

The API includes a fallback mechanism that will create a simple model if the trained models cannot be loaded.
