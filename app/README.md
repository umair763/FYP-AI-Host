# Sentiment Analysis API

This API provides sentiment analysis capabilities using a pre-trained machine learning model.

## Issue Resolution: 'Neutral' Predictions

If you're experiencing issues with the API always returning "neutral" predictions, follow these steps to diagnose and fix the problem:

## Testing the Model

1. Run the test script to verify the model and vectorizer are working correctly:

```bash
cd app
python test_model.py
```

This script will:

-  Test model and vectorizer loading
-  Verify preprocessing functions
-  Test predictions against sample texts with known expected outputs
-  Print detailed logs about what's happening at each step

## Common Issues and Solutions

### 1. Model Files Missing or Corrupted

**Symptoms:**

-  Error logs indicating model files can't be found
-  Health check endpoint shows "unhealthy" status

**Solution:**

-  Ensure the model files exist in the correct location:
   -  Check that `model/sentiment_model_logistic_regression_NEW.pkl` and `model/advanced_vectorizer_NEW.pkl` exist
   -  If not, copy them from your training environment to the `app/model/` directory

### 2. NLTK Resources Not Available

**Symptoms:**

-  Error logs mentioning NLTK resources missing
-  Preprocessing steps failing

**Solution:**

-  Run `python -m nltk.downloader stopwords punkt wordnet omw-1.4`
-  Ensure your deployment environment has internet access for the initial download

### 3. Preprocessing Mismatch

**Symptoms:**

-  Model always predicts "neutral" despite varied input
-  Processed text looks different than expected

**Solution:**

-  Check log files to see how the text is being processed
-  Ensure the preprocessing function matches exactly what was used during training
-  The enhanced `comprehensive_preprocess_text` function should help resolve this

### 4. Vectorizer Vocabulary Mismatch

**Symptoms:**

-  Few or no non-zero elements in vectorized text
-  Poor prediction performance

**Solution:**

-  Verify the vectorizer vocabulary size matches what was used in training
-  Check that the processed text contains terms from the vocabulary

## Monitoring and Logging

Enhanced logging has been implemented to help diagnose issues:

-  Check the `app/logs/sentiment_api.log` file for detailed information
-  Use the `/health` endpoint to verify model and vectorizer status
-  Sample model predictions are run at startup to verify correct operation

## Testing with Sample Sentences

Send requests to the API with a range of sentiments to verify it works:

```bash
curl -X POST "http://localhost:5000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is absolutely amazing! I love it!"}'

curl -X POST "http://localhost:5000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is terrible. I hate it."}'

curl -X POST "http://localhost:5000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "This is neither good nor bad."}'
```

## Additional Information

If you continue to experience issues, check:

-  That the model was trained correctly with balanced data
-  That the model's accuracy is reasonable (current reported accuracy is 0.6476)
-  That the deploy environment has sufficient resources

The updated code implements enhanced preprocessing, more comprehensive logging, and validation of model and vectorizer objects to help identify and resolve issues.
