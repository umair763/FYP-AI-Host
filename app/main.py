from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer once
model = joblib.load(os.path.join("model", "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join("model", "vectorizer.pkl"))

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    comment = data.get("comment", "").strip()
    if not comment:
        return jsonify({"error": "No comment provided"}), 400
    
    try:
        vectorized = vectorizer.transform([comment])
        prediction = model.predict(vectorized)[0]
        return jsonify({"sentiment": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
