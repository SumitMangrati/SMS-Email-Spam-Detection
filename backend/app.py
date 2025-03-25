import pickle
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows communication with the extension

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to clean and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS for API access

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "SMS Spam Detector API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message = data.get("message")

        if not message:
            return jsonify({"error": "No message provided"}), 400

        transformed_message = transform_text(message)
        transformed_message = [transformed_message]
        vectorized_message = tfidf.transform(transformed_message)
        prediction = model.predict(vectorized_message)

        result = "Spam" if prediction == 1 else "Not Spam"
        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
