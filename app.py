# app.py (Flask backend API)
from flask import Flask, request, jsonify
import nltk
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import json
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)


# Load model and other necessary files
model = load_model('chatbot_model.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json', 'r'))
lemmatizer = WordNetLemmatizer()

def preprocess_input(message):
    # Tokenize and lemmatize the user's message
    user_input = nltk.word_tokenize(message)
    user_input = [lemmatizer.lemmatize(word.lower()) for word in user_input]
    bag = [1 if word in user_input else 0 for word in words]
    return np.array([bag])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Get the message sent from React
    message = data['message']  # Extract the message

    # Process the message
    bag = preprocess_input(message)

    # Predict response
    prediction = model.predict(bag)
    tag = classes[np.argmax(prediction)]

    # Find the matching intent and select a response
    response = ""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            break

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
