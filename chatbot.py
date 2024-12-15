#%% run_model.py
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model
import random

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Load words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
model = load_model('chatbot_model.keras')

# Test inference
while True:
    message = input("You: ")
    if message.lower() == "quit":
        break

    # Preprocess user input
    user_input = nltk.word_tokenize(message)
    user_input = [lemmatizer.lemmatize(word.lower()) for word in user_input]
    bag = [1 if word in user_input else 0 for word in words]
    bag = np.array([bag])

    # Predict response
    prediction = model.predict(bag)
    tag = classes[np.argmax(prediction)]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            print(f"Bot: {random.choice(intent['responses'])}")
