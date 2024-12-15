
#%%
# train_model.py
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Load intents file
with open('intents.json', 'r') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Preprocessing data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Associate patterns with their tags
        documents.append((word_list, intent['tag']))
        # Add the tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Output is a '0' for each class and '1' for the current tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# Shuffle and convert to array
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Define model
model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Implement EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(train_x, train_y, epochs=200, batch_size=8, verbose=1, callbacks=[early_stopping])

# Save model
model.save('chatbot_model.keras')

print("Model training complete and saved!")
