import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data from JSON file
data_file = open('data.json').read()
intents = json.loads(data_file)

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Process intents and patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents to the corpus
        documents.append((w, intent['tag']))

        # Add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize, lowercase, and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Print information about the data
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes using pickle
with open('texts.pkl', 'wb') as file:
    pickle.dump(words, file)

with open('labels.pkl', 'wb') as file:
    pickle.dump(classes, file)

# Create training data with fixed-size bag of words
training = []
output_empty = [0] * len(classes)

# Find the maximum length of the bag of words
max_len = max(len(pattern) for pattern, _ in documents)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Limit the bag of words to the maximum length
    pattern_words = pattern_words[:max_len]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle the training data
random.shuffle(training)

# Separate into train_x and train_y
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print("Training data created")

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Print model summary
model.summary()

# Train and save the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)

print("Model created")
