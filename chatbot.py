import json
import random
import spacy
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import streamlit as st
from collections import defaultdict

# Load the dataset
with open('intents.json', 'r') as file:
    dataset = json.load(file)

# Extract intents and patterns
intents = dataset['intents']
patterns_dict = {}
responses_dict = {}
context_dict = {}

for intent in intents:
    patterns_dict[intent['tag']] = intent['patterns']
    responses_dict[intent['tag']] = intent['responses']
    if 'context' in intent:
        context_dict[intent['tag']] = intent['context']

# Combine patterns and intents for training data
train_data = []
for intent, patterns in patterns_dict.items():
    for pattern in patterns:
        train_data.append((pattern, intent))

# Shuffle the training data
random.shuffle(train_data)

# Split train and test data
train_texts, test_texts, train_labels, test_labels = train_test_split(
    [data[0] for data in train_data], [data[1] for data in train_data], test_size=0.2, random_state=42
)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

# Fit and transform on train data, transform test data
train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()
test_tfidf = tfidf_vectorizer.transform(test_texts).toarray()

# Initialize SVM classifier
svm_classifier = LinearSVC()

# Train SVM classifier
svm_classifier.fit(train_tfidf, train_labels)

# Predict on test data
predictions = svm_classifier.predict(test_tfidf)

# Calculate evaluation metrics
intent_accuracy = accuracy_score(test_labels, predictions)
intent_precision = precision_score(test_labels, predictions, average='weighted')
intent_recall = recall_score(test_labels, predictions, average='weighted')
intent_f1 = f1_score(test_labels, predictions, average='weighted')

# Initialize spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Define a context tracker
class ContextTracker:
    def __init__(self):
        self.context = defaultdict(dict)

    def update_context(self, intent, entities):
        self.context[intent] = entities

    def get_context(self, intent):
        return self.context[intent]

# Load context tracker
context_tracker = ContextTracker()

def classify_intent(user_input):
    # Use simple rule-based approach to tokenize input
    tokens = user_input.lower().split()
    # Convert tokens to TF-IDF vector
    input_tfidf = tfidf_vectorizer.transform([' '.join(tokens)]).toarray()
    # Predict intent using SVM classifier
    intent = svm_classifier.predict(input_tfidf)[0]
    return intent

def extract_entities(user_input):
    # Process user input using spaCy
    doc = nlp(user_input)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = ent.text
    return entities

def get_response(intent, entities):
    if intent in responses_dict:
        response = random.choice(responses_dict[intent])
        if intent in context_dict:
            next_intent = context_dict[intent]['next_intent']
            response += " " + context_dict[intent]['response']
            if next_intent == "accommodation_preferences":
                lodging_type = entities.get('lodging_type', '')
                response = response.replace("[lodging_type]", lodging_type)
            elif next_intent == "activity_preferences":
                activity_type = entities.get('activity_type', '')
                response = response.replace("[activity_type]", activity_type)
            elif next_intent == "restaurant_recommendation":
                cuisine_type = entities.get('cuisine_type', '')
                response = response.replace("[cuisine_type]", cuisine_type)
            # Update context for the next intent
            context_tracker.update_context(next_intent, entities)
        return response
    else:
        return "I'm sorry, I didn't understand that."

def handle_user_input(user_input):
    intent = classify_intent(user_input)
    entities = extract_entities(user_input)
    # Merge entities with context
    entities.update(context_tracker.get_context(intent))
    response = get_response(intent, entities)
    return intent, response

# Main heading with emojis
st.title("🌍 TravelPal - Your Personal Travel Assistant 🧳")

# Chat interface
chat_history = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("👤 You: ", "")

if st.button("Send"):
    user_response = "👤 You: " + user_input
    st.session_state.chat_history.append(user_response)
    intent, response = handle_user_input(user_input)
    st.session_state.chat_history.append("🤖 Bot: " + response)
    st.write(f"🔍 Intent: {intent}")

# Display chat history
for message in st.session_state.chat_history:
    st.text(message)

# Display evaluation metrics
st.write("### Intent Recognition Metrics (TF-IDF + SVM):")
st.write(f"Accuracy: {intent_accuracy:.2f}")
st.write(f"Precision: {intent_precision:.2f}")
st.write(f"Recall: {intent_recall:.2f}")
st.write(f"F1 Score: {intent_f1:.2f}")
