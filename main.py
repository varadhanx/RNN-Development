# step 1 : import libraries and load the model 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB data set word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pretrained model with RELU activation 
model = load_model("imdb_simpleRNN_relu.h5")

# Helper function: decode encoded reviews
def decoded_review(encoded_review):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in encoded_review])

# Function to preprocess user input 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function 
def predict_sentiment(review):
    preprocessed_review = preprocess_text(review)
    prediction = model.predict(preprocessed_review)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment, prediction[0][0]

# Streamlit app
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as **positive** or **negative**")

# User input
user_input = st.text_area("✍️ Enter your review here:")

if st.button("Classify"):
    if user_input.strip():  # check input is not empty
        sentiment, score = predict_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Prediction score:** {score:.4f}")
    else:
        st.warning("⚠️ Please enter a movie review first.")
