# step 1 : import libraries and load the model 
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model


# Load the IMDB data set word index
word_index = imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# Load the pretrained model with RELU activation 
model=load_model('imdb_simpleRNN_relu.h5')

#step : helper function
#function to decode reviews

def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# function to prepare user input 

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(wod,2)+3 for word in words]

    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# step :3 prediction function 
def predict_sentiment(review):
    preprocesed_review=preprocess_text(review)
    prediction=model.predict(preprocessed_review)
    sentiment='positive' if prediction [0][0]>0.5 else 'nagetive'

    return sentiment,prediction[0][0]

import Streamlit as st
#streamlit app
st.title("IMDB Movie review sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

#user input
user_input=st.text_area("Movie review")
if st.button('classify'):
    preprocessed_input=preprocess_text(user_input)
    sentiment='positive' if prediction[0][0] > 0.5 else 'negative'

    # display the result

    st.write(f'Sentiment : {sentiment}')

    st.write(f'Prediction score : {prediction[0][0]}')

else :

    st.write('Please enter a movie review .')


