import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# To get word indexes
word_index = imdb.get_word_index()
rev_word_index = {value:key for key,value in word_index.items()}

# defining this is necessary as I have used a custom activation function for the output layer.
def custom_activation(x):
    return (tf.nn.tanh(x) + 1) / 2

model=load_model('simple_rnn_model.keras',custom_objects={'custom_activation':custom_activation})

# To compile the model after loading it.
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# To process the input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 0), 9999) for word in words]  # 0 for OOV words
    paded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return paded_review

# Prediction
def predict_sentiment(review):
    inp = preprocess_text(review)
    prediction = model.predict(inp)
    sentiment = 'Positive' if prediction[0][0]>=0.5 else 'Negative'
    return sentiment, prediction[0][0]

# doing the app stuff

st.title("IMDB Movie Sentiment Analysys")
st.write('Enter a movie detailed movie review in between 100 to 500 words to see weather is it positive or negative.')

#User input
user_input = st.text_area("Movie review")

if st.button('Classify'):
    length = len(user_input.split())
    if length>=100 and length<=500:
        sentiment,score = predict_sentiment(user_input)
        st.write(f'Sentiment:{sentiment}')
        st.write(f'Prediction Score:{score}')
    else:
        st.write('Please use 100 to 500 words')
        st.write(f'Current review length:{length}')