import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

word_index = imdb.get_word_index()

model = load_model("simple_rnn_imdb.h5", compile=False)

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

st.title("IMDB Movie Review Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)

        score = prediction[0][0]
        sentiment = "Positive" if score > 0.5 else "Negative"

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")
