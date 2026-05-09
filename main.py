import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

word_index = imdb.get_word_index()

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=500),
    SimpleRNN(128),
    Dense(1, activation="sigmoid")
])

model.build(input_shape=(None, 500))
model.load_weights("simple_rnn_imdb.weights.h5")

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500,
        padding="pre",
        truncating="pre"
    )
    return padded_review

st.title("IMDB Movie Review Analysis")
st.write("Enter a movie review to classify it as Positive or Negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a movie review.")
    else:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input, verbose=0)

        score = float(prediction[0][0])
        sentiment = "Positive" if score > 0.5 else "Negative"

        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")