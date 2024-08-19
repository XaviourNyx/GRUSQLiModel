import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load model locally
new_model = tf.keras.models.load_model('testModel.keras')



tokenizer = joblib.load('tokenizer.pkl')


label_encoder = joblib.load('label_encoder.pkl')

max_length = joblib.load('max_length.pkl')


def preprocess_input(user_input):
    user_input_encoded = tokenizer.texts_to_sequences([user_input])
    user_input_padded = tf.keras.preprocessing.sequence.pad_sequences(user_input_encoded, maxlen=max_length)
    return user_input_padded



st.title("SQL Injection Input Classifier")
st.write("Enter a sentence to classify whether it is a SQL injection input or not.")

# User input
user_input = st.text_input("Input Sentence:")

if st.button("Classify"):
    if user_input.strip():  # Check if user input is not empty
        # Tokenize the user input
        user_input_encoded = tokenizer.texts_to_sequences([user_input])
        user_input_padded = tf.keras.preprocessing.sequence.pad_sequences(user_input_encoded, maxlen=max_length)

        # Make a prediction
        prediction = new_model.predict(user_input_padded)
        probability = prediction[0][0]

        if probability > 0.5:
            st.write(f"The input sentence is classified as positive with a probability of {probability:.4f}.")
        else:
            st.write(f"The input sentence is classified as negative with a probability of {1 - probability:.4f}.")
    else:
        st.write("Please enter a sentence to classify.")
