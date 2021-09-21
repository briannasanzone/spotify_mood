import streamlit as st
import pandas as pd
import numpy as np
import pickle
import spotipy
import audio_analysis_function
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth


page = st.sidebar.selectbox(
'Select a page', ('Home', 'About', 'Make a Prediction')
)

if page == 'Home':
    st.title('Mood Maker')
    st.write('What is the mood of your playlist?')

if page == 'About':
    st.write('Hello, my name is Brianna Sanzone and I am a data scientist')
    st.write('Thank you for visiting my app, you can contact me at sanzonebrianna@gmail.com')


if page == 'Make a Prediction':
    st.title('Mood Maker')
    st.write('What is the mood of your playlist?')
    st.write('Enter your Spotify playlist id')
    st.write('Copy and paste your spotify playlist id')

    with open('final_mood_model.pkl', 'rb') as pickle_in:
        pipe=pickle.load(pickle_in)

    user_text = st.text_input('When you copy and paste it, get rid of https://open.spotify.com/playlist/ so it looks like: ',
    value='7od0I5IC3GfzXeGj3i6ugy?si=ed549e9989e246b7')

    id = audio_analysis_function.audio_analysis(user_text)

    preds = pipe.predict(id)
    preds_list = [x for x in preds]

    n=len(preds_list)
    sad = []
    chill = []
    energetic = []

    for pred in preds_list:
        if pred == 0:
            sad.append(pred)
    for pred in preds_list:
        if pred == 1:
            chill.append(pred)
    for pred in preds_list:
        if pred == 2:
            energetic.append(pred)

    n_sad = len(sad)
    n_chill = len(chill)
    n_energetic = len(energetic)
    st.write(f'This playlist is {np.round((n_sad/(n))*100)}% Sad, {np.round((n_chill/(n))*100)}% Chill, and {np.round((n_energetic/(n))*100)}% Energetic')
