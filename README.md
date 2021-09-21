## **Music Mood Classification with Audio Features**

### **Problem Statement**
Music is an important aspect that is woven into our daily lives. Even in our most memorable and most important life events, music is right there with us. Music has the power to enhance or dictate emotions and mood and vice versa. Arguably, sometimes when you are sad there is nothing better than listening to a sad song that is able to enhance those emotions. 

With this knowledge, music and mood can be used to create music or playlist recommendations based on the listener's mood in a way that can truly reach the listener. 

The goal of this project is to create a model to predict the mood of a playlist or track based on the audio features of a track or tracks within a playlist.

### **Data Collection**
Spotify is a music streaming and sharing application that allows users to stream music, create playlists, and follow artists, users, and other playlists. Spotify also incorporates machine learning and recommender systems to create playlists for users based off of their recent listening history. I decided to use Spotify's API and Spotify API Wrapper Spotipy to get audio features of a single track as well as the audio features of all tracks within a playlist by accessing the playlist ID. Spotify has a category 'Mood' where I was able to access 50 playlists within this category and labeled "happy", "sad", "energetic", "calm" based off of the names and descriptions of the playlists. Once I labeled these tracks in each of these playlists, I obtained their audio features. I created a dataframe that had the name of the track, artist, track id, audio features and labeled mood. I also utilized a dataset from Kaggle that had approximately 600 observations of tracks and audio features with labeled moods. 

### **Data Modeling**
I utilized several unsupervised models such as KMeans and Principal Component Analysis (PCA) in order to reduce feature dimensionality. I then utilized multiclassification models in order to predict the mood. The best model was a Pipeline of transformers: PolynomialFeatures, MinMaxScaler, PCA, and classified with LogisticRegressionCV. My final model had a training fit of 86% accuracy and 87% accuracy

### **Deployment of Model using Streamlit**
I deployed the model using Streamlit where a user is able to enter the id of a public playlist and the app returns a mood profile.


### **Conclusion & Future Work**
The models performed relatively well based off of the amount of data that I was able to gather within this timeframe. I did run into timer issues when collecting data which limited my scope. However, getting labels for mood poses an issue in of itself. Mostly because emotions and mood is subjective as well as labeling playlists as a certain time. I would like to continue this project by gathering more data and developing a better


#### **Resources**
https://towardsdatascience.com/how-to-get-data-from-apis-with-python-dfb83fdc5b5b

https://www.youtube.com/watch?v=xdq6Gz33khQ&t=2512s

https://medium.com/@maxtingle/getting-started-with-spotifys-api-spotipy-197c3dc6353b