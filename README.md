## **Music Mood Classification with Audio Features**

## **Problem Statement**

The goal of this project was to train a model to use audio features of a track on Spotify to accurately predict the mood of a track, classified as `energetic`, `happy`, `chill`, or `sad`, to ultimately evaluate a user's playlist to predict the mood profile of the playlist. 

## **Executive Summary**

#### **Background**

Music is intertwined into almost every facet of our daily lives. Even in our most memorable and important life events, music is right there with us. We listen to a certain song and it could immediately transport us back to a point in time, allowing us to relive memories, feelings, and emotions. Music can stimulate the mind and has the power to enhance or dictate our emotions and vice versa ([source](https://www.gilbertgalindo.com/importanceofmusic)). How many times have you listened to a sad song when you were already sad, but it just made that feeling or emotion so much more real, more palpable, and easier to process? Music therapy exists for a reason. 

This dialogue between music and humans holds a powerful potential for meaningful possibilities. Emotions and mood can be used to create song and playlist recommendations based on the listener's mood in a way that can truly reach the listener. In conjunction to this, knowing what songs or playlist someone has been recently listening to can give us insight into one's emotions and mood. These two use cases can be applied in various applications. Music apps like Spotify already utilizes a recommender system to create recommended playlists for a user based on a user's listening history and most frequently played songs, albums, and artists. In January 2021, Spotify was granted a patent for emotion-based music recommendations using speech recognitiion combined with a user's metadata including: previously played music, rating of songs, saved content, and their friend's music taste to recommend new music ([source](https://www.musictech.net/news/spotify-patent-detect-emotion-recommend-music/)). Music mood classification can determine the mood and mental state of a user which can be used in combination with wellness therapy techniques, content, and resources, a new embodiment of music therapy. 

The goal of this project is to create a model to predict the mood of a track based on the audio features which would be used to create a mood profile of a playlist based on the moods of the tracks within the playlist. 

#### **Metrics**
Accuracy was used to evaluate the multiclass classification model. In this project, accuracy was most important to optimize due to its application and association with mental health and deep approach to personalization. The ethical implications of this model and future uses should be strongly considered and held at a very high moral standard as it exposes the user to vulnerabilities regarding mental health. 

#### **Modeling**

Mood is subjective, therefore in order to classify the mood of a track based on its audio features, transfer learning with unsupervised and supervised multiclassification models were utilized to produce optimal accuracy scores. Prior to modeling, the features were standardized using MinMaxScaler and StandardScaler, and transformed with PolynomialFeatures. Unsupervised techniques were used to reduce feature dimensionality, including KMeans Clustering and Principal Component Analysis (PCA). Due to the response variable (mood) being multiclass, several multiclassification models were employed, including Multiclass Classification Neural Networks, Logistic Regression, AdaBoost Classifier, and XGB Boost Classifier.

#### **Findings**
The final model employed a Pipeline of transformers: PolynomialFeatures, MinMaxScaler, and PCA, and a final estimator for multiclassification: LogisticRegression. Best prediction of mood included feature engineering, using KMeans Clustering, found the data best clustered with a k of 2 which was interpreted as positive (high) or negative (down) energy of a track and was included in the final model. During the modeling process,the models were incorrectly predicting the mood label `happy` as `energetic` and I decided to combine these labels as `energetic` as I did not feel that this was jeopardizing the integrity of the data or results. The final model had an 86% accuracy score on the training data used to fit the model and an 87% accuracy at predicting the mood of a track on new and unseen tracks. 

### Data
#### **Data Acquisition**
Spotify is a music streaming and sharing application that allows users to stream music, create playlists, and follow artists, users, and other playlists. Spotify's API and Spotify API Wrapper Spotipy was used to scrape audio features of tracks within a playlist by accessing the playlist ID. The Spotify category 'Mood' was used to obtain 50 playlist IDs as well as their respective names and descriptions. Based on the name and description, each playlist was labeled as "happy", "sad", "energetic", "chill". After the playlists were labeled, I obtained the audio features of approximately 50 tracks within each of the 50 playlists and labeled mood according to its playlist mood. A dataset from Kaggle that had approximately 700 observations of tracks and audio features with labeled moods was utilized in combined with my 5000 observations to increase training data and optimization of modeling. 

#### **Data Dictionary**

|Feature Name|Type|Description|
|---|---|---|
|name|object|Name of the song|
|album|object|Name of the album of the track/song is in|
|artist|object|Name of the artist of the track/song|
|track_id|object|id of the track/song that is used on Spotify to identify track|
|instrumentalness|float|A measure from 0.0 to 1.0 of probability whether a track contains no vocals.|
|danceability|float|A measure from 0.0 to 1.0 how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity|
|energy|float|A measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity|
|liveness|float|Detects the presence of an audience in the recording|
|loudness|float|The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks|
|acousticness|float|A confidence measure from 0.0 to 1.0 of whether the track is acoustic|
|valence|float|A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track|
|speechiness|float|A measure from 0.0 to 1.0 that detects the presence of spoken words in a track.|
|tempo|float|The overall estimated tempo of a track in beats per minute (BPM)|
|key|integer|The key the track is in. Integers map to pitches using standard Pitch Class notation|
|mood|object|Labeled mood of the playlist the track is in|
|mood_map|integer|Integer that represents the corresponding mood|
|vibe|integer|Kmeans clustering cluster|

### **App**
The model was deployed using Streamlit where a user is able to enter the id of a public playlist and the app returns a mood profile which details a percentage of each mood of the playlist. 

### **Conclusion & Future Work**
This project serves as a method to prototype music mood classification and prediction based on audio features of a track within a playlist. Throughout this process, I found that there are many different ways of approaching this problem and the problems within to improve functionality. This revealed the importance of intuition and creativity of the data scientist or team when creating this project. 

More data always improves model accuracy, however getting labels for mood of a track or playlist and method of labeling posed as a challenge. As highlighted in the modeling phase, mood and emotion is subjective, therefore labeling introduces some subjectivity and human interaction with the data. Additionally, the songs found in each playlist were labeled based off of the playlist mood which was determined by the name and description of the playlist. However, if evaluating the song individually rather than as the whole of the playlist, the mood of the track may have been labeled differently. 

For future work, I plan to explore other methods of labeling mood, expanding the target class to predict more mood classes, as well as incorporating lyric data with Natural Language Processing to improve the model. I also plan to create a recommender system based on the predicted mood and mood profile and make improvements on the streamlit app. The app is currently a blueprint, however I would like to add more features and functionality for user interaction including Spotify user login so that a user can evaluate their own playlists, recommend songs/playlists based on mood, and incorporate audio visualizations and other therapeutic techniques and content as an effective alternative for those can't afford therapy.


#### **Resources**

https://towardsdatascience.com/how-to-get-data-from-apis-with-python-dfb83fdc5b5b

https://www.youtube.com/watch?v=xdq6Gz33khQ&t=2512s

https://medium.com/@maxtingle/getting-started-with-spotifys-api-spotipy-197c3dc6353b

https://www.gilbertgalindo.com/importanceofmusic

https://www.musictech.net/news/spotify-patent-detect-emotion-recommend-music/
