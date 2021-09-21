#!/usr/bin/env python
# coding: utf-8

# In[7]:


def audio_analysis(playlist_id):   
    import pandas as pd
    import numpy as np
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.pipeline import Pipeline

    # user credentials
    client_id = 'a71512db396c4f9f9132928928f9ddff'
    client_secret = '66041252b1f44debae39f3e09953dd74'
    redirect_url = 'https://www.google.com'
    user_id = "bsanzone225"
    genius_access_token = "F_FuandD31vRz1yy6nXxXyeK4eNw6fIlFzGN72_PJRgQkhE294AIk3m_jGUHd03E"
    scope_playlist = 'playlist-read-private'
    
    # get user authorization
    global_auth_manager = SpotifyClientCredentials(client_id, client_secret)
    sp = spotipy.Spotify(client_credentials_manager=global_auth_manager)
    
    # get list of track ids in playlist
    playlist = sp.playlist_tracks(playlist_id)
    song_ids = []
    for result in playlist['tracks']['items']:
        song_ids.append(result['track']['id'])
    
    features_data = []
    
    for track in song_ids:
        audio_features_list = sp.audio_features(tracks=track)

        for feature in audio_features_list:
        
            acousticness = feature['acousticness']
            danceability = feature['danceability']
            energy = feature['energy']
            instrumentalness = feature['instrumentalness']
            liveness = feature['liveness']
            valence = feature['valence']
            loudness = feature['loudness']
            speechiness = feature['speechiness']
            tempo = feature['tempo']
            key = feature['key']
            time_signature = feature['time_signature']
            track_id = feature['id']
            track = sp.track(track_id)
            name = track['name']
            artist = track['artists'][0]['name']
            length = track['duration_ms']
            popularity = track['popularity']
            features_data.append([name, 
                                  artist, 
                                  track_id, 
                                  instrumentalness, 
                                  danceability, 
                                  energy, 
                                  liveness, 
                                  loudness, 
                                  acousticness, 
                                  valence, 
                                  speechiness,
                                  tempo, 
                                  key,
                                  time_signature,
                                  length,
                                  popularity])
    df = pd.DataFrame(features_data, columns = ['name', 
                                                'artist', 
                                                'track_id', 
                                                'instrumentalness', 
                                                'danceability', 
                                                'energy', 
                                                'liveness', 
                                                'loudness', 
                                                'acousticness', 
                                                'valence', 
                                                'speechiness',
                                                'tempo', 
                                                'key',
                                                'time_signature',
                                                'length',
                                                'popularity'])
    
    X_audio = df[['danceability', 
       'energy', 
       'loudness', 
       'speechiness', 
       'acousticness',
       'instrumentalness', 
       'liveness', 
       'valence', 
       'tempo',
       'key']]
    
    xc = StandardScaler()
    X = xc.fit_transform(X_audio)
    
    km = KMeans(n_clusters=2)
    km.fit(X)
    df['vibe'] = km.labels_
    
    X = df[['danceability', 
           'energy', 
           'loudness', 
           'speechiness', 
           'acousticness',
           'instrumentalness', 
           'liveness', 
           'valence', 
           'tempo',
           'key',
           'vibe']]
    return X


# In[ ]:




