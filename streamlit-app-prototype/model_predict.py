#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import datetime
import missingno as msno
import tensorflow as tf

# Import from sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, OneHotEncoder, LabelBinarizer, LabelEncoder
import category_encoders as ce
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay, confusion_matrix, plot_confusion_matrix, multilabel_confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, plot_roc_curve, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier

import pickle

# Set a random seed
from numpy.random import seed
seed(8)
from tensorflow.random import set_seed
set_seed(8)

# Data Visualization
sns.set_theme(context='notebook', style='darkgrid', palette='viridis')

import warnings
warnings.filterwarnings("ignore")


# In[10]:


mood = pd.read_csv('mood_label.csv')


# In[12]:


mood['vibe'] = mood['mood_map'].map({0:0, 1:0, 2:1})
X = mood[['danceability', 
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

y = mood['mood_map']

# set up train_test_split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    stratify=y, 
                                                    random_state=42)


# In[13]:


get_ipython().run_cell_magic('time', '', 'pipe = Pipeline([\n    (\'poly\', PolynomialFeatures(degree=2)),\n    ("minmax", MinMaxScaler()),\n    ("pc", PCA(n_components=32)),\n    ("logreg", LogisticRegressionCV(multi_class=\'multinomial\', random_state=42, max_iter=50))])\n\npipe.fit(X_train, y_train)')


# In[347]:


def audio_analysis(playlist_id):    
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

    # user credentials
    client_id = 'a71512db396c4f9f9132928928f9ddff'
    client_secret = '66041252b1f44debae39f3e09953dd74'
    redirect_url = 'https://www.google.com'
    user_id = "bsanzone225"
    genius_access_token = "F_FuandD31vRz1yy6nXxXyeK4eNw6fIlFzGN72_PJRgQkhE294AIk3m_jGUHd03E"
    scope_playlist = 'playlist-read-private'
    
    # get user authorization
    user_auth_manager = SpotifyOAuth(scope=scope_playlist,client_id=client_id,client_secret=client_secret,username=user_id, redirect_uri=redirect_url)
    sp = spotipy.Spotify(auth_manager=user_auth_manager)
    
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
    
    
    pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ("minmax", MinMaxScaler()),
    ("pc", PCA(n_components=32)),
    ("logreg", LogisticRegressionCV(multi_class='multinomial', random_state=42, max_iter=50))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X)
    preds_list=[x for x in preds]
    
    n = len(preds_list)
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
    return print(f'This playlist is {np.round((n_sad/(n))*100,0)}% Sad, {np.round((n_chill/(n))*100,0)}% Chill, and {np.round((n_energetic/(n))*100,0)}% Energetic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


mood['mood'].value_counts(normalize=True)


# In[31]:


# X audio features
X = mood[['danceability', 
          'energy', 
          'loudness', 
          'speechiness', 
          'acousticness',
          'instrumentalness', 
          'liveness', 
          'valence', 
          'tempo',
          'key']]

# response variable
y = mood['mood_map']

# # set up train_test_split with stratification to include equal classes of each
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=42)

# standardize X data
x = MinMaxScaler()
X_train_x = x.fit_transform(X_train)
X_test_x = x.transform(X_test)

# LabelBinarizer for multiclass classification response variable
# binarize = LabelBinarizer()
# y_train_binarize = binarize.fit_transform(y_train)
# y_test_binarize = binarize.transform(y_test)


# In[32]:


logreg=LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(X_train_x, y_train)
# Score on training and testing sets.
print(f'Training Score: {round(logreg.score(X_train_x, y_train),4)}')
print(f'Testing Score: {round(logreg.score(X_test_x, y_test),4)}')


# In[33]:


# X audio features
X_audio = mood[['danceability', 
          'energy', 
          'loudness', 
          'speechiness', 
          'acousticness',
          'instrumentalness', 
          'liveness', 
          'valence', 
          'tempo',
          'key']]

# response variable
y = mood['mood_map']


pf = PolynomialFeatures(degree = 3)
X = pf.fit_transform(X_audio)

# # set up train_test_split with stratification to include equal classes of each
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    stratify=y, 
                                                    random_state=42)

# standardize X data
x = MinMaxScaler()
X_train_x = x.fit_transform(X_train)
X_test_x = x.transform(X_test)

# instantiate PCA
pca = PCA(n_components=40, random_state = 42)
Z_train = pca.fit_transform(X_train_x)
Z_test = pca.transform(X_test_x)

# LabelBinarizer for multiclass classification response variable
binarize = LabelBinarizer()
y_train_binarize = binarize.fit_transform(y_train)
y_test_binarize = binarize.transform(y_test)


# In[34]:


def mood_model():
    
    n_input = Z_train[0].shape

    model = Sequential()

    model.add(Dense(12, 
                    activation='relu', 
                    input_shape=(n_input)))
    model.add(Dropout(0.2))
    
    model.add(Dense(6, activation='relu'))

    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[['accuracy'],
                           [tf.keras.metrics.Precision()],
                           [tf.keras.metrics.Recall()]])

    return model


# In[35]:


classifier_2 = KerasClassifier(build_fn=mood_model,
                             epochs=200,
                             verbose=0)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, 
                               verbose=1, mode='auto')
history_2=classifier_2.fit(Z_train,
               y_train_binarize,
               validation_data=(Z_test, y_test_binarize),
               epochs=200,
               batch_size=32,
               callbacks=[early_stop],
               verbose=0)
print(f'Train score: {classifier_2.score(Z_train, y_train_binarize)}')
print(f'Test score: {classifier_2.score(Z_test, y_test_binarize)}')


# In[36]:


logreg=LogisticRegression(multi_class='multinomial', solver='lbfgs')
logreg.fit(Z_train, y_train)
# Score on training and testing sets.
print(f'Training Score: {round(logreg.score(Z_train, y_train),4)}')
print(f'Testing Score: {round(logreg.score(Z_test, y_test),4)}')


# In[37]:


abc = AdaBoostClassifier(random_state=42, n_estimators=3)
abc.fit(Z_train,y_train)
print(abc.score(Z_train, y_train))
print(abc.score(Z_test, y_test))


# In[38]:


gbc = GradientBoostingClassifier(n_estimators = 10, max_depth=1, random_state=42)
gbc.fit(Z_train, y_train)
print(gbc.score(Z_train, y_train))
print(gbc.score(Z_test, y_test))


# In[39]:


train_acc = history_2.history['accuracy']
test_acc = history_2.history['val_accuracy']
train_recall = history_2.history['recall_1']
test_recall = history_2.history['val_recall_1']
train_precision = history_2.history['precision_1']
test_precision = history_2.history['val_precision_1']
plt.figure(figsize=(12,8))
plt.plot(history_2.history['loss'], label='Training Loss', color="darkorchid")
plt.plot(history_2.history['val_loss'], label='Validation Loss', color='plum')
plt.plot(train_recall, label='Training Recall', color='palegreen')
plt.plot(test_recall, label='Testing Recall', color = 'darkgreen')
plt.plot(train_acc, label='Training Accuracy', color='turquoise')
plt.plot(test_acc, label='Testing Accuracy', color="darkslategray");
plt.plot(train_precision, label='Training Precision', color="darkorange")
plt.plot(test_precision, label='Testing Precision', color='peachpuff')
plt.legend();


# In[40]:


print(f'Train score: {classifier_2.model.evaluate(Z_train, y_train_binarize)}')
print(f'Test score: {classifier_2.model.evaluate(Z_test, y_test_binarize)}')


# In[41]:


y_preds_2=classifier_2.predict(Z_test)


# In[44]:


cm_2 = confusion_matrix(y_test, y_preds_2, labels=classifier_2.classes_)
cm_df_2 = pd.DataFrame(cm, index=['Sad', 'Calm', 'Energetic'], columns=['Sad', 'Calm', 'Energetic'])
plt.figure(figsize=(6,5))
sns.heatmap(cm_df, annot=True, cmap='viridis')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()


# In[47]:


pred_df_2 = pd.DataFrame({'true_values'     : y_test,
                        'prediction_prob' : classifier_2.predict_proba(Z_test)[:,1]})


# In[48]:


# Generate class membership probabilities
y_pred_probs_2 = classifier_2.predict_proba(Z_test)

roc_curve_weighted=roc_auc_score(y_test, y_pred_probs_2, average="weighted", multi_class="ovr")
roc_curve_macro=roc_auc_score(y_test, y_pred_probs_2, average='macro', multi_class="ovr")
print(roc_curve_weighted)
print(roc_curve_macro)


# In[49]:


# roc curve for classes
fpr = {}
tpr = {}
thresh ={}

n_class = 3

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_pred_probs_2[:,i], pos_label=i)
    
# plotting  
sns.set_palette('viridis')
plt.plot(fpr[0], tpr[0], linestyle='-',label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='-',label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='-',label='Class 2 vs Rest')

# add worst case scenario line
plt.plot([0, 1], [0, 1], label="baseline", linestyle="--")

plt.title('Multiclass ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC'); 


# #### **Transfer with KMeans**

# In[50]:


mood.info()


# #### **Null model:** 
# ###### Evaluate class imbalances
# ###### Classes are fairly balanced

# In[51]:


mood['mood'].value_counts(normalize=True)


# In[53]:


get_ipython().run_cell_magic('time', '', "# set X features\nX = mood[['danceability', \n               'energy', \n               'loudness', \n               'speechiness', \n               'acousticness',\n               'instrumentalness', \n               'liveness', \n               'valence', \n               'tempo']]\n\n# standardize X data\nxc = StandardScaler()\nX_xc= xc.fit_transform(X)\n\nscores = []\nfor k in range(2,31):\n    cl = KMeans(n_clusters=k)\n    cl.fit(X_xc)\n    inertia = cl.inertia_\n    sil = silhouette_score(X_xc, cl.labels_)\n    scores.append([k, inertia, sil])\nscore_df = pd.DataFrame(scores)\nscore_df.columns = ['k', 'inertia', 'silhouette']")


# In[57]:


score_df.head()


# In[58]:


km = KMeans(n_clusters=4)
km.fit(X_xc)
# Scaled X gives best results
mood['cluster'] = km.labels_


# In[66]:


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
x = mood['energy']
y = mood['danceability']
z = mood['acousticness']
a = mood['speechiness']
ax.scatter(x,y,z,c=mood['cluster'], s=40, cmap='viridis')
ax.legend(fontsize=12, title_fontsize=12)
ax.set_xlabel('Energy',fontsize=12)
ax.set_ylabel('Danceability',fontsize=12)
ax.set_zlabel('Loudness',fontsize=12)
ax.set_title("3D Scatter Plot of Songs Clusters");

