#!/usr/bin/env python
# coding: utf-8

# In[34]:



from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


# genres,id,keywords,title,overview,cast,crew
movies=movies[['genres','movie_id','keywords','title','overview','cast','crew']]


# In[7]:


movies.head()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.dropna(inplace=True)
movies.iloc[0].genres


# In[10]:


# obj is string is this
import ast
def convert(obj): 
    L=[]
#     to convert string into list
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
    


# In[11]:


movies['genres']=movies['genres'].apply(convert)


# In[12]:


movies.head()


# In[13]:


movies['keywords']=movies['keywords'].apply(convert)


# In[14]:


movies.head()


# In[15]:


import ast
def convert3(obj): 
    L=[]
    counter=0
#     to convert string into list
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            L.append(i['character'])
            counter+=1
        else:
            break
    return L


# In[16]:


movies['cast']=movies['cast'].apply(convert3)


# In[17]:


import ast
def fetch_director(obj): 
    L=[]
#     to convert string into list
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
             L.append(i['name'])
    return L


# In[18]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[19]:


movies.head()


# In[20]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[21]:


# we have to remove space between two words otherwise model will get 
# confused as there will be two tags with same name as one can be name 
# of director and other can be of cast
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[22]:


movies.head()


# In[23]:


movies['tags']=movies['overview']+ movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[24]:


movies.head()


# In[25]:


new_df=movies[['movie_id','title','tags']]
# convert tags list to string
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[26]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df.head()


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[28]:


vector=cv.fit_transform(new_df['tags']).toarray()
vector

# In[29]:
import requests
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

cv.get_feature_names()


# In[30]:
vector.shape
similarity = cosine_similarity(vector)

new_df[new_df['title'] == 'The Lego Movie'].index[0]


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    
    for i in distances[1:6]:
        # fetch the movie poster
        recommended_movie_posters = []
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_posters.append(movies.iloc[i[0]].title)
        recommended_movie_names.append(recommended_movie_posters)
    return recommended_movie_names


recommend('Gandhi')



# In[35]:
pickle.dump(new_df, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))


# In[ ]:



# %%






# %%

# %%

