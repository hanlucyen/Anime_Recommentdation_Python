#!/usr/bin/env python
# coding: utf-8

# Demo_Cartoon_Recommendation_System

# In[3]:


## Goi thu vien 
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[4]:


import matplotlib


# In[5]:


print(f"Number of ratings: ")


# In[6]:


import tsne


# In[7]:


from sklearn.manifold import TSNE


# In[8]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[10]:


ratings = pd.read_csv('rating_ani.csv')
ratings.head()


# In[11]:


animes = pd.read_csv('anime.csv')
animes.head()


# In[12]:


n_ratings = len(ratings)
n_ratings


# In[13]:


n_animes = len(ratings['anime_id'].unique())
n_animes


# In[14]:


n_users = len(ratings['user_id'].unique())
n_users


# In[15]:


user_freq = ratings[['user_id', 'anime_id']].groupby('user_id').count().reset_index()
user_freq.columns = ['user_id', 'n_ratings']
user_freq.head()


# In[16]:


mean_rating = ratings.groupby('anime_id')[['rating']].mean()


# In[17]:


lowest_rated = mean_rating['rating'].idxmin()
animes.loc[animes['anime_id'] == lowest_rated]


# In[18]:


highest_rated = mean_rating['rating'].idxmax()
animes.loc[animes['anime_id'] == highest_rated]


# In[19]:


ratings[ratings['anime_id']==highest_rated]


# In[20]:


ratings[ratings['anime_id']==lowest_rated]
  


# In[21]:


anime_stats = ratings.groupby('anime_id')[['rating']].agg(['count', 'mean'])
anime_stats.columns = anime_stats.columns.droplevel()


# In[22]:


from scipy.sparse import csr_matrix


# In[35]:


def create_matrix(df):
      
    N = len(df['user_id'].unique())
    M = len(df['anime_id'].unique())
      
    ## Id ánh xạ tới các chỉ số
    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    anime_mapper = dict(zip(np.unique(df["anime_id"]), list(range(M))))
      
    ## Ánh xạ chỉ số vào ID
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
    anime_inv_mapper = dict(zip(list(range(M)), np.unique(df["anime_id"])))
      
    user_index = [user_mapper[i] for i in df['user_id']]
    anime_index = [anime_mapper[i] for i in df['anime_id']]
  
    X = csr_matrix((df["rating"], (anime_index, user_index)), shape=(M, N))
      
    return X, user_mapper, anime_mapper, user_inv_mapper, anime_inv_mapper
  
X, user_mapper, anime_mapper, user_inv_mapper, anime_inv_mapper = create_matrix(ratings)


# In[24]:


from sklearn.neighbors import NearestNeighbors


# In[38]:


def find_similar_animes(anime_id, X, k, metric='cosine', show_distance=False):
    
    neighbour_ids = []
    
    anime_ind = anime_mapper[anime_id]
    anime_vec = X[anime_ind]
    k+=1
    
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    anime_vec = anime_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(anime_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(anime_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
  


# In[37]:


anime_names = dict(zip(animes['anime_id'], animes['name']))
anime_names  

anime_id = 3 
anime_similar_ids  = find_similar_animes(anime_id, X, k=10)
anime_name2= anime_names[anime_id]
anime_name2

print(f"Since you watched {name}")
for i in anime_similar_ids:
    print(anime_names[i])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




