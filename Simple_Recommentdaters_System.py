#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
animes = pd.read_csv('anime.csv')
animes


# In[21]:


animes.columns = ['anime_id','name','genre','type','episodes','rating','members']
animes_df = animes.merge(animes, on='anime_id')
animes_df.head()


# ## tính các chỉ số đơn giản

# In[22]:


C = animes_df['rating_x'].mean()
C


# In[23]:


m= animes_df['members_x'].quantile(0.9)
m


# In[24]:


new_animes_df = animes_df.copy().loc[animes_df['members_x'] >= m]
print(new_animes_df.shape)


# In[25]:



def weighted_rating(x, C=C, m=m):
    v = x['members_x']
    R = x['rating_x']
    return (v/(v + m) * R) + (m/(v + m) * C)

new_animes_df["score"] = new_animes_df.apply(weighted_rating, axis=1)
new_animes_df = new_animes_df.sort_values('score', ascending=False)
new_animes_df[['name_x', 'members_x', 'rating_x', "score"]].head(10)


# In[40]:



def plot():
    members = animes_df.sort_values('members_x', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(members['name_x'].head(10), members['members_x'].head(10), align='center', color='cyan')
    plt.gca().invert_yaxis()
    plt.title('Top 10 animes')
    plt.xlabel('Members')
    plt.show()
plot()


# In[27]:


## đề xuất 5 thể loại
animes_df['genre_x'].head(5)


# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[29]:


tfidf = TfidfVectorizer(stop_words='english')


animes_df['genre_x'] = animes_df['genre_x'].fillna('')


# In[30]:


tfidf_matrix = tfidf.fit_transform(animes_df['genre_x'])
tfidf_matrix


# In[31]:


tfidf_matrix.shape


# In[32]:


from sklearn.metrics.pairwise import linear_kernel


# In[33]:


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim


# In[34]:


indices = pd.Series(animes_df.index, index=animes_df['name_x']).drop_duplicates()
indices


# In[35]:


def get_recommendations(name, cosine_sim=cosine_sim):
    idx = indices[name]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]


    movie_indices = [i[0] for i in sim_scores]

  
    return animes_df['name_x'].iloc[indices]


# In[36]:


## ở đây đề xuất full chuỗi dữ liệu
get_recommendations('Naruto')


# In[39]:



filtered_names = get_recommendations('Sword Art Online')
filtered_names = random.choices(list(filtered_names))
filtered_names



# ## ở đây chỉ mới đề xuất ngẫu nhiên 1 phim hoạt hình
# ## em muốn được hướng dẫn làm sao đề xuất thêm nhiều hướng
