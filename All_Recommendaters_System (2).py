#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


animes = pd.read_csv('anime.csv')
animes.head()


# In[3]:


ratings = pd.read_csv('rating.csv')
ratings.head()


# In[4]:



  
animes = pd.read_csv('anime.csv')
animes


# In[5]:


ratings = pd.read_csv('rating.csv')
ratings


# In[6]:


animes.isnull()


# In[7]:


ratings.isnull()


# In[8]:


animes.duplicated()


# In[9]:


ratings.duplicated()


# In[10]:


genre_list=pd.unique(animes['genre'])
genre_list


# In[11]:


animes.dtypes


# In[12]:


ratings.dtypes


# In[13]:


animes.min


# In[14]:


ratings.min


# In[15]:


animes.size


# In[16]:


ratings.size


# In[17]:


animes.shape


# In[18]:


ratings.shape


# In[19]:


animes.info()


# In[20]:


ratings.info()


# In[21]:


X=np.mean(animes)
X


# In[22]:


Y=np.mean(ratings)
Y


# In[23]:


animes.plot.scatter(x='rating',y='members')


# In[24]:


animes.describe()


# In[25]:


ratings.describe()


# In[26]:


animes.columns = ['anime_id','name','genre','type','episodes','rating','members']
animes_df = animes.merge(animes, on='anime_id')
animes_df.head()


# In[27]:


C = animes_df['rating_x'].mean()
C


# In[ ]:





# In[28]:


m= animes_df['members_x'].quantile(0.9)
m


# In[29]:


new_animes_df = animes_df.copy().loc[animes_df['members_x'] >= m]
print(new_animes_df.shape)


# In[30]:


new_animes_df


# In[31]:




def weighted_rating(x, C=C, m=m):
    v = x['members_x']
    R = x['rating_x']
    return (v/(v + m) * R) + (m/(v + m) * C)

new_animes_df["score"] = new_animes_df.apply(weighted_rating, axis=1)
new_animes_df = new_animes_df.sort_values('score', ascending=False)
new_animes_df[['name_x', 'members_x', 'rating_x', "score"]].head(10)


# In[32]:


def plot():
    ratings = animes_df.sort_values('rating_x', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(ratings['name_x'].head(10), ratings['rating_x'].head(10), align='center', color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 animes')
    plt.xlabel('Rating')
    plt.show()
plot()


# In[33]:


print(' Recommendation 10 name anime hightest rating:\n ')
ratings1 = animes_df.sort_values('rating_x', ascending=False).head(10)
ratings1[['name_x','rating_x']]


# In[ ]:





# In[34]:



def plot():
    members = animes_df.sort_values('members_x', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(members['name_x'].head(10), members['members_x'].head(10), align='center', color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 animes')
    plt.xlabel('Members')
    plt.show()
plot()


# In[35]:


print(' Recommendation 10 name anime hightest members:\n')
ratings1 = animes_df.sort_values('members_x', ascending=False).head(10)
ratings1[['name_x','members_x']]


# In[ ]:





# In[36]:


import numpy as np

from sklearn import datasets

np.random.seed(0)
iris = datasets.load_iris()
iris


# In[37]:


import spacy
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()
import time
animes = pd.read_csv('anime.csv')
animes.head()
def get_vector(x):
    doc = nlp(x)
    return doc.vector
a=get_vector(animes.genre[0])
a


# In[38]:


new_dataset = pd.DataFrame()


# In[39]:


import time

new_dataset['vec'] =animes[animes['genre'].notnull()]['genre'].apply(lambda x: get_vector(x))


# In[40]:


new_dataset['name']=animes['name']
new_dataset = new_dataset.drop(['name'],axis=1)


# In[41]:


new_dataset.insert(0,'name',animes['name'])


# In[42]:


aa = new_dataset['vec'].to_numpy()
aa = aa.reshape(-1,1)
aa


# In[43]:


nn = np.concatenate(np.concatenate(np.array(new_dataset['vec']).reshape(-1,1), axis = 0), axis = 0).reshape(-1,96)


# In[44]:


nn.shape


# In[45]:


df = pd.DataFrame(data = nn)
df.insert(0,'name',new_dataset['name'])
df


# In[46]:


from sklearn.manifold import TSNE


# In[47]:


ts= TSNE()
df.columns = df.columns.astype(str)
x=ts.fit_transform(df.loc[:, df.columns != 'name'])
x


# In[48]:


fi=pd.DataFrame(data=x)
fi['name']=new_dataset['name']


# In[49]:


fi.columns = ['x', 'y', 'name' ]
fi


# In[50]:


x=fi['x']
x


# In[51]:


y=fi['y']
y


# In[52]:


## su tuong quang o gia tri (-1,1) 
cs = fi.drop(['name'] , axis = 1)
np.array(cs.iloc[0]).reshape(-1,1)


# In[53]:


import gensim
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import gensim.downloader as api
import numpy as np
import seaborn as sns


# In[54]:


def cos(x,y):
    cs=x*y/(np.sqrt(sum((x*x)))*sum((y*y)))
    return cs


# In[55]:


cos_sim=pd.DataFrame(data=cos(cs.x,cs.y),columns=['cosin_sim'])


# In[56]:


cos_sim.insert(0,'name',new_dataset['name'])


# In[57]:


cos_sim=cos_sim.sort_values(by='cosin_sim')


# In[58]:


sns.lineplot(x='name',y='cosin_sim',data=cos_sim.sort_values(by='cosin_sim'))


# In[59]:


## Tim kieu mau tuong tu
idx=cos_sim[cos_sim.name=='White Tree'].index.values
## idx=np.concatenate(idx,axis=0);
idx.astype(int)


# In[68]:


## Sau cung, in ket qua goi y
## 6666 la chi so gia dinh co the thay doi 
## + - 5 la khoang gioi han tu cho
## Tim kieu mau tuong tu
result_recommentdation = cos_sim.iloc[10739 - 5 : 10739 + 5]
result_recommentdation


# In[69]:


result_recommentdation['name']


# In[62]:


get_ipython().system('pip install scikit-surprise ')


# In[63]:


import pandas as pd
from surprise import Dataset
from surprise import Reader


# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from ast import literal_eval

# in case you have placed the files outside of your working directory, you need to specify a path
path = '' # for example: 'data/movie_recommendations/'  

# load the movie metadata
df_anime=pd.read_csv(path + 'anime.csv', low_memory=False) 
print(df_anime.shape)
print(df_anime.columns)
df_anime.head(10)


# In[65]:


df_ratings=pd.read_csv(path + 'rating.csv', low_memory=False) 

print(df_ratings.shape)
print(df_ratings.columns)
df_ratings.head(10)

rankings_count = df_ratings.rating.value_counts().sort_values()
sns.barplot(x=rankings_count.index.sort_values(), y=rankings_count, color="Cyan")
sns.set_theme(style="whitegrid")


# In[66]:


reader = Reader()
ratings_by_users = Dataset.load_from_df(df_ratings[['user_id', 'anime_id', 'rating']], reader)

train_df, test_df = train_test_split(ratings_by_users, test_size=.2)


# In[1]:



"""# 10-fold cross validation 
cross_val_results = cross_validate(svd_model_trained, ratings_by_users, measures=['RMSE', 'MAE', 'MSE'], cv=10, verbose=False)
test_mae = cross_val_results['test_mae']

# mean squared errors per fold
df_test_mae = pd.DataFrame(test_mae, columns=['Mean Absolute Error'])
df_test_mae.index = np.arange(1, len(df_test_mae) + 1)
df_test_mae.sort_values(by='Mean Absolute Error', ascending=False).head(15)

# plot an overview of the performance per fold
plt.figure(figsize=(6,4))
sns.set_theme(style="whitegrid")
sns.barplot(y='Mean Absolute Error', x=df_test_mae.index, data=df_test_mae, color="b")
# plt.title('Mean Absolute Error')
"""


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




