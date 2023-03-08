#!/usr/bin/env python
# coding: utf-8

# ANIME RECOMMENTDATION
# 

# ## Thu _nhập dữ liệu &  thư viện

# In[1]:


## GOI THU VIEN CAN SU DUNG
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim


# In[2]:


## bo ghi 20GB
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


## doc file can su ly
anime =pd.read_csv('anime.csv')
anime


# In[4]:


ratings = pd.read_csv('rating.csv')
ratings


# In[5]:


## Phát hiện các giá trị bị thiếu cho một đối tượng giống như mảng.
anime.isnull().sum().sort_values(ascending = False)


# In[6]:


## mo ta the loai cua bo phim hoat hinh
anime = anime[['name','genre']]
anime


# In[7]:


## the loai cua bo thu 2 
anime.genre[1]


# In[8]:


## thu vie phan tich vector SpaCy

import spacy

## trich dan duong dan 

import en_core_web_sm
nlp = en_core_web_sm.load()

import time


# ## Chuẩn hóa dữ liệu theo hướng vector

# In[9]:


def get_vector(x):
    doc = nlp(x)
    return doc.vector


# In[10]:


get_vector(anime.genre[1])


# In[11]:


new_dataset = pd.DataFrame()


# In[12]:


get_ipython().run_cell_magic('time', '', "new_dataset['vec'] =anime[anime['genre'].notnull()]['genre'].apply(lambda x: get_vector(x))")


# In[13]:


new_dataset['name']=anime['name']
new_dataset = new_dataset.drop(['name'],axis=1)


# In[14]:


new_dataset.insert(0,'name',anime['name'])


# In[15]:


new_dataset


# In[16]:


aa = new_dataset['vec'].to_numpy()
aa = aa.reshape(-1,1)
aa


# In[17]:


nn = np.concatenate(np.concatenate(np.array(new_dataset['vec']).reshape(-1,1), axis = 0), axis = 0).reshape(-1,96)


# In[18]:


nn.shape


# In[19]:


df = pd.DataFrame(data = nn)
df.insert(0,'name',new_dataset['name'])
df


# In[20]:


## sklearn : được thiết kế để tương tác với các thư viện số và khoa học Python NumPy và khoa học.
## matplotlib : để thực hiện các suy luận thống kê , trực quan hóa dữ liệu.
import sklearn
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


# In[107]:



df.columns = df.columns.astype(str)
x=ts.fit_transform(df.loc[:, df.columns != 'name'])
x


# In[93]:


fi=pd.DataFrame(data=x)
fi['name']=new_dataset['name']


# In[94]:


fi.columns = ['x', 'y', 'name' ]
fi


# In[95]:


## cosine do muc do tuong tu giua hai vector
from sklearn.metrics.pairwise import cosine_similarity


# In[96]:


## su tuong quang o gia tri (-1,1) 
cs = fi.drop(['name'] , axis = 1)
np.array(cs.iloc[0]).reshape(-1,1)


# In[97]:


import gensim


# ## https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
# Similarity = (A.B) / (||A||.||B||) 
#         AB là tích vô hướng của A và B: Nó được tính bằng tổng tích các phần tử của A và B.
#         ||A|| là L2 chuẩn của A: Nó được tính bằng căn bậc hai của tổng bình phương các phần tử của vectơ A.
#  

# In[99]:


import numpy as np
from numpy.linalg import norm
 
cosine = np.sum(x*y, axis=1)/(norm(x, axis=1)*norm(y, axis=1))
cosine


# In[100]:


def cos(x,y):
    cs=x*y/(np.sqrt(sum((x*x)))*sum((y*y)))
    return cs


# In[101]:


cos_sim=pd.DataFrame(data=cos(cs.x,cs.y),columns=['cosin_sim'])


# In[102]:


cos_sim.insert(0,'name',new_dataset['name'])


# In[103]:


cos_sim=cos_sim.sort_values(by='cosin_sim')


# In[104]:


sns.lineplot(x='name',y='cosin_sim',data=cos_sim.sort_values(by='cosin_sim'))


# In[105]:


## Tim kieu mau tuong tu
idx=cos_sim[cos_sim.name=='White Tree'].index.values
## idx=np.concatenate(idx,axis=0);
idx.astype(int)


# In[106]:


## Nguyen Thi Huynh Nhu 20166050 
## Sau cung, in ket qua goi y
## 6666 la chi so gia dinh co the thay doi 
## + - 5 la khoang gioi han tu cho
result_recommentdation = cos_sim.iloc[6666 - 5 : 6666 + 5]
result_recommentdation

