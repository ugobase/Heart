#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


b = pd.read_csv('heart.csv')


# In[3]:


b


# In[4]:


b.isnull().sum()


# In[5]:


b.head()


# In[6]:


b.tail()


# In[7]:


b.shape


# In[8]:


b.shape[0]


# In[9]:


b.thal.nunique()


# In[10]:


b.insert(2, 'gender',b.sex.apply(lambda x: 'Male' if x > 0 else 'Female'))
b


# In[11]:


b.describe()


# In[12]:


b.trestbps.nunique()


# In[13]:


b.sort_values('trestbps', ascending = False).head(5)


# In[14]:


b.nlargest(5, ['trestbps'])


# In[15]:


b.sort_values('age', ascending = False).head(15)


# In[16]:


b.nlargest(15,['age'])


# In[17]:


b.sort_values('age', ascending = True).head(15)


# In[18]:


b.nsmallest(15, ['age'])


# In[19]:


b.chol.value_counts()


# In[20]:


b[b.gender.str.contains('Male')]


# In[21]:


b[b.gender.str.contains('Female')]


# In[22]:


b[b.gender== 'Female']


# In[23]:


b[b.thal.isin(['3', '2'])]


# In[24]:


b[b.chol.isin(['212', '203'])].shape[0]


# In[25]:


b[b.chol.isin(['212', '203'])]


# In[26]:


b['slope'] = b.slope.apply(lambda x: x + 2)
b


# In[27]:


b.loc[(b['gender'] == 'Male') & (b['age'] >= 50)]


# In[28]:


b[(b.gender == 'Male') & (b.age >= 50)]


# In[29]:


b.loc[(b['gender'] == 'Female') & (b['age'] <= 50)]


# In[30]:


b[(b.gender == 'Female') & (b.age <= 50)]


# In[31]:


b[(b['gender'] == 'Female') & (b.age >= 60) &  (b.slope == 2)].shape[0]


# In[32]:


b.loc[(b.gender == 'Female') & (b['age'] >= 60) &  (b['slope'] == 2)].shape[0]


# In[33]:


b.loc[(b['cp'] == 2) & (b['slope'] == 2) & (b['target'] == 1) & (b['thal'] == 3)]


# In[34]:


b[(b.cp == 2) & (b.slope == 2) & (b.target == 1) & (b.thal == 3)]


# In[35]:


b.loc[(b['gender'] == 'Female') | (b['age'] >= 70)]


# In[36]:


b[(b.gender == 'Female') | (b.age >= 70)]


# In[37]:


b.gender.value_counts()


# In[38]:


b[b.age >= 50].gender.value_counts()


# In[39]:


b.loc[(b['gender'] == 'Female')].groupby(['age']).count().shape[0]


# In[40]:


b.loc[b.gender == 'Female'].age.value_counts().shape[0]


# In[41]:


b[b.gender == 'Female'].age.value_counts().shape[0]


# In[42]:


b.groupby('gender').age.describe()


# In[43]:


b.groupby('gender').age.describe().value_counts()


# In[44]:


b.groupby('age').gender.describe().shape[0]


# In[45]:


b.groupby('age').gender.describe()


# In[46]:


b.groupby('age').thal.describe()


# In[47]:


plt.figure(figsize=(12, 6))
sns.histplot(b['chol'])


# In[48]:


plt.figure(figsize=(12, 6))
sns.boxplot(data = b, x = 'age' , y = 'chol')


# In[49]:


plt.figure(figsize=(12, 6))
sns.displot(b['slope'], kind = 'kde')


# In[50]:


plt.figure(figsize=(8, 6))
sns.countplot(b['slope'])


# In[51]:


plt.figure(figsize=(8, 6))
sns.countplot(b['gender'])


# In[52]:


b.thal.value_counts().plot(kind = 'bar', figsize = (6,6))


# In[53]:


b.thal.value_counts().plot(kind = 'pie', figsize = (8,6))


# In[54]:


plt.figure(figsize = (10,6))
sns.heatmap(b.corr(), annot = True, fmt = '0.1f')


# In[ ]:




