#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

#Read the csv file
b = pd.read_csv('heart.csv')


# In[3]:

#View the file
b


# In[4]:

#Checking for null values
b.isnull().sum()


# In[5]:

#First 5 rows
b.head()


# In[6]:

#Last 5 rows
b.tail()


# In[7]:

#Shape to know number of rows and columns
b.shape


# In[8]:

#Number of rows
b.shape[0]


# In[9]:

#Number of unique values of thal
b.thal.nunique()


# In[10]:

#Creating a new column using lambda functions and apply 
b.insert(2, 'gender',b.sex.apply(lambda x: 'Male' if x > 0 else 'Female'))
b


# In[11]:

#Statistical description of dataset
b.describe()


# In[12]:

#Number of unique values of trestbps
b.trestbps.nunique()


# In[13]:

#Top 5 largest values of trestbps
b.sort_values('trestbps', ascending = False).head(5)


# In[14]:

#Alternative to previous
b.nlargest(5, ['trestbps'])


# In[15]:

#Top 15 values of age
b.sort_values('age', ascending = False).head(15)


# In[16]:

#Alternative to previous
b.nlargest(15,['age'])


# In[17]:

#Least 15 values of age
b.sort_values('age', ascending = True).head(15)


# In[18]:

#Alternative to previous
b.nsmallest(15, ['age'])


# In[19]:

#Value counts of chol
b.chol.value_counts()


# In[20]:

#Screening out all males in the gender column
b[b.gender.str.contains('Male')]


# In[21]:

#Screening out all females in the gender column
b[b.gender.str.contains('Female')]


# In[22]:

#Alternative to previous
b[b.gender== 'Female']


# In[23]:

#Screening out all 2's and 3's in thal
b[b.thal.isin(['3', '2'])]


# In[24]:

#Number of 212's and 203's in chol
b[b.chol.isin(['212', '203'])].shape[0]


# In[25]:

#Screening out all 212's and 203's in chol
b[b.chol.isin(['212', '203'])]


# In[26]:

#Using the lambda function and apply to add 2 to every value in slope
b['slope'] = b.slope.apply(lambda x: x + 2)
b


# In[27]:

#Screening out all males >= 50 years
b.loc[(b['gender'] == 'Male') & (b['age'] >= 50)]


# In[28]:

#Alternative to previous
b[(b.gender == 'Male') & (b.age >= 50)]


# In[29]:

#Screening out all females <= 50
b.loc[(b['gender'] == 'Female') & (b['age'] <= 50)]


# In[30]:

#Alternative to previous
b[(b.gender == 'Female') & (b.age <= 50)]


# In[31]:

#Screening out all females >= 60 and slope = 2
b[(b['gender'] == 'Female') & (b.age >= 60) &  (b.slope == 2)].shape[0]


# In[32]:

#Alternative to previous
b.loc[(b.gender == 'Female') & (b['age'] >= 60) &  (b['slope'] == 2)].shape[0]


# In[33]:

#Condition to screen out all cp = 2 and slope = 2 and target = 1 and thal = 1
b.loc[(b['cp'] == 2) & (b['slope'] == 2) & (b['target'] == 1) & (b['thal'] == 3)]


# In[34]:

#Condition to screen out all cp = 2 and slope = 2 and target = 1 and thal = 3
b[(b.cp == 2) & (b.slope == 2) & (b.target == 1) & (b.thal == 3)]


# In[35]:

#Condition to screen out all female or any gender >= 70
b.loc[(b['gender'] == 'Female') | (b['age'] >= 70)]


# In[36]:

#Alternative to previous
b[(b.gender == 'Female') | (b.age >= 70)]


# In[37]:

#Value count of gender
b.gender.value_counts()


# In[38]:

#Value count of gender >=50
b[b.age >= 50].gender.value_counts()


# In[39]:

#Number of values after grouping the age of females
b.loc[(b['gender'] == 'Female')].groupby(['age']).count().shape[0]


# In[40]:

#Alternative to previous
b.loc[b.gender == 'Female'].age.value_counts().shape[0]


# In[41]:

#Alternative to previous
b[b.gender == 'Female'].age.value_counts().shape[0]


# In[42]:

#Grouping and counting gender by age >=50
b.groupby('gender').age.describe()


# In[43]:

#Value count of statistical description of grouped age with gender
b.groupby('gender').age.describe().value_counts()


# In[44]:

#Number of values of statistical description of grouped age with gender
b.groupby('age').gender.describe().shape[0]


# In[45]:

#Grouping age with gender and statistical description
b.groupby('age').gender.describe()


# In[46]:

#Grouping age with thal and statistical description
b.groupby('age').thal.describe()


# In[47]:

#Histogram plot of chol
plt.figure(figsize=(12, 6))
sns.histplot(b['chol'])


# In[48]:

#Box plot of age against chol
plt.figure(figsize=(12, 6))
sns.boxplot(data = b, x = 'age' , y = 'chol')


# In[49]:

#Density plot of slope
plt.figure(figsize=(12, 6))
sns.displot(b['slope'], kind = 'kde')


# In[50]:

#Count plot of slope
plt.figure(figsize=(8, 6))
sns.countplot(b['slope'])


# In[51]:

#Count plot of gender
plt.figure(figsize=(8, 6))
sns.countplot(b['gender'])


# In[52]:

#Bar plot of thal
b.thal.value_counts().plot(kind = 'bar', figsize = (6,6))


# In[53]:

#Pie plot of thal
b.thal.value_counts().plot(kind = 'pie', figsize = (8,6))


# In[54]:

#Correlation plot of the dataset
plt.figure(figsize = (10,6))
sns.heatmap(b.corr(), annot = True, fmt = '0.1f')


# In[ ]:




