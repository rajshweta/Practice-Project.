#!/usr/bin/env python
# coding: utf-8

# In[166]:


import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[167]:


happiness = pd.read_csv("D:/Shweta/happiness_score_dataset.csv")
happiness


# Data is in descending order (1 being the highest) of ranking

# First 5

# In[168]:


happiness.head()


# Last 5

# In[169]:


happiness.tail()


# In[170]:


happiness.columns


# In[171]:


happiness.info()


# In[172]:


sns.heatmap(happiness.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# No null values

# In[173]:


happiness.describe()


# In[174]:


sns.pairplot(happiness)


# In[175]:


sns.heatmap(happiness.corr(), annot = True)


# Happiness score is strongly dependant on Economy, Family, hyealth, and freedom. Moderately corelated to Trust, Dystopia residual. Generosity dosn't seem to have much impact

# In[176]:


happiness = happiness.drop('Generosity', axis = 1)
happiness


# We can also drop the columns Country, region, and rank since the score is dependent on other variables, not these

# In[177]:


happiness = happiness.drop(['Country', 'Region', 'Happiness Rank'], axis = 1)
happiness


# In[178]:


happiness.hist(figsize = (12,12))


# In[179]:


happiness.plot(kind = 'box', subplots = True, layout = (4,4), fontsize = 8, figsize = (12,12))


# There are outliers in Standard error, Trust, Family, and Dystopia Residual

# In[180]:


from scipy.stats import zscore
z = np.abs(zscore(happiness))
z


# In[181]:


print(np.where(z>3))


# In[182]:


happiness_new = happiness[(z<3).all(axis = 1)]
happiness_new


# In[183]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# In[184]:


happiness_new = happiness_new.apply(LabelEncoder().fit_transform)


# In[185]:


Y = happiness_new["Happiness Score"]
X = happiness_new.drop('Happiness Score', axis = 1)


# In[186]:


Y = Y.values.reshape(-1,1)
Y.shape


# In[187]:


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))


# In[188]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[189]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
print(Y_train.shape, Y_test.shape)


# In[190]:


lnr = LinearRegression()
lnr.fit(X_train, Y_train)


# In[191]:


predictions = lnr.predict(X_test)
predictions


# In[196]:


lnr_accuracy = round(lnr.score(X_train, Y_train)*100)
lnr_accuracy


# In[192]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[193]:


print('Mean absolute error:', mean_absolute_error(Y_test, predictions))
print('Mean squared error:', mean_squared_error(Y_test, predictions))
print('Root Mean squared error:', np.sqrt(mean_squared_error(Y_test, predictions)))
print('R2 score is:', r2_score(Y_test, predictions))


# In[194]:


import pickle


# In[195]:


filename = 'happiness_pickle.pkl'
pickle.dump(lnr, open(filename, 'wb'))


# In[ ]:




