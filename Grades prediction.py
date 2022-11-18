#!/usr/bin/env python
# coding: utf-8

# ## Creating different models on Grades Dataset
# #### Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split


# ## 1. Data preprocessing

# In[2]:


df=pd.read_csv(r'../input/grades-of-students/Grades.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# ## Renaming Headers ( since course codes are in the format XX-XXX)

# In[5]:


df.columns


# In[6]:


# few headers have inconsistent naming we rename them to ease access later and for the sake of convention of course codes
df.rename(columns={'HS-105/12': 'HS-105', 'HS-205/20': 'HS-205'},inplace=True)


# In[7]:


df.columns


# ## Dropping inconsistent columns 

# In[8]:


# Since seat number doesnot seem like a strong predictor of cgpa we drop it
df.drop(['Seat No.'],axis='columns',inplace=True)


# In[9]:


df.head()


# ## Treating missing values 

# In[10]:


missingdata=df.isnull()


# In[11]:


#returns counts of values where True if null
for column in missingdata.columns.values.tolist():
    print(column)
    print (missingdata[column].value_counts())
    print("")    


# Many columns have missing values we must treat them.

# In[12]:


# returns records where HS-205 has null values
df[df['HS-205'].isnull()]


# In[13]:


# for each column, get value counts in decreasing order and take the index (value) of most common class
# replaces missing data with modes

df_most_common_imputed = df.apply(lambda x: x.fillna(x.value_counts().index[0]))


# In[14]:


#confirmation that all null values are replaced 
for column in df_most_common_imputed.isnull().columns.values.tolist():
    print(column)
    print (df_most_common_imputed.isnull()[column].value_counts())
    print("")


# ## Comparing value counts before after 

# In[15]:


for i in df.columns:
    x = df[i].value_counts()
    print("\nColumn name is:",i,"and it value is:\n",x)


# In[16]:


# this is the dataframe we imputed with modes
for i in df_most_common_imputed.columns:
    x = df_most_common_imputed[i].value_counts()
    print("\nColumn name is:",i,"and it value is:\n",x)


# ## Removing records with inconsistent grades
# Note: These grades are consistent to be fair, they have a meaning but for simplicity I removed them ; generally we should'nt since they hold a meaning and have their own weight in prediction of target.

# In[17]:


# locates records with Wu and W then drops by their indices
DF=df_most_common_imputed   
for i in DF.columns:
    DF.drop(DF[(DF.loc[:,i]=='WU')| (DF.loc[:,i]=='W')].index,inplace=True)


# In[18]:


DF.reset_index(drop=True,inplace=True)


# In[19]:


DF


# In[20]:


# confirms the removal of WU and W graded records
for column in DF.columns.values.tolist():
    print(column)
    print (DF[column].isin(['WU','W']).value_counts())
    print("")


# In[21]:


DF['PH-121'].unique()


# ## Encoding Categorical Values

# The grades are encoded with the GPAs they are equivalent to

# In[22]:


for column in DF.columns:
    
    DF[column]=DF[column].replace('A+',4.0)
    DF[column]=DF[column].replace('A',4.0)
    DF[column]=DF[column].replace('A-',3.7)
    DF[column]=DF[column].replace('B+',3.4)
    DF[column]=DF[column].replace('B',3.0)
    DF[column]=DF[column].replace('B-',2.7)
    DF[column]=DF[column].replace('C+',2.4)
    DF[column]=DF[column].replace('C',2.0)
    DF[column]=DF[column].replace('C-',1.7)
    DF[column]=DF[column].replace('D+',1.4)
    DF[column]=DF[column].replace('D',1.0)
    DF[column]=DF[column].replace('F',0.0)


# In[23]:


DF.head()


# In[24]:


DF.info()


# ## Visualizing correlation by heatmap and further checking feature significance

# In[25]:


plt.figure(figsize=(20,20))
sns.heatmap(DF.corr(),annot=True)
plt.show()
# the heat map looks complicated but just focus on the last column the correlation of all the course GPAs are pretty strong as it should be.


# In[26]:


# removing target variable
Z=DF.drop(['CGPA'],axis='columns')
Z


# #  Creating Models
# 
# ### First Model ( prediction of final CGPA based on GPs of first 3 years using Linear Regression & Decision Tree Regressor )
# 
# ### a) Linear Regression

# In[27]:


from sklearn.linear_model import LinearRegression
am = LinearRegression()
am


# In[28]:


# filters for selecting only first 3 year courses
first_three_years=Z
for column in Z.columns[Z.columns.str.contains('-4')]:
    first_three_years.drop([column],axis='columns',inplace=True)

first_three_years


# In[29]:


# confirms no final year courses
first_three_years.columns[first_three_years.columns.str.contains('-4')]


# In[30]:


first_three_years.shape


# In[31]:


am.fit(first_three_years,DF['CGPA'])


# In[32]:


Yhat=am.predict(first_three_years)
Yhat[0:5]


# In[33]:


DF.loc[0:5,'CGPA']


# ## We can look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

# In[34]:


width = 12
height = 10
plt.figure(figsize=(width, height))


ax1 = sns.distplot(DF['CGPA'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for CGPA')
plt.xlabel('CGPA')
plt.ylabel('Grades till 3rd year')

plt.legend()
plt.show()
plt.close()


# ## R-score & Mean Squared Error  Of Linear Regression

# In[35]:


from sklearn.metrics import mean_squared_error
R_model1_lm=am.score(first_three_years,DF['CGPA'])*100

mse_model1_lm = mean_squared_error(DF['CGPA'],Yhat)


# In[36]:


print(" R^2: {:.2f} %".format(R_model1_lm))
print("The mean square error of CGPA and predicted value is: {:.5f}".format(mse_model1_lm))


# ## b) Decision Tree

# In[37]:


from sklearn.tree import DecisionTreeRegressor

X_train,X_test,Y_train,Y_test = train_test_split(first_three_years,DF[['CGPA']],test_size=0.40,random_state=0)
decisiontree = DecisionTreeRegressor()
dec_tree = decisiontree.fit(X_train, Y_train)


# In[38]:


Yhat2=dec_tree.predict(X_test)


# In[39]:


Yhat2[0:5]


# In[40]:


DF.loc[0:5,'CGPA']


# ## R-score & MSE Of Decision Tree 

# In[41]:


R_model1_DT=dec_tree.score(X_test,Y_test)*100

mse_model1_DT = mean_squared_error(Y_test,Yhat2)


# In[42]:



print("Test set R^2: {:.2f} %".format(R_model1_DT))

print("The mean square error of CGPA and predicted value is: {:.5f}".format(mse_model1_DT))


# # Second Model  ( prediction of final CGPA based on GPs of first two years using KNN & SVM )

# In[43]:


# filtering out for first two years
first_two_years=first_three_years
for column in first_three_years.columns[first_three_years.columns.str.contains('-3')]:
    first_two_years.drop([column],axis='columns',inplace=True)
first_two_years
    


# In[44]:


first_two_years.columns[first_two_years.columns.str.contains('-3')]


# ## a) KNN

# In[45]:


from sklearn.neighbors import KNeighborsRegressor
X_train, X_test, y_train, y_test = train_test_split(first_two_years, DF['CGPA'],test_size=0.4 ,random_state=0)


# In[46]:


KNN = KNeighborsRegressor(n_neighbors=3)
KNN.fit(X_train, y_train)
y_pred=KNN.predict(X_test)
# print("Test set predictions:\n{}".format(KNN.predict(X_test)))


# ## R-score & Mean Squared Error Of KNN

# In[47]:


R_model2_KNN=KNN.score(X_test, y_test)*100
mse_model2_KNN = mean_squared_error(y_test,y_pred)


# In[48]:


print("Test set R^2: {:.2f} %".format(R_model2_KNN))

print("The mean square error of CGPA and predicted value is: {:.5f}".format(mse_model2_KNN))


# ## b) SVM

# In[49]:


from sklearn.svm import SVR
X_train,X_test,Y_train,Y_test = train_test_split(first_two_years, DF['CGPA'],test_size=0.50,random_state=0)


# In[50]:


sv = SVR(kernel='linear')
sv.fit(X_train, Y_train)


# In[51]:


X_train.shape


# In[52]:


y_pred2 = sv.predict(X_test)


# ## R-score & Mean Squared Error Of SVM

# In[53]:


R_model2_SVM=sv.score(X_test, Y_test)*100
mse_model2_SVM = mean_squared_error(Y_test,y_pred2)


# In[54]:


print("Test set R^2: {:.2f} %".format(R_model2_SVM))

print("The mean square error of CGPA and predicted value is: {:.5f}".format(mse_model2_SVM))


# ## Third Model ( prediction of final CGPA based on GPs of first year using Linear Regression & GPR )

# In[55]:


first_year=first_two_years
for column in Z.columns[Z.columns.str.contains('-2')]:
    first_two_years.drop([column],axis='columns',inplace=True)
first_year


# ## a) Linear Regression

# In[56]:


lm2 = LinearRegression()
lm2
lm2.fit(first_year,DF['CGPA'])


# In[57]:


pred=lm2.predict(first_year)


# ## R-score & Mean Squared Error Of Linear Regression

# In[58]:


R_model3_lr=lm2.score(first_year, DF['CGPA'])*100
mse_model3_lr = mean_squared_error(DF['CGPA'],pred)


# In[59]:



print(" R^2: {:.2f} %".format(R_model3_lr))

print("The mean square error of CGPA and predicted value is: {:.5f}".format(mse_model3_lr))


# ## b) Gaussian Process Regression

# In[60]:


from sklearn.gaussian_process.kernels import RBF
import sklearn.gaussian_process as gp


# In[61]:


X_train,X_test,Y_train,Y_test = train_test_split(first_year,DF[['CGPA']],test_size=0.40,random_state=0)


# In[62]:


kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))


# In[63]:


model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)


# In[64]:


model.fit(X_train, Y_train)
params = model.kernel_.get_params()


# In[65]:


pred2 = model.predict(X_test)


# ## R-score & Mean Squared Error Of Gaussian Process Regression

# In[66]:


R_model3_GPR=model.score(X_test, Y_test)*100
mse_model3_GPR = mean_squared_error(Y_test,pred2)


# In[67]:



print("Test set R^2: {:.2f} %".format(model.score(X_test, Y_test)*100))

print("The mean square error of CGPA and predicted value is: {:.5f}".format(mse_model3_GPR))


# ## R-score & Mean Squared Error of all models

# In[68]:


print(" R^2 of linear regression of first model : {:.2f} %".format(R_model1_lm))
print("The mean square error of CGPA and predicted value is: {:.5f}\n".format(mse_model1_lm))

print("Test set R^2 Decision Tree of first model: {:.2f} %".format(R_model1_DT))
print("The mean square error of CGPA and predicted value is: {:.5f}\n".format(mse_model1_DT))

print("Test set R^2 KNN of second model: {:.2f} %".format(R_model2_KNN))
print("The mean square error of CGPA and predicted value is: {:.5f}\n".format(mse_model2_KNN))


print("Test set R^2 of SVM second model: {:.2f} %".format(R_model2_SVM))
print("The mean square error of CGPA and predicted value is: {:.5f}\n".format(mse_model2_SVM))

print(" R^2 linear regression third model: {:.2f} %".format(R_model3_lr))
print("The mean square error of CGPA and predicted value is: {:.5f}\n".format(mse_model3_lr))

print("Test set GPR third model R^2: {:.2f} %".format(model.score(X_test, Y_test)*100))
print("The mean square error of CGPA and predicted value is: {:.5f}\n".format(mse_model3_GPR))

