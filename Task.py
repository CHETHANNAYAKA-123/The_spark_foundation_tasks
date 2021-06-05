#!/usr/bin/env python
# coding: utf-8

# # GRIPJUNE2 @THE SPARKS FOUNDATION (COMPANY).
# 
# ## DATA SCIENCE AND BUSINESS ANALYTICS
# 
# ## AUTHOR:CHETHAN NAYAKA C
# 
# ## TASK1: PREDICTION USING SUPERVISED MACHINE LEARNING
# 

# ## Linear Regression with Python Scikit Learn
# 
# #### In the section we will see how the Python Scikit_learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables 

# ## Simple Linear Regression
# #### In this task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied.This is a simple linear regression task as it involves just two variables.

# In[1]:


#importing the required libraries and dataset


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Reading data from given link
url = "http://bit.ly/w-data"
data=pd.read_csv(url)
print("data imported successfully")
data.head(10)


# ## Visualizing the data

# In[4]:


# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### From the above graph,we can see there is a positive linear relation between the number of hours studied and percentage of score 

# ### Preparing the data

# In[5]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[6]:


#importing train_test_split from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 
X_train.shape


# ### Applying Linear Regression

# In[7]:


from sklearn.linear_model import LinearRegression  
L_regresson = LinearRegression()  
L_regresson.fit(X_train, y_train)


# ### Plotting the regression line

# In[8]:


line = L_regresson.coef_*X+L_regresson.intercept_
plt.title("Plotting for the test data")
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# ### Making predictions

# In[9]:


print(X_test)
y_pred = L_regresson.predict(X_test)
y_pred


# In[10]:


#Compare the actual value and predicted


# In[11]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# ### Predicting the score of the student based on hour studied

# In[12]:


hours = 9.25
o_pred = L_regresson.predict([[hours]])
print("No of Hours studied by student= {}".format(hours))
print("Predicted Score of the student = {}".format(o_pred[0]))


# ### Accuracy of the model

# In[13]:


np.round(L_regresson.score(X_test,y_test)*100,2)


# ### Evaluating the model

# #### The final step is to evaluate the performance of algorithm. We have chosen the mean square error.

# In[14]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 

