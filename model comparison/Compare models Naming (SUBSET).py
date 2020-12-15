#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
from scipy import stats
from mlxtend.evaluate import cochrans_q
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# In[2]:


# Load arrays or pickled objects from .npy
courses = np.load("time_courses.npy", allow_pickle=True)
ID = np.load("CCID.npy", allow_pickle=True)


# In[3]:


# Build dictionary from participant ID to hippocampal time course
id_course = {}

for i, ID in enumerate(ID):
    id_course[ID] = courses[i]


# In[4]:


# Get target data file (subset)
df = pd.read_csv("D:\Documenten\CamCAN\subset_targetdata.csv")


# In[5]:


# Create the input and target list
inputs = [] # The input list is the hippocampal time courses
z_inputs = [] # The z-scored hippocampal time courses (input data)
y = [] # naming performance (target data)

for i in range(len(df)):
    ID = df["CCID"].values[i] # Define participant ID
    if ID in id_course:
        data = id_course[ID] # Find data for participant based on ID
        inputs.append(data) # Add hippocampal time course to list
        z_inputs.append(stats.zscore(data)) # Add Z-scored hippocampal time course to list
        y.append(df["naming_binned"].values[i]) # Add target label to list


# In[6]:


# Split the dataset in random train (70%) and test (30%) subset
X_train, X_test, y_train, y_test = train_test_split(
    z_inputs, y, test_size = 0.3, random_state = 777)


# In[7]:


# Prepare the cross-validation procedure
cv = KFold(n_splits = 10, random_state = 7, shuffle = True)


# In[8]:


# Define best classification model
clf = SVC(random_state = 777, class_weight = "balanced")


# In[9]:


# Define Logistic Regression classification model
lr = LogisticRegression(random_state = 777, class_weight = "balanced")


# In[10]:


# Define Multi-layer Perceptron model
mlp = MLPClassifier(random_state = 777, nesterovs_momentum = True, 
                      learning_rate = "adaptive", max_iter = 200, early_stopping = True)


# In[11]:


# Set the parameters for SVM-model
parameters_svm = {
            "kernel":["linear"],
            "cache_size": [100],
            "C":[0.25],
        }


# In[12]:


# Set the parameters for LR-model
parameters_lr = {
            "penalty":["l2"],
            "C":[0.1],
            "solver": ["saga"],
        }


# In[13]:


# Set the hyperparameters MLP
parameters_mlp = {
            "hidden_layer_sizes": [(500)],
            "activation":["logistic"],
            "solver":["lbfgs"], 
        }


# In[14]:


# Implementing GridSearchCV on SVM model
model1 = GridSearchCV(clf, parameters_svm, cv = cv)
model1.fit(X_train, y_train) # Fitting on training data


# In[15]:


# Implementing GridSearchCV on Logistic Regression model
model2 = GridSearchCV(lr, parameters_lr, cv = cv)
model2.fit(X_train, y_train) # Fitting on training data


# In[16]:


# Implementing GridSearchCV on MLP classifier
model3 = GridSearchCV(mlp, parameters_mlp, cv = cv)
model3.fit(X_train, y_train) # Fitting on training data


# In[17]:


y_model1 = model1.predict(X_test)
y_model2 = model2.predict(X_test)
y_model3 = model3.predict(X_test)
y_test = np.array(y_test)


# In[18]:


# Results of Cochran's Q test
q, p_value = cochrans_q(y_test, y_model1, y_model2, y_model3)

print('Q: %.3f' % q)
print('p-value: %.3f' % p_value)

