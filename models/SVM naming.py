#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot


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


# Get training data
train = pd.read_csv("D:\Documenten\CamCAN\memory_scores.csv")


# In[5]:


# Create the input and target list
inputs = [] # The input list is the hippocampal time courses
z_inputs = [] # The z-scored hippocampal time courses (input data)
y = [] # naming performance (target data)

for i in range(0, len(train)):
    if train["CCID"][i] in id_course:
        data = id_course[train["CCID"][i]]
        inputs.append(data)
        z_inputs.append(stats.zscore(data))
        y.append(train["naming_binned"][i])


# In[6]:


# Split the dataset in random train (80%) and test (20%) subset
X_train, X_test, y_train, y_test = train_test_split(
    z_inputs, y, test_size = 0.2, random_state = 777)


# In[7]:


# Prepare the cross-validation procedure
cv = KFold(n_splits = 5, random_state = 7, shuffle = True)


# In[8]:


# Define classification model
clf = SVC(random_state = 777)


# In[9]:


# Set the hyperparameters
parameters = {
            "kernel":["rbf"],
            "cache_size": [100],
            "C":[1],
        }


# In[10]:


# Implementing GridSearchCV
GS = GridSearchCV(clf, parameters, cv = cv, verbose = 1)
GS.fit(X_train, y_train) # Train model on training data


# In[11]:


print(GS.best_estimator_) # Estimator that was chosen by the search
print(GS.best_score_) # Mean cross-validated score of the best_estimator
print(GS.best_params_) # Parameter setting that gave the best results on the hold out data


# In[12]:


# Evaluate performance on the test set
y_pred_test = GS.predict(X_test)
print(accuracy_score(y_pred_test, y_test))
print(confusion_matrix(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[13]:


# Predict decision_function with chosen model
svm_probs = GS.decision_function(X_test)


# In[14]:


# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
svm_auc = roc_auc_score(y_test, svm_probs)
print(svm_auc)


# In[15]:


# Calculate roc curves with the label of the positive class being "bad"
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs, pos_label = "bad")


# In[16]:


# Plot the roc curve for the model
pyplot.plot(svm_fpr, svm_tpr, label='svm')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

