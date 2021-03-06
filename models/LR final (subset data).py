#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
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


# Get target data file (subset)
df = pd.read_csv("D:\Documenten\CamCAN\subset_targetdata.csv")


# In[5]:


# Create the input and target list
inputs = [] # The input list is the hippocampal time courses
z_inputs = [] # The z-scored hippocampal time courses (input data)
y = [] # Final memory performance (target data)

for i in range(len(df)):
    ID = df["CCID"].values[i] # Define participant ID
    if ID in id_course:
        data = id_course[ID] # Find data for participant based on ID
        inputs.append(data) # Add hippocampal time course to list
        z_inputs.append(stats.zscore(data)) # Add Z-scored hippocampal time course to list
        y.append(df["final_score_binned"].values[i]) # Add target label to list


# In[6]:


# Split the dataset in random train (70%) and test (30%) subset
X_train, X_test, y_train, y_test = train_test_split(
    z_inputs, y, test_size = 0.3, random_state = 777)


# In[7]:


# Prepare the cross-validation procedure
cv = KFold(n_splits = 10, random_state = 7, shuffle = True)


# In[8]:


# Define classification model
lr = LogisticRegression(random_state = 777, class_weight = "balanced") 


# In[9]:


# Set the parameters by cross-validation
parameters = {
            "penalty":["l2", "l1", "elasticnet"],
            "C":[0.07, 0.1,0.3,0.5,0.8,1],
            "solver": ["newton-cg", "saga", "liblinear", "sag", "lbfgs"],
        }


# In[10]:


# Implementing GridSearchCV for hyperparameter tuning
GS = GridSearchCV(lr, parameters, cv = cv, verbose = 1, n_jobs = -1)
GS.fit(X_train, y_train) # Fitting on training data


# In[11]:


print(GS.best_estimator_) # Estimator that was chosen by the search (which gave highest score)
print(GS.best_score_) # Mean cross-validated score of the best_estimator
print(GS.best_params_) # Parameter setting that gave the best results on the hold out data


# In[12]:


# Evaluate performance on the test set
print(accuracy_score(y_pred_test, y_test))
print(confusion_matrix(y_pred_test, y_test))
print(classification_report(y_pred_test, y_test))


# In[13]:


# Predict decision_function with chosen model
lr_probs = GS.decision_function(X_test)


# In[14]:


# Calculate ROC AUC score
lr_auc = roc_auc_score(y_test, lr_probs)
print(lr_auc)


# In[15]:


# Calculate roc curves with the label of the positive class being "bad"
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs, pos_label = "bad")


# In[16]:


# Plot the roc curve for the model
pyplot.plot(lr_fpr, lr_tpr, label= "Logistic")
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.legend()
pyplot.show()

