#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
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
    z_inputs, y, test_size = 0.3, random_state = 7)


# In[7]:


# Prepare the cross-validation procedure
cv = KFold(n_splits = 10, random_state = 777, shuffle = True)


# In[8]:


# Define Multi-layer Perceptron model
model = MLPClassifier(random_state = 777, nesterovs_momentum = True, 
                      learning_rate = "adaptive", max_iter = 200, early_stopping = True)


# In[9]:


# Set the parameters 
parameters = {
            "hidden_layer_sizes": [(500)],
            "activation":["logistic"],
            "solver":["lbfgs"], 
        }


# In[10]:


# Implementing GridSearchCV 
GS = GridSearchCV(model, parameters, cv = cv, verbose = 1, n_jobs = -1)
GS.fit(X_train, y_train) # Fitting model on training data


# In[11]:


print(GS.best_estimator_) # Estimator that was chosen by the search (which gave highest score)
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
mlp_probs = GS.predict_proba(X_test)[:,1]


# In[14]:


# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
print(roc_auc_score(y_test, mlp_probs))


# In[15]:


# Compute Receiver operating characteristic (ROC)
fpr2, tpr2, threshold = roc_curve(y_test, GS.predict_proba(X_test)[:,1], pos_label = "bad")


# In[16]:


# Plot the roc curve for the model
pyplot.plot(fpr2, tpr2, label = "MLP")
pyplot.xlabel("False Positive Rate")
pyplot.ylabel("True Positive Rate")
pyplot.legend()
pyplot.show()

