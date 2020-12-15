#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt


# In[2]:


# Get target data file
train = pd.read_csv("D:\Documenten\CamCAN\memory_scores.csv")


# In[3]:


# Select the 10% rows with with lowest final memory scores. Duplicates are NOT dropped.
worst = train.nsmallest(63, "final_score", "all")


# In[4]:


# Select the 10% rows with with highest final memory scores. Duplicates are NOT dropped.
best = train.nlargest(63, "final_score", "all")


# In[5]:


# Stack the new data frames on top of each other
df = pd.concat([best, worst], axis=0)


# In[6]:


# Check the data
df


# In[7]:


# Get descriptives data
descriptives = pd.read_csv("D:\Documenten\CamCAN\Silvy_Collin_428\standard_data.csv")


# In[8]:


# Load participant IDs of people relevant for this study (participation in both experiments)
ID = np.load("CCID.npy", allow_pickle=True)


# In[9]:


# Remove irrelevant columns by specifying these column names
memory_data = df.drop(columns=["recognition", "naming", "occupation", "false_recognition", "true_recognition",
                                        "unfam", "Unnamed: 0", "final_score", "percent_recognition"])
descriptives = descriptives.drop(columns=["Hand", "Coil", "MT_TR"])


# In[10]:


# Merge data frames based on CCID 
dfs = descriptives, df
merged_data = reduce(lambda left, right: pd.merge(left, right, on = "CCID"), dfs)


# In[11]:


# Quick look at the size of the dataset
len(merged_data)


# In[12]:


# Quick look at the dataset
merged_data.head()


# In[13]:


# Write df to a csv file
merged_data.to_csv(r'D:\Documenten\CamCAN\subset_targetdata.csv', index = True)


# In[14]:


# Show class distribution 
b_scores = np.array(merged_data.columns[12:])
value_counts = []
for i in b_scores:
    print(merged_data[i].value_counts())
    value_counts.append(merged_data[i].value_counts())


# In[15]:


# Split data in "good" and "bad" group (subset)
df1 = merged_data[merged_data['final_score_binned'] == 'good']
df2 = merged_data[merged_data['final_score_binned'] == 'bad']


# In[20]:


# Print mean final score for bad and good group (subset)
print("Good group:", df1["final_score"].mean())
print("Bad group:", df2["final_score"].mean())

