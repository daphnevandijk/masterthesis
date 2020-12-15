#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
from functools import reduce
import numpy as np


# In[2]:


# Read CSV files into data frame
memory_data = pd.read_csv("D:\Documenten\CamCAN\memory_scores.csv")
descriptives = pd.read_csv("D:\Documenten\CamCAN\Silvy_Collin_428\standard_data.csv")


# In[3]:


# Load participant IDs of people relevant for this study (participation in both experiments)
ID = np.load("CCID.npy", allow_pickle=True)


# In[4]:


# Quick look at the data
memory_data.head()


# In[5]:


# Do some EDA
EDA_merged = memory_data[memory_data["CCID"].isin(ID)]
scores = ["naming", "occupation", "true_recognition", "final_score"]

for i in scores:
    print(i)
    print("mean =", np.mean(EDA_merged[i]))
    print("median =", EDA_merged.loc[:,i].median())
    print("SD =", np.std(EDA_merged[i]))
    print("min_score =", np.min(EDA_merged[i]))
    print("max_score =", np.max(EDA_merged[i]))
    print("------")


# In[6]:


# Remove irrelevant columns by specifying these column names
memory_data = memory_data.drop(columns=["recognition", "naming", "occupation", 
                                        "unfam", "Unnamed: 0"])
descriptives = descriptives.drop(columns=["Hand", "Coil", "MT_TR"])


# In[7]:


# Merge data frames based on CCID 
dfs = descriptives, memory_data
merged_data = reduce(lambda left, right: pd.merge(left, right, on = "CCID"), dfs)


# In[8]:


# Quick look at the data
merged_data.head()


# In[9]:


# Delete the observations that are not relevant to this study
final_merged = merged_data[merged_data["CCID"].isin(ID)]


# In[10]:


# Check the number of observations
len(final_merged)


# In[11]:


# Write object to a csv file
final_merged.to_csv("D:\Documenten\CamCAN\merged_data.csv", index = False)


# In[12]:


# Show class distribution 
b_scores = np.array(final_merged.columns[7:])
for i in b_scores:
    print(final_merged[i].value_counts())

