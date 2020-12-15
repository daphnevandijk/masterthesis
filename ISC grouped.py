#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
from isc_standalone import isc, isfc, squareform, compute_summary_statistic, bootstrap_isc, permutation_isc, timeshift_isc, phaseshift_isc, compute_summary_statistic


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


# Get target data
train = pd.read_csv("D:\Documenten\CamCAN\memory_scores.csv")


# In[5]:


# Create the input and target list
timecourses = [] # = hippocampal time courses
group_assignment = [] # = target labels

# Match labels and features
for i in range(0, len(train)):
    if train["CCID"][i] in id_course:
        data = id_course[train["CCID"][i]]
        timecourses.append(data)
        group_assignment.append(train["final_score_binned"][i])


# In[6]:


# Split data in "good" and "bad" group
group_bad = np.where(np.array(group_assignment) == "bad")
group_good = np.where(np.array(group_assignment) == "good")


# In[8]:


# Get list with time courses for the "bad" category
bad = []
for i in group_bad[0]:
    bad.append(timecourses[i])


# In[9]:


# Get list with time courses for the "good" category
good = []
for i in group_good[0]:
    good.append(timecourses[i])


# In[10]:


# Reverse the axes of the time courses data
bad_courses = np.transpose(bad)
good_courses = np.transpose(good)


# In[11]:


# Calculate median ISC for the "bad" group using pairwise approach
iscs_bad = isc(bad_courses, pairwise = True, summary_statistic = "median")


# In[12]:


# Calculate median ISC for the "good" group using pairwise approach
iscs_good = isc(good_courses, pairwise = True, summary_statistic = "median")


# In[13]:


# Get actual ISC values and the actual observed group difference
print(f"Actual observed group difference in ISC values "
      f"= {iscs_bad - iscs_good},"
      f"\n ISC of 'bad' group = {iscs_bad},"
      f"\n ISC of 'good' group = {iscs_good},")

