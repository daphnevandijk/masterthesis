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


# Reverse the axes of the time courses data
data = np.transpose(timecourses)


# In[7]:


# Check shape of the data
np.shape(data)


# In[8]:


# Compute ISC summary statistic "median" using pairwise approach
iscs_median = isc(data, pairwise=True, summary_statistic="median")
print(iscs_median)


# In[9]:


# Compute ISCs (pairwise approach) and then run two-sample permutation test on ISCs
iscs = isc(data, pairwise=True, summary_statistic=None)
observed, p, distribution = permutation_isc(iscs,
                                            group_assignment=group_assignment,
                                            pairwise = True,
                                            summary_statistic = 'median',
                                            n_permutations = 1000,
                                           )


# In[10]:


# Show number of computed correlation values (ISCs)
len(iscs)


# In[11]:


# Inspect shape of null distribution
print(f"Null distribution shape = {distribution.shape}"
      f"\ni.e., {distribution.shape[0]} permutations "
      f"and {distribution.shape[1]} voxels")

# Get actual ISC value and p-value for first voxel
print(f"Actual observed group difference in ISC values "
      f"= {observed[0]:.3f},"
      f"\np-value from permutation test = {p[0]:.3f}")

