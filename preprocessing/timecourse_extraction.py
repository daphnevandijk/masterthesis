#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy import stats


# In[2]:


# Create two empty lists
time_courses = []
CCID = []


# In[3]:


# Obtaining the paths to the required MATLAB-files
mat_files = glob.glob("D:\Documenten\CamCAN\cc700\**\Movie\ROI_epi.mat", recursive = True)


# In[4]:


# Obtain mean hippocampal time course from the individual MATLAB-files
for i, file in enumerate(mat_files):
    mat_contents = loadmat(file) # Load MATLAB file
    data = mat_contents["ROI"] # Get right data
    left_hemisphere = data["mean"][0][36] # Extract time course of the left hemisphere
    right_hemisphere = data["mean"][0][37] # Extract time course of the right hemisphere
    hippocampus = ((np.array(left_hemisphere) + np.array(right_hemisphere)) / 2.0) # Compute mean hippocampal time course
    time_courses.append(hippocampus.flatten()) # Add flat hippocampal time course to the empty list
    CCID.append(mat_files[i][27:35]) # Extract CCID number and add to empty list


# In[5]:


# Save arrays to a file in NumPy .npy format
np.save("time_courses.npy", time_courses)
np.save("CCID.npy", CCID)

