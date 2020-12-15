#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Get merged data with the descriptives
df = pd.read_csv("D:\Documenten\CamCAN\merged_data.csv")


# In[3]:


# Check the data
df


# In[4]:


# Split data in "good" and "bad" group 
df3 = df[df['final_score_binned'] == 'good']
df4 = df[df['final_score_binned'] == 'bad']


# In[5]:


# Print mean final score for bad and good group (subset)
print("Good group:", df3["final_score"].mean())
print("Bad group:", df4["final_score"].mean())


# In[6]:


# Show the gender distribution of the study population
df["Sex"].value_counts().plot(kind = "bar")
plt.title("gender distribution")
plt.xticks(rotation=0)
plt.ylabel("number of participants")
plt.show()


# In[7]:


# Show distribution of age of participants and average age
plt.hist(df["Age"])
plt.style.use("ggplot")
plt.axvline(df["Age"].mean(), color= "black", linewidth = 2)
plt.show()


# In[8]:


# Print the descriptive statistics
print("Total number of participants:", df["Sex"].count())
print("Average age:", df["Age"].mean())
print("Standard deviation of age:", df["Age"].std())
print("Youngest age:", df["Age"].min())
print("Oldest age:",df["Age"].max())
print("Male / female distribution:", df['Sex'].value_counts())


# In[9]:


# Run same analysis for subset dataset
df2 = pd.read_csv("D:\Documenten\CamCAN\subset_targetdata.csv")


# In[10]:


# Show the gender distribution of the study population (subset data)
df2["Sex"].value_counts().plot(kind = "bar")
plt.title("gender distribution")
plt.xticks(rotation=0)
plt.ylabel("number of participants")
plt.show()


# In[11]:


# Show distribution of age of participants and average age (subset data)
plt.hist(df2["Age"])
plt.style.use("ggplot")
plt.axvline(df2["Age"].mean(), color= "black", linewidth = 2)
plt.show()


# In[12]:


# Print the descriptive statistics (subset data)
print("Total number of participants:", df2["Sex"].count())
print("Average age:", df2["Age"].mean())
print("Standard deviation of age:", df2["Age"].std())
print("Youngest age:", df2["Age"].min())
print("Oldest age:",df2["Age"].max())
print("Male / female distribution:", df2['Sex'].value_counts())

