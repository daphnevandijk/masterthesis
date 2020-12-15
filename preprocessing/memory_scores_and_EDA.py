#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib2tikz


# In[2]:


# Load summary data of Famous Faces results
faces_summary = pd.read_csv(r"D:\Documenten\CamCAN\FamousFaces-20200918T083539Z-001\FacesTest_summary.csv")


# In[3]:


# Quick look at the data
faces_summary.head()


# In[4]:


# Create new data frame with the CCID numbers
memory_scores = pd.DataFrame(faces_summary["CCID"])


# In[5]:


# Calculate and define memory scores
memory_scores["recognition"] = faces_summary["FAMfam"] 
memory_scores["naming"] = faces_summary["FAMnam"]
memory_scores["occupation"] = faces_summary["FAMocc"] 
memory_scores["unfam"] = faces_summary["UNunfam"]
memory_scores["percent_recognition"] = ((faces_summary["FAMfam"] / 30)*100) 
memory_scores["false_recognition"] = (100 - ((faces_summary["UNunfam"] / 10)*100)) 
memory_scores["true_recognition"] = memory_scores["percent_recognition"] - memory_scores["false_recognition"]
memory_scores["final_score"] = memory_scores["naming"] + memory_scores["true_recognition"] + memory_scores["occupation"]


# In[6]:


memory_scores


# In[7]:


# Show distribution of the final memory scores and mean score
f = plt.figure()
plt.hist(memory_scores["final_score"], bins = 8)
plt.style.use("ggplot")
plt.axvline(memory_scores["final_score"].mean(), color= "black", linewidth = 2)
plt.title("distribution of the final memory scores")
plt.show()


# In[8]:


# Show distribution of the true recognition memory scores and mean score
f = plt.figure()
plt.hist(memory_scores["true_recognition"], bins = 8)
plt.style.use("ggplot")
plt.axvline(memory_scores["true_recognition"].mean(), color= "black", linewidth = 2)
plt.title("distribution of the final memory scores")
plt.show()


# In[9]:


# Show distribution of the naming scores and mean score
plt.hist(memory_scores["naming"], bins = 8)
plt.style.use("ggplot")
plt.axvline(memory_scores["naming"].mean(), color= "black", linewidth = 2)
plt.title("distribution of the naming scores")
plt.show()


# In[10]:


# Show distribution of the occupation scores and mean score
plt.hist(memory_scores["occupation"], bins = 8)
plt.style.use("ggplot")
plt.axvline(memory_scores["occupation"].mean(), color= "black", linewidth = 2)
plt.title("distribution of the occupation scores")
plt.show()


# In[11]:


# Print the mean, median, minimum and maximum score and the standard deviation
scores = ["naming", "occupation", "true_recognition", "final_score"]

for i in scores:
    print(i)
    print("mean =", np.mean(memory_scores[i]))
    print("median =", memory_scores.loc[:,i].median())
    print("SD =", np.std(memory_scores[i]))
    print("min_score =", np.min(memory_scores[i]))
    print("max_score =", np.max(memory_scores[i]))
    print("------")


# In[12]:


# Set threshold and bin values into discrete intervals
for i in scores:
    threshold = memory_scores.loc[:,i].median() - 0.001
    max_score = np.max(memory_scores[i])
    memory_scores[i + "_" + "binned"] = pd.cut(x = memory_scores[i], bins=[0, threshold, max_score], labels = ["bad", "good"])


# In[13]:


# Check the data frame
memory_scores


# In[14]:


# Remove rows with missing values
memory_scores = memory_scores.dropna()


# In[15]:


# View changes in data frame
memory_scores


# In[16]:


# Save data to csv file
memory_scores.to_csv("D:\Documenten\CamCAN\memory_scores.csv")


# In[17]:


# Show distribution of the binary final score variable
memory_scores["final_score_binned"].value_counts().plot(kind = "bar")
plt.title("distribution of the binary final score")
plt.xticks(rotation=0)
plt.ylabel("number of participants")


# In[18]:


# Show distribution of the binary naming score variable
memory_scores["naming_binned"].value_counts().plot(kind = "bar")
plt.title("distribution of the binary naming score")
plt.xticks(rotation=0)
plt.ylabel("number of participants")


# In[19]:


# Show distribution of the binary occupation score variable
memory_scores["occupation_binned"].value_counts().plot(kind = "bar")
plt.title("distribution of the binary occupation score")
plt.xticks(rotation=0)
plt.ylabel("number of participants")


# In[20]:


# Show distribution of the binary true recognition score 
memory_scores["true_recognition_binned"].value_counts().plot(kind = "bar")
plt.title("distribution of the binary unfam score")
plt.xticks(rotation=0)
plt.ylabel("number of participants")


# In[21]:


# df with only participants with "bad" final memory score
bad_memory = memory_scores[memory_scores["final_score_binned"] == "bad"]


# In[22]:


# df with only participants with "good" final memory score
good_memory = memory_scores[memory_scores["final_score_binned"] == "good"]


# In[23]:


# Statistics per target group for the final memory score
dfs = good_memory, bad_memory

for i in range (len(dfs)):
    print("N:", len(dfs[i]))
    print("Mean score:", np.mean(dfs[i]["final_score"]))
    print("SD:", np.std(dfs[i]["final_score"]))
    print("Min score:", np.min(dfs[i]["final_score"]))
    print("Max score:", np.max(dfs[i]["final_score"]))
    print("--------------")

