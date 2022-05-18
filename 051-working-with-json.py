#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>5.1. Working with JSON files</strong></font>

# In this project, we'll be looking at tracking corporate bankruptcies in Poland. To do that, we'll need to get data that's been stored in a `JSON` file, explore it, and turn it into a DataFrame that we'll use to train our model.

# In[1]:


import gzip
import json

import pandas as pd
import wqet_grader
from IPython.display import VimeoVideo

wqet_grader.init("Project 5 Assessment")


# In[2]:


VimeoVideo("694158732", h="73c2fb4e4f", width=600)


# # Prepare Data

# ## Open

# The first thing we need to do is access the file that contains the data we need. We've done this using multiple strategies before, but this time around, we're going to use the command line.

# In[3]:


VimeoVideo("693794546", h="6e1fab0a5e", width=600)


# **Task 5.1.1:** Open a terminal window and navigate to the directory where the data for this project is located.
# 
# - [What's the  Linux command line?](../%40textbook/18-linux-command-line.ipynb)
# - [Navigate a file system using the Linux command line.](../%40textbook/18-linux-command-line.ipynb) 

# As we've seen in our other projects, datasets can be large or small, messy or clean, and complex or easy to understand. Regardless of how the data looks, though, it needs to be saved in a file somewhere, and when that file gets too big, we need to *compress* it. Compressed files are easier to store because they take up less space. If you've ever come across a `ZIP` file, you've worked with compressed data. 
# 
# The file we're using for this project is compressed, so we'll need to use a file utility called `gzip` to open it up.

# In[4]:


VimeoVideo("693794604", h="a8c0f15712", width=600)


# **Task 5.1.2:** In the terminal window, locate the data file for this project and decompress it.
# 
# - [What's gzip?](../%40textbook/18-linux-command-line.ipynb#gzip)
# - [What's data compression?](../%40textbook/18-linux-command-line.ipynb#Data-Compressing)
# - [Decompress a file using gzip.](../%40textbook/18-linux-command-line.ipynb#gzip)

# In[5]:


VimeoVideo("693794641", h="d77bf46d41", width=600)


# In[3]:


get_ipython().run_cell_magic('bash', '', 'cd data\ngzip -dkf poland-bankruptcy-data-2009.json.gz')


# ## Explore

# Now that we've decompressed the data, let's take a look and see what's there.

# In[6]:


VimeoVideo("693794658", h="c8f1bba831", width=600)


# **Task 5.1.3:** In the terminal window, examine the first 10 lines of `poland-bankruptcy-data-2009.json`.
# 
# - [Print lines from a file in the Linux command line.](../%40textbook/18-linux-command-line.ipynb#Viewing-File-Contents)

# Does this look like any of the data structures we've seen in previous projects?

# In[7]:


VimeoVideo("693794680", h="7f1302444b", width=600)


# **Task 5.1.4:** Open `poland-bankruptcy-data-2009.json` by opening the `data` folder to the left and then double-clicking on the file. 👈 

# How is the data organized? 

# Curly brackets? Key-value pairs? It looks similar to a Python dictionary. It's important to note that JSON is not _exactly_ the same as a dictionary, but a lot of the same concepts apply. Let's try reading the file into a DataFrame and see what happens.

# In[8]:


VimeoVideo("693794696", h="dd5b5ad116", width=600)


# **Task 5.1.5:** Load the data into a DataFrame. 
# 
# - [Read a JSON file into a DataFrame using pandas.](../%40textbook/03-pandas-getting-started.ipynb#JSON-Files)

# In[ ]:


df = ...
df.head()


# In[9]:


VimeoVideo("693794711", h="fdb009c4eb", width=600)


# Hmmm. It looks like something went wrong, and we're going to have to fix it. Luckily for us, there's an error message to help us figure out what's happening here:
# 
# <code style="background-color:#FEDDDE;"><span style="color:#E45E5C">ValueError</span>: Mixing dicts with non-Series may lead to ambiguous ordering.
# </code>
# 
# What should we do? That error sounds serious, but the world is big, and we can't possibly be the first people to encounter this problem. When you come across an error, copy the message into a search engine and see what comes back. You'll get lots of results. The web has lots of places to look for solutions to problems like this one, and [Stack Overflow](https://stackoverflow.com/) is one of the best. [Click here to check out a possible solution to our problem.](https://stackoverflow.com/questions/57018859/valueerror-mixing-dicts-with-non-series-may-lead-to-ambiguous-ordering)
# 
# There are three things to look for when you're browsing through solutions on Stack Overflow. 
# 
# 1. **Context:** A good question is specific; if you click through that link, you'll see that the person asks a **specific** question, gives some relevant information about their OS and hardware, and then offers the code that threw the error. That's important, because we need...
# 2. **Reproducible Code:** A good question also includes enough information for you to reproduce the problem yourself. After all, the only way to make sure the solution actually applies to your situation is to see if the code in the question throws the error you're having trouble with! In this case, the person included not only the code they used to get the error, but the actual error message itself. That would be useful on its own, but since you're looking for an actual solution to your problem, you're really looking for...
# 3. **An answer:** Not every question on Stack Overflow gets answered. Luckily for us, the one we've been looking at did. There's a big green check mark next to the first solution, which means that the person who asked the question thought that solution was the best one.
# 
# Let's try it and see if it works for us too!

# In[10]:


VimeoVideo("693794734", h="fecea6a81e", width=600)


# **Task 5.1.6:** Using a context manager, open the file `poland-bankruptcy-data-2009.json` and load it as a dictionary with the variable name `poland_data`.
# 
# - [What's a context manager?](../%40textbook/02-python-advanced.ipynb#Create-files-using-Context-Manager)
# - [Open a file in Python.](../%40textbook/02-python-advanced.ipynb#Create-files-using-Context-Manager)
# - [Load a JSON file into a dictionary using Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-Dictionaries)

# In[4]:


# Open file and load JSON
with open("data/poland-bankruptcy-data-2009.json","r") as read_file:
    poland_data = json.load(read_file)
    
print(type(poland_data))


# Okay! Now that we've successfully opened up our dataset, let's take a look and see what's there, starting with the keys. Remember, the **keys** in a dictionary are categories of things in a dataset.

# In[11]:


VimeoVideo("693794754", h="18e70f4225", width=600)


# **Task 5.1.7:** Print the keys for `poland_data`.
# 
# - [List the keys of a dictionary in Python.](../%40textbook/01-python-getting-started.ipynb#Dictionary-Keys)

# In[ ]:


# Print `poland_data` keys


# `schema` tells us how the data is structured, `metadata` tells us where the data comes from, and `data` is the data itself.

# Now let's take a look at the values. Remember, the **values** in a dictionary are ways to describe the variable that belongs to a key.

# In[12]:


VimeoVideo("693794768", h="8e5b53b0ca", width=600)


# **Task 5.1.8:** Explore the values associated with the keys in `poland_data`. What do each of them represent? How is the information associated with the `"data"` key organized?

# In[6]:


# Continue Exploring `poland_data`
# poland_data.keys()
poland_data["data"][0]


# This dataset includes all the information we need to figure whether or not a Polish company went bankrupt in 2009. There's a bunch of features included in the dataset, each of which corresponds to some element of a company's balance sheet. You can explore the features by looking at the [data dictionary](./056-data-dictionary.ipynb). Most importantly, we also know whether or not the company went bankrupt. That's the last key-value pair.

# Now that we know what data we have for each company, let's take a look at how many companies there are.

# In[13]:


VimeoVideo("693794783", h="8d333027cc", width=600)


# **Task 5.1.9:** Calculate the number of companies included in the dataset.
# 
# - [Calculate the length of a list in Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-Lists)
# - [List the keys of a dictionary in Python.](../%40textbook/01-python-getting-started.ipynb#Dictionary-Keys)

# In[9]:


# Calculate number of companies
# type(poland_data)
len(poland_data["data"])


# And then let's see how many features were included for one of the companies.

# In[14]:


VimeoVideo("693794797", h="3c1eff82dc", width=600)


# **Task 5.1.10:** Calculate the number of features associated with `"company_1"`.

# In[10]:


# Calculate number of features
len(poland_data["data"][0])


# Since we're dealing with data stored in a JSON file, which is common for semi-structured data, we can't assume that all companies have the same features. So let's check!

# In[15]:


VimeoVideo("693794810", h="80e195944b", width=600)


# **Task 5.1.11:** Iterate through the companies in `poland_data["data"]` and check that they all have the same number of features.
# 
# - [What's an iterator?](../%40textbook/02-python-advanced.ipynb#Iterators-and-Iterables)
# - [Access the items in a dictionary in Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-Lists)
# - [Write a for loop in Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-for-Loops)

# In[11]:


# Iterate through companies
for item in poland_data["data"]:
    if len(item) != 66:
        print("Alert!!!")


# It looks like they do! 

# Let's put all this together. First, open up the compressed dataset and load it directly into a dictionary.

# In[16]:


VimeoVideo("693794824", h="dbfc9b43ee", width=600)


# **Task 5.1.12:** Using a context manager, open the file `poland-bankruptcy-data-2009.json.gz` and load it as a dictionary with the variable name `poland_data_gz`. 
# 
# - [What's a context manager?](../%40textbook/02-python-advanced.ipynb#Create-files-using-Context-Manager)
# - [Open a file in Python.](../%40textbook/02-python-advanced.ipynb#Create-files-using-Context-Manager)
# - [Load a JSON file into a dictionary using Python.](../%40textbook/01-python-getting-started.ipynb#Working-with-Dictionaries)

# In[13]:


# Open compressed file and load contents
with gzip.open("data/poland-bankruptcy-data-2009.json.gz","r") as read_file:
    poland_data_gz = json.load(read_file)
    
print(type(poland_data_gz))


# Since we now have two versions of the dataset — one compressed and one uncompressed — we need to compare them to make sure they're the same.

# In[17]:


VimeoVideo("693794837", h="925b5e4e5a", width=600)


# **Task 5.1.13:** Explore `poland_data_gz` to confirm that is contains the same data as `data`, in the same format. <span style="display: none">WorldQuant University Canary</span> 

# In[14]:


# Explore `poland_data_gz`
print(poland_data_gz.keys())
print(len(poland_data_gz["data"]))
print(len(poland_data_gz["data"][0]))


# Looks good! Now that we have an uncompressed dataset, we can turn it into a DataFrame using `pandas`.

# In[18]:


VimeoVideo("693794853", h="b74ef86783", width=600)


# **Task 5.1.14:** Create a DataFrame `df` that contains the all companies in the dataset, indexed by `"company_id"`. Remember the principles of *tidy data* that you learned in Project 1, and make sure your DataFrame has shape `(9977, 65)`. 
# 
# - [Create a DataFrame from a dictionary in pandas.](../%40textbook/03-pandas-getting-started.ipynb#Dictionaries)

# In[17]:


df = pd.DataFrame().from_dict(poland_data_gz["data"]).set_index("company_id")
print(df.shape)
df.head()


# ## Import

# Now that we have everything set up the way we need it to be, let's combine all these steps into a single function that will decompress the file, load it into a DataFrame, and return it to us as something we can use.

# In[20]:


VimeoVideo("693794879", h="f51a3a342f", width=600)


# **Task 5.1.15:** Create a `wrangle` function that takes the name of a compressed file as input and returns a tidy DataFrame. After you confirm that your function is working as intended, submit it to the grader. 

# In[21]:


def wrangle(filename):
    with gzip.open(filename,"r") as f:
        data = json.load(f)
    df = pd.DataFrame().from_dict(data["data"]).set_index("company_id")
    return df


# In[22]:


df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()


# In[23]:



wqet_grader.grade(
    "Project 5 Assessment",
    "Task 5.1.15",
    wrangle("data/poland-bankruptcy-data-2009.json.gz"),
)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
