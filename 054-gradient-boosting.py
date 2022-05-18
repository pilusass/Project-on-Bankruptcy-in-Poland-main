#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>5.4. Gradient Boosting Trees</strong></font>

# You've been working hard, and now you have all the tools you need to build and tune  models. We'll start this lesson the same way we've started the others: preparing the data and building our model, and this time with a new ensemble model. Once it's working, we'll learn some new performance metrics to evaluate it. By the end of this lesson, you'll have written your first Python module!  

# In[1]:


import gzip
import json
import pickle

import ipywidgets as widgets
import pandas as pd
import wqet_grader
from imblearn.over_sampling import RandomOverSampler
from IPython.display import VimeoVideo
from ipywidgets import interact
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from teaching_tools.widgets import ConfusionMatrixWidget

wqet_grader.init("Project 5 Assessment")


# In[2]:


VimeoVideo("696221191", h="275ffd1421", width=600)


# # Prepare Data

# All the data preparation for this module is the same as it was last time around. See you on the other side!

# ## Import

# **Task 5.4.1:** Complete the `wrangle` function below using the code you developed in the  lesson 5.1. Then use it to import `poland-bankruptcy-data-2009.json.gz` into the DataFrame `df`.
# 
# - [<span id='technique'>Write a function in <span id='tool'>Python</span></span>.](../%40textbook/02-python-advanced.ipynb#Functions)

# In[2]:


def wrangle(filename):
    with gzip.open(filename, "r") as f:
        data = json.load(f)
    df = pd.DataFrame().from_dict(data["data"]).set_index("company_id")
    return df


# In[3]:


df = wrangle("data/poland-bankruptcy-data-2009.json.gz")
print(df.shape)
df.head()


# ## Split

# **Task 5.4.2:** Create your feature matrix `X` and target vector `y`. Your target is `"bankrupt"`. 
# 
# - [What's a <span id='term'>feature matrix</span>?](../%40textbook/15-ml-regression.ipynb#Linear-Regression)
# - [What's a <span id='term'>target vector</span>?](../%40textbook/15-ml-regression.ipynb#Linear-Regression)
# - [<span id='technique'>Subset a DataFrame by selecting one or more columns in <span id='tool'>pandas</span></span>.](../%40textbook/04-pandas-advanced.ipynb#Subset-a-DataFrame-by-Selecting-One-or-More-Columns) 
# - [<span id='technique'>Select a Series from a DataFrame in <span id='tool'>pandas</span></span>.](../%40textbook/04-pandas-advanced.ipynb#Combine-multiple-categories-in-a-Series) 

# In[4]:


target = "bankrupt"
X = df.drop(columns=target)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


# **Task 5.4.3:** Divide your data (`X` and `y`) into training and test sets using a randomized train-test split. Your test set should be 20% of your total data. And don't forget to set a `random_state` for reproducibility. 
# 
# - [<span id='technique'>Perform a randomized train-test split using <span id='tool'>scikit-learn</span></span>.](../%40textbook/14-ml-classification.ipynb#Randomized-Train-Test-split)

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ## Resample

# **Task 5.4.4:** Create a new feature matrix `X_train_over` and target vector `y_train_over` by performing random over-sampling on the training data.
# 
# - [What is over-sampling?](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Over-sampling)
# - [Perform random over-sampling using imbalanced-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Over-sampling)

# In[6]:


over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()


# # Build Model

# Now let's put together our model. We'll start by calculating the baseline accuracy, just like we did last time.

# ## Baseline

# **Task 5.4.5:** Calculate the baseline accuracy score for your model.
# 
# - [What's <span id='tool'>accuracy score</span>?](../%40textbook/14-ml-classification.ipynb#Calculating-Accuracy-Score)
# - [<span id='technique'>Aggregate data in a Series using `value_counts` in <span id='tool'>pandas</span></span>.](../%40textbook/04-pandas-advanced.ipynb#Working-with-value_counts-in-a-Series)

# In[7]:


acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))


# ## Iterate

# Even though the building blocks are the same, here's where we start working with something new. First, we're going to use a new type of ensemble model for our classifier. 

# In[3]:


VimeoVideo("696221115", h="44fe95d5d9", width=600)


# **Task 5.4.6:** Create a pipeline named `clf` (short for "classifier") that contains a `SimpleImputer` transformer and a `GradientBoostingClassifier` predictor.
# 
# - [What's an ensemble model?](../%40textbook/14-ml-classification.ipynb#Classification-with-Ensemble-Models)
# - What's a gradient boosting model?

# In[9]:


clf = make_pipeline(
    SimpleImputer(),
    GradientBoostingClassifier()
)
clf


# Remember while we're doing this that we only want to be looking at the *positive* class. Here, the positive class is the one where the companies really did go bankrupt. In the dictionary we made last time, the positive class is made up of the companies with the `bankrupt: true` key-value pair.
# 
# Next, we're going to tune some of the hyperparameters for our model.

# In[4]:


VimeoVideo("696221055", h="b675d7fec0", width=600)


# **Task 5.4.7:** Create a dictionary with the range of hyperparameters that we want to evaluate for our classifier. 
# 
# 1. For the `SimpleImputer`, try both the `"mean"` and `"median"` strategies. 
# 2. For the `GradientBoostingClassifier`, try `max_depth` settings between 2 and 5. 
# 3. Also for the `GradientBoostingClassifier`, try `n_estimators` settings between 20 and 31, by steps of 5.
# 
# - [What's a dictionary?](../%40textbook/01-python-getting-started.ipynb#Python-Dictionaries)
# - [What's a hyperparameter?](../%40textbook/16-ts-core.ipynb#Hyperparameters)
# - [Create a range in Python.](../%40textbook/17-ts-models.ipynb#Hyperparameters)
# - [Define a hyperparameter grid for model tuning in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Hyperparameter-Tuning)

# In[10]:


params = {
    "simpleimputer__strategy": ["mean", "median"],
    "gradientboostingclassifier__n_estimators": range(20, 31, 5),
    "gradientboostingclassifier__max_depth": range(2,5)
}
params


# Note that we're trying much smaller numbers of `n_estimators`. This is because `GradientBoostingClassifier` is slower to train than the `RandomForestClassifier`. You can try increasing the number of estimators to see if model performance improves, but keep in mind that you could be waiting a long time!

# In[5]:


VimeoVideo("696221023", h="218915d38e", width=600)


# **Task 5.4.8:** Create a `GridSearchCV` named `model` that includes your classifier and hyperparameter grid. Be sure to use the same arguments for `cv` and `n_jobs` that you used above, and set `verbose` to 1. 
# 
# - [What's cross-validation?](../%40textbook/14-ml-classification.ipynb#Cross-Validation)
# - [What's a grid search?](../%40textbook/14-ml-classification.ipynb#Grid-Search)
# - [Perform a hyperparameter grid search in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Grid-Search)

# In[12]:


model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)


# Now that we have everything we need for the model, let's fit it to the data and see what we've got.

# In[6]:


VimeoVideo("696220978", h="008d915f33", width=600)


# **Task 5.4.9:** Fit your `model` to the over-sampled training data.

# In[13]:


# Fit model to over-sampled training data
model.fit(X_train_over, y_train_over)


# This will take longer than our last grid search, so now's a good time to get coffee or cook dinner. 🍲
# 
# Okay! Let's take a look at the results!

# In[7]:


VimeoVideo("696220937", h="9148032400", width=600)


# **Task 5.4.10:** Extract the cross-validation results from `model` and load them into a DataFrame named `cv_results`.
# 
# - Get cross-validation results from a hyperparameter search in scikit-learn.

# In[14]:


results = pd.DataFrame(model.cv_results_)
results.sort_values("rank_test_score").head(10)


# There are quite a few hyperparameters there, so let's pull out the ones that work best for our model.

# In[8]:


VimeoVideo("696220899", h="342d55e7d7", width=600)


# **Task 5.4.11:** Extract the best hyperparameters from `model`.
# 
# - [Get settings from a hyperparameter search in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Grid-Search)

# In[15]:


# Extract best hyperparameters
model.best_params_


# ## Evaluate

# Now that we have a working model that's actually giving us something useful, let's see how good it really is.

# **Task 5.4.12:** Calculate the training and test accuracy scores for `model`. 
# 
# - [<span id='technique'>Calculate the accuracy score for a model in <span id='term'>scikit-learn</span></span>.](../%40textbook/14-ml-classification.ipynb#Calculating-Accuracy-Score)

# In[16]:


acc_train = model.score(X_train_over, y_train_over)
acc_test = model.score(X_test, y_test)

print("Training Accuracy:", round(acc_train, 4))
print("Validation Accuracy:", round(acc_test, 4))


# Just like before, let's make a confusion matrix to see how our model is making its correct and incorrect predictions. 

# **Task 5.4.13:** Plot a confusion matrix that shows how your best model performs on your test set. 
# 
# - [What's a confusion matrix?](../%40textbook/14-ml-classification.ipynb#Confusion-Matrix)
# - [Create a confusion matrix using scikit-learn.](../%40textbook/14-ml-classification.ipynb#Confusion-Matrix) 

# In[17]:


# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


# This matrix is a great reminder of how imbalanced our data is, and of why accuracy isn't always the best metric for judging whether or not a model is giving us what we want. After all, if 95% of the companies in our dataset didn't go bankrupt, all the model has to do is always predict `{"bankrupt": False}`, and it'll be right 95% of the time. The accuracy score will be amazing, but it won't tell us what we really need to know.
# 
# Instead, we can evaluate our model using two new metrics: **precision** and **recall**.  The precision score is important when we want our model to only predict that a company will go bankrupt if its very confident in its prediction. The *recall* score is important if we want to make sure to identify all the companies that will go bankrupt, even if that means being incorrect sometimes. 
# 
# Let's start with a report you can create with scikit-learn to calculate both metrics. Then we'll look at them one-by-one using a visualization tool we've built especially for the Data Science Lab.

# In[9]:


VimeoVideo("696297886", h="fac5454b22", width=600)


# **Task 5.4.14:** Print the classification report for your model, using the test set.
# 
# - [Generate a classification report with scikit-learn.](../%40textbook/14-ml-classification.ipynb#Classification-Report)

# In[18]:


# Print classification report
print(classification_report(y_test, model.predict(X_test)))


# In[10]:


VimeoVideo("696220837", h="f93be5aba0", width=600)


# In[11]:


VimeoVideo("696220785", h="8a4c4bff58", width=600)


# **Task 5.4.15:** Run the cell below to load the confusion matrix widget.
# 
# - [What's precision?](../%40textbook/14-ml-classification.ipynb#Precision-Score)
# - [What's recall?](../%40textbook/14-ml-classification.ipynb#Recall-Score)

# In[19]:


model.predict(X_test)[:5]


# In[21]:


model.predict_proba(X_test)[:5,-1]


# In[22]:


c = ConfusionMatrixWidget(model, X_test, y_test)
c.show()


# If you move the probability threshold, you can see that there's a tradeoff between precision and recall. That is, as one gets better, the other suffers. As a data scientist, you'll often need to decide whether you want a model with better precision or better recall. What you choose will depend on how to intend to use your model.
# 
# Let's look at two examples, one where recall is the priority and one where precision is more important. First, let's say you work for a regulatory agency in the European Union that assists companies and investors navigate [insolvency proceedings](https://en.wikipedia.org/wiki/Insolvency_Regulation). You want to build a model to predict which companies could go bankrupt so that you can send debtors information about filing for legal protection before their company becomes insolvent. The administrative costs of sending information to a company is €500. The legal costs to the European court system if a company doesn't file for protection before bankruptcy is €50,000.
# 
# For a model like this, we want to focus on **recall**, because recall is all about *quantity*. A model that prioritizes recall will cast the widest possible net, which is the way to approach this problem. We want to send information to as many potentially-bankrupt companies as possible, because it costs a lot less to send information to a company that might not become insolvent than it does to skip a company that does. 

# In[12]:


VimeoVideo("696209314", h="36a14b503c", width=600)


# **Task 5.4.16:** Run the cell below, and use the slider to change the probability threshold of your model. What relationship do you see between changes in the threshold and changes in wasted administrative and legal costs? In your opinion, which is more important for this model: high precision or high recall?
# 
# - [What's precision?](../%40textbook/14-ml-classification.ipynb#Precision-Score)
# - [What's recall?](../%40textbook/14-ml-classification.ipynb#Recall-Score)

# In[23]:


c.show_eu()


# For the second example, let's say we work at a private equity firm that purchases distressed businesses, improve them, and then sells them for a profit. You want to build a model to predict which companies will go bankrupt so that you can purchase them ahead of your competitors. If the firm purchases a company that is indeed insolvent, it can make a profit of €100 million or more. But if it purchases a company that isn't insolvent and can't be resold at a profit, the firm will lose €250 million. 
# 
# For a model like this, we want to focus on **precision**. If we're trying to maximize our profit, the *quality* of our predictions is much more important than the *quantity* of our predictions. It's not a big deal if we don't catch every single insolvent company, but it's *definitely* a big deal if the companies we catch don't end up becoming insolvent.
# 
# This time we're going to build the visualization together. 

# In[13]:


VimeoVideo("696209348", h="f7e1981c9f", width=600)


# **Task 5.4.17:** Create an interactive dashboard that shows how company profit and losses change in relationship to your model's probability threshold. Start with the `make_cnf_matrix` function, which should calculate and print profit/losses, and display a confusion matrix. Then create a FloatSlider `thresh_widget` that ranges from 0 to 1. Finally combine your function and slider in the `interact` function.
# 
# - [What's a function?](../%40textbook/02-python-advanced.ipynb#Functions)
# - [What's a confusion matrix?](../%40textbook/14-ml-classification.ipynb#Confusion-Matrix)
# - [Create a confusion matrix using scikit-learn.](../%40textbook/14-ml-classification.ipynb#Confusion-Matrix) 

# In[31]:


threshold = 0.2
y_pred_prob = model.predict_proba(X_test)[:,-1]
y_pred = y_pred_prob > threshold
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print(f"Profit: €{tp * 100_000_000}")
print(f"Losses: €{fp * 250_000_000}")
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False);


# In[32]:


def make_cnf_matrix(threshold):
    y_pred_prob = model.predict_proba(X_test)[:,-1]
    y_pred = y_pred_prob > threshold
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"Profit: €{tp * 100_000_000}")
    print(f"Losses: €{fp * 250_000_000}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=False)
    
thresh_widget = widgets.FloatSlider(min=0, max=1, value=0.5, step=0.05)

interact(make_cnf_matrix, threshold=thresh_widget);


# <div class="alert alert-info" role="alert">
#     <b>Go Further:</b>💡 Some students have suggested that this widget would be better if it showed the sum of profits and losses. Can you add that total?
# </div>

# # Communicate

# Almost there! Save the best model so we can share it with other people, then put it all together with what we learned in the last lesson.

# **Task 5.4.18:** Using a context manager, save your best-performing model to a file named `"model-5-4.pkl"`. 
# 
# - [What's serialization?](../%40textbook/03-pandas-getting-started.ipynb#Pickle-Files)
# - [Store a Python object as a serialized file using pickle.](../%40textbook/03-pandas-getting-started.ipynb#Pickle-Files) 

# In[33]:


# Save model
with open("model-5-4.pkl", "wb") as f:
    pickle.dump(model, f)


# In[14]:


VimeoVideo("696220731", h="8086ff0bcd", width=600)


# **Task 5.4.19:** Open the file `my_predictor_lesson.py`, add the `wrangle` and `make_predictions` functions from the last lesson, and add all the necessary import statements to the top of the file. Once you're done, save the file. You can check that the contents are correct by running the cell below. 
# 
# - [What's a function?](../%40textbook/02-python-advanced.ipynb#Functions)

# In[34]:


get_ipython().run_cell_magic('bash', '', '\ncat my_predictor_lesson.py')


# Congratulations: You've created your first module!

# In[15]:


VimeoVideo("696220643", h="8a3f141262", width=600)


# **Task 5.4.20:** Import your `make_predictions` function from your `my_predictor` module, and use the code below to make sure it works as expected. Once you're satisfied, submit it to the grader.  

# In[35]:


# Import your module
from my_predictor_lesson import make_predictions

# Generate predictions
y_test_pred = make_predictions(
    data_filepath="data/poland-bankruptcy-data-2009-mvp-features.json.gz",
    model_filepath="model-5-4.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()


# In[36]:


wqet_grader.grade(
    "Project 5 Assessment",
    "Task 5.4.20",
    make_predictions(
        data_filepath="data/poland-bankruptcy-data-2009-mvp-features.json.gz",
        model_filepath="model-5-4.pkl",
    ),
)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
