#!/usr/bin/env python
# coding: utf-8

# <font size="+3"><strong>5.3. Ensemble Models: Random Forest</strong></font>

# So far in this project, we've learned how to retrieve and decompress data, and how to manage imbalanced data to build a decision-tree model.
# 
# In this lesson, we're going to expand our decision tree model into an entire forest (an example of something called an **ensemble model**); learn how to use a **grid search** to tune hyperparameters; and create a function that loads data and a pre-trained model, and uses that model to generate a Series of predictions.

# In[1]:


import gzip
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import wqet_grader
from imblearn.over_sampling import RandomOverSampler
from IPython.display import VimeoVideo
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

wqet_grader.init("Project 5 Assessment")


# In[2]:


VimeoVideo("694695674", h="538b4d2725", width=600)


# # Prepare Data

# As always, we'll begin by importing the dataset.

# ## Import

# **Task 5.3.1:** Complete the `wrangle` function below using the code you developed in the  lesson 5.1. Then use it to import `poland-bankruptcy-data-2009.json.gz` into the DataFrame `df`.
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

# **Task 5.3.2:** Create your feature matrix `X` and target vector `y`. Your target is `"bankrupt"`. 
# 
# - [What's a <span id='term'>feature matrix</span>?](../%40textbook/15-ml-regression.ipynb#Linear-Regression)
# - [What's a <span id='term'>target vector</span>?](../%40textbook/15-ml-regression.ipynb#Linear-Regression)
# - [<span id='technique'>Subset a DataFrame by selecting one or more columns in <span id='tool'>pandas</span></span>.](../%40textbook/04-pandas-advanced.ipynb#Subset-a-DataFrame-by-Selecting-One-or-More-Columns) 
# - [<span id='technique'>Select a Series from a DataFrame in <span id='tool'>pandas</span></span>.](../%40textbook/04-pandas-advanced.ipynb#Combine-multiple-categories-in-a-Series) 

# In[5]:


target = "bankrupt"
X = df.drop(columns=target)
y = df[target]

print("X shape:", X.shape)
print("y shape:", y.shape)


# Since we're not working with time series data, we're going to randomly divide our dataset into training and test sets — just like we did in project 4.

# **Task 5.3.3:** Divide your data (`X` and `y`) into training and test sets using a randomized train-test split. Your test set should be 20% of your total data. And don't forget to set a `random_state` for reproducibility. 
# 
# - [<span id='technique'>Perform a randomized train-test split using <span id='tool'>scikit-learn</span></span>.](../%40textbook/14-ml-classification.ipynb#Randomized-Train-Test-split)

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# You might have noticed that we didn't create a validation set, even though we're planning on tuning our model's hyperparameters in this lesson. That's because we're going to use cross-validation, which we'll talk about more later on.

# ## Resample

# In[ ]:


VimeoVideo("694695662", h="dc60d76861", width=600)


# **Task 5.3.4:** Create a new feature matrix `X_train_over` and target vector `y_train_over` by performing random over-sampling on the training data.
# 
# - [What is over-sampling?](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Over-sampling)
# - [Perform random over-sampling using imbalanced-learn.](../%40textbook/13-ml-data-pre-processing-and-production.ipynb#Over-sampling)

# In[7]:


over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
print("X_train_over shape:", X_train_over.shape)
X_train_over.head()


# # Build Model

# Now that we have our data set up the right way, we can build the model. 🏗

# ## Baseline

# **Task 5.3.5:** Calculate the baseline accuracy score for your model.
# 
# - [What's <span id='tool'>accuracy score</span>?](../%40textbook/14-ml-classification.ipynb#Calculating-Accuracy-Score)
# - [<span id='technique'>Aggregate data in a Series using `value_counts` in <span id='tool'>pandas</span></span>.](../%40textbook/04-pandas-advanced.ipynb#Working-with-value_counts-in-a-Series)

# In[8]:


acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 4))


# ## Iterate

# So far, we've built single models that predict a single outcome. That's definitely a useful way to predict the future, but what if the one model we built isn't the *right* one? If we could somehow use more than one model simultaneously, we'd have a more trustworthy prediction.
# 
# **Ensemble models** work by building multiple models on random subsets of the same data, and then comparing their predictions to make a final prediction. Since we used a decision tree in the last lesson, we're going to create an ensemble of trees here. This type of model is called a **random forest**.
# 
# We'll start by creating a pipeline to streamline our workflow.

# In[3]:


VimeoVideo("694695643", h="32c3d5b1ed", width=600)


# **Task 5.3.6:** Create a pipeline named `clf` (short for "classifier") that contains a `SimpleImputer` transformer and a `RandomForestClassifier` predictor.
# 
# - [What's an ensemble model?](../%40textbook/14-ml-classification.ipynb#Classification-with-Ensemble-Models)
# - [What's a random forest model?](../%40textbook/14-ml-classification.ipynb#Random-Forest)

# In[9]:


clf = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(random_state=42)
)
print(clf)


# By default, the number of trees in our forest (`n_estimators`) is set to 100. That means when we train this classifier, we'll be fitting 100 trees. While it will take longer to train, it will hopefully lead to better performance.
# 
# In order to get the best performance from our model, we need to tune its hyperparameter. But how can we do this if we haven't created a validation set? The answer is **cross-validation**. So, before we look at hyperparameters, let's see how cross-validation works with the classifier we just built.

# In[4]:


VimeoVideo("694695619", h="2c41dca371", width=600)


# **Task 5.3.7:** Perform cross-validation with your classifier, using the over-sampled training data. We want five folds, so set `cv` to 5. We also want to speed up training, to set `n_jobs` to -1.
# 
# - [What's cross-validation?](../%40textbook/14-ml-classification.ipynb#Cross-Validation)
# - [Perform k-fold cross-validation on a model in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Cross-Validation)

# In[10]:


cv_acc_scores = cross_val_score(clf, X_train_over, y_train_over, cv=5, n_jobs=-1)
print(cv_acc_scores)


# That took kind of a long time, but we just trained 500 random forest classifiers (100 jobs x 5 folds). No wonder it takes so long!
# 
# Pro tip: even though `cross_val_score` is useful for getting an idea of how cross-validation works, you'll rarely use it. Instead, most people include a `cv` argument when they do a hyperparameter search. 

# Now that we have an idea of how cross-validation works, let's tune our model. The first step is creating a range of hyperparameters that we want to evaluate. 

# In[5]:


VimeoVideo("694695593", h="5143f0b63f", width=600)


# **Task 5.3.8:** Create a dictionary with the range of hyperparameters that we want to evaluate for our classifier. 
# 
# 1. For the `SimpleImputer`, try both the `"mean"` and `"median"` strategies. 
# 2. For the `RandomForestClassifier`, try `max_depth` settings between 10 and 15, by steps of 10. 
# 3. Also for the `RandomForestClassifier`, try `n_estimators` settings between 25 and 100 by steps of 25.
# 
# - [What's a dictionary?](../%40textbook/01-python-getting-started.ipynb#Python-Dictionaries)
# - [What's a hyperparameter?](../%40textbook/16-ts-core.ipynb#Hyperparameters)
# - [Create a range in Python](../%40textbook/17-ts-models.ipynb#Hyperparameters)
# - [Define a hyperparameter grid for model tuning in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Hyperparameter-Tuning)

# In[12]:


params = {
    "simpleimputer__strategy": ["mean", "median"],
    "randomforestclassifier__n_estimators": range(25, 100, 25),
    "randomforestclassifier__max_depth": range(10,50,10)
}
params


# Now that we have our hyperparameter grid, let's incorporate it into a **grid search**.

# In[6]:


VimeoVideo("694695574", h="8588bf015f", width=600)


# **Task 5.3.9:** Create a `GridSearchCV` named `model` that includes your classifier and hyperparameter grid. Be sure to use the same arguments for `cv` and `n_jobs` that you used above, and set `verbose` to 1. 
# 
# - [What's cross-validation?](../%40textbook/14-ml-classification.ipynb#Cross-Validation)
# - [What's a grid search?](../%40textbook/14-ml-classification.ipynb#Grid-Search)
# - [Perform a hyperparameter grid search in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Grid-Search)

# In[13]:


model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)
model


# Finally, now let's fit the model.

# In[7]:


VimeoVideo("694695566", h="f4e9910a9e", width=600)


# **Task 5.3.10:** Fit `model` to the over-sampled training data.

# In[14]:


# Train model
model.fit(X_train_over, y_train_over)


# This will take some time to train, so let's take a moment to think about why. How many forests did we just test? 4 different `max_depth`s times 3 `n_estimator`s times 2 imputation strategies... that makes 24 forests. How many fits did we just do? 24 forests times 5 folds is 120. And remember that each forest is comprised of 25-75 trees, which works out to *at least* 3,000 trees. So it's computationally expensive! 
# 
# Okay, now that we've tested all those models, let's take a look at the results.

# In[8]:


VimeoVideo("694695546", h="4ae60129c4", width=600)


# **Task 5.3.11:** Extract the cross-validation results from `model` and load them into a DataFrame named `cv_results`.
# 
# - [Get cross-validation results from a hyperparameter search in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Grid-Search)

# In[15]:


cv_results = pd.DataFrame(model.cv_results_)
cv_results.head(10)


# In addition to the accuracy scores for all the different models we tried during our grid search, we can see how long it took each model to train. Let's take a closer look at how different hyperparameter settings affect training time. 
# 
# First, we'll look at `n_estimators`. Our grid search evaluated this hyperparameter for various `max_depth` settings, but let's only look at models where `max_depth` equals 10.

# In[9]:


VimeoVideo("694695537", h="e460435664", width=600)


# **Task 5.3.12:**DataFrameate a mask for `cv_results` for rows where `"param_randomforestclassifier__max_depth"` equals 10. Then plot `"param_randomforestclassifier__n_estimators"` on the x-axis and `"mean_fit_time"` on the y-axis. Don't forget to label your axes and include a title. 
# 
# - [Subset a DataFrame with a mask using pandas.](../%40textbook/04-pandas-advanced.ipynb)
# - [Create a line plot in Matplotlib.](../%40textbook/06-visualization-matplotlib.ipynb)

# In[16]:


# Create mask
mask = cv_results["param_randomforestclassifier__max_depth"] == 10
# Plot fit time vs n_estimators
plt.plot(cv_results[mask]["param_randomforestclassifier__n_estimators"],
         cv_results[mask]["mean_fit_time"]
        )
# Label axes
plt.xlabel("Number of Estimators")
plt.ylabel("Mean Fit Time [seconds]")
plt.title("Training Time vs Estimators (max_depth=10)");


# Next, we'll look at `max_depth`. Here, we'll also limit our data to rows where `n_estimators` equals 25.

# In[10]:


VimeoVideo("694695525", h="99f2dfc9eb", width=600)


# **Task 5.3.13:** Create a mask for `cv_results` for rows where `"param_randomforestclassifier__n_estimators"` equals 25. Then plot `"param_randomforestclassifier__max_depth"` on the x-axis and `"mean_fit_time"` on the y-axis. Don't forget to label your axes and include a title. 
# 
# - [Subset a DataFrame with a mask using pandas.](../%40textbook/04-pandas-advanced.ipynb)
# - [Create a line plot in Matplotlib.](../%40textbook/06-visualization-matplotlib.ipynb)

# In[17]:


# Create mask
mask = cv_results["param_randomforestclassifier__n_estimators"] == 25
# Plot fit time vs max_depth
plt.plot(cv_results[mask]["param_randomforestclassifier__max_depth"],
         cv_results[mask]["mean_fit_time"]
        )
# Label axes
plt.xlabel("Max Depth")
plt.ylabel("Mean Fit Time [seconds]")
plt.title("Training Time vs Max Depth (n_estimators=25)");


# In[18]:


cv_results[mask][["mean_fit_time", "param_randomforestclassifier__max_depth", "param_simpleimputer__strategy"]]


# There's a general upwards trend, but we see a lot of up-and-down here. That's because for each max depth, grid search tries two different imputation strategies: mean and median. Median is a lot faster to calculate, so that speeds up training time. 
# 
# Finally, let's look at the hyperparameters that led to the best performance. 

# In[11]:


VimeoVideo("694695505", h="f98f660ce1", width=600)


# **Task 5.3.14:** Extract the best hyperparameters from `model`.
# 
# - [Get settings from a hyperparameter search in scikit-learn.](../%40textbook/14-ml-classification.ipynb#Cross-Validation)

# In[19]:


# Extract best hyperparameters
model.best_params_


# In[20]:


model.best_estimator_


# Note that we don't need to build and train a new model with these settings. Now that the grid search is complete, when we use `model.predict()`, it will serve up predictions using the best model — something that we'll do at the end of this lesson.

# ## Evaluate

# All right: The moment of truth. Let's see how our model performs.

# **Task 5.3.15:** Calculate the training and test accuracy scores for `model`. 
# 
# - [<span id='technique'>Calculate the accuracy score for a model in <span id='term'>scikit-learn</span></span>.](../%40textbook/14-ml-classification.ipynb#Calculating-Accuracy-Score)

# In[23]:


acc_train = model.score(X_train_over, y_train_over)
acc_test = model.score(X_test, y_test)

print("Training Accuracy:", round(acc_train, 4))
print("Test Accuracy:", round(acc_test, 4))


# We beat the baseline! Just barely, but we beat it. 

# Next, we're going to use a confusion matrix to see how our model performs. To better understand the values we'll see in the matrix, let's first count how many observations in our test set belong to the positive and negative classes. 

# In[26]:


y_test.value_counts()


# In[12]:


VimeoVideo("694695486", h="1d6ac2bf77", width=600)


# **Task 5.3.16:** Plot a confusion matrix that shows how your best model performs on your test set. 
# 
# - [What's a confusion matrix?](../%40textbook/14-ml-classification.ipynb#Confusion-Matrix)
# - [Create a confusion matrix using scikit-learn.](../%40textbook/14-ml-classification.ipynb#Confusion-Matrix) 

# In[24]:


# Plot confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test);


# Notice the relationship between the numbers in this matrix with the count you did the previous task. If you sum the values in the bottom row, you get the total number of positive observations in `y_train` ($72 + 11 = 83$). And the top row sum to the number of negative observations ($1902 + 11 = 1913$).

# # Communicate

# In[13]:


VimeoVideo("698358615", h="3fd4b2186a", width=600)


# **Task 5.3.17:** Create a horizontal bar chart with the 10 most important features for your model. 

# In[29]:


# Get feature names from training data
features = X_train_over.columns
# Extract importances from model
importances = model.best_estimator_.named_steps["randomforestclassifier"].feature_importances_
# Create a series with feature names and importances
feat_imp = pd.Series(importances, index=features).sort_values()
# Plot 10 most important features
feat_imp.tail(10).plot(kind="barh")
plt.xlabel("Gini Importance")
plt.ylabel("Feature")
plt.title("Feature Importance");


# The only thing left now is to save your model so that it can be reused.

# In[14]:


VimeoVideo("694695478", h="a13bdacb55", width=600)


# **Task 5.3.18:** Using a context manager, save your best-performing model to a a file named `"model-5-3.pkl"`. 
# 
# - [What's serialization?](../%40textbook/03-pandas-getting-started.ipynb#Pickle-Files)
# - [Store a Python object as a serialized file using pickle.](../%40textbook/03-pandas-getting-started.ipynb#Pickle-Files) 

# In[30]:


# Save model
with open("model-5-3.pkl", "wb") as f:
    pickle.dump(model, f)


# In[15]:


VimeoVideo("694695451", h="fc96dd8d1f", width=600)


# **Task 5.3.19:** Create a function `make_predictions`. It should take two arguments: the path of a JSON file that contains test data and the path of a serialized model. The function should load and clean the data using the `wrangle` function you created, load the model, generate an array of predictions, and convert that array into a Series. (The Series should have the name `"bankrupt"` and the same index labels as the test data.) Finally, the function should return its predictions as a Series. 
# 
# - [What's a function?](../%40textbook/02-python-advanced.ipynb#Functions)
# - [Load a serialized file](../%40textbook/03-pandas-getting-started.ipynb#Pickle-Files)
# - [What's a Series?](../%40textbook/05-pandas-summary-statistics.ipynb#Series)
# - [Create a Series in pandas](../%40textbook/03-pandas-getting-started.ipynb#Working-with-Columns)

# In[31]:


def make_predictions(data_filepath, model_filepath):
    # Wrangle JSON file
    X_test = wrangle(data_filepath)
    # Load model
    with open(model_filepath, "rb") as f:
        model = pickle.load(f)
    # Generate predictions
    y_test_pred = model.predict(X_test)
    # Put predictions into Series with name "bankrupt", and same index as X_test
    y_test_pred = pd.Series(y_test_pred, index=X_test.index, name="bankrupt")
    return y_test_pred


# In[16]:


VimeoVideo("694695426", h="f75588d43a", width=600)


# **Task 5.3.20:** Use the code below to check your `make_predictions` function. Once you're satisfied with the result, submit it to the grader. 

# In[32]:


y_test_pred = make_predictions(
    data_filepath="data/poland-bankruptcy-data-2009-mvp-features.json.gz",
    model_filepath="model-5-3.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()


# In[33]:


wqet_grader.grade(
    "Project 5 Assessment",
    "Task 5.3.19",
    make_predictions(
        data_filepath="data/poland-bankruptcy-data-2009-mvp-features.json.gz",
        model_filepath="model-5-3.pkl",
    ),
)


# ---
# Copyright © 2022 WorldQuant University. This
# content is licensed solely for personal use. Redistribution or
# publication of this material is strictly prohibited.
# 
