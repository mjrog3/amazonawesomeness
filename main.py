# **Imports**

# Data

import pandas as pd
import os

current_folder = os.getcwd()
CDs_folder = 'CDs_and_Vinyl'

# Open and load json training files
x = pd.read_json(os.path.join(current_folder, CDs_folder, 'train', 'review_training.json'))
y = pd.read_json(os.path.join(current_folder, CDs_folder, 'train', 'product_training.json'))

# Other imports  
import numpy as np
from nltk import sentiment
from sklearn.model_selection import cross_validate
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Patch to speed up sklearn (ONLY IF YOU HAVE AN INTEL CHIP)   
from sklearnex import patch_sklearn 
patch_sklearn()

# ## Pre Processing

# Review Count
dropUnverified = x[x.verified == True]
reviewCount = x.groupby('asin')["reviewerID"].count()
reviewCount = reviewCount.rename("Review_Count")


# Percent verified
percent_verified = dropUnverified.groupby("asin")["reviewerID"].count()
percent_verified = percent_verified/reviewCount
percent_verified = percent_verified.apply(lambda x: x if x > 0  else 0)
percent_verified = percent_verified.rename("%_Verified")

# Total number of votes across reviews
x_copy = x.copy(deep=True)
total_review_num = x_copy.vote
total_review_num = total_review_num.apply(lambda x: float(x.replace(",", "")) if type(x) == str  else 0)
total_review_num = total_review_num.rename("vote").to_frame()
x_copy["vote"] = total_review_num["vote"]
total_votes = x_copy.groupby('asin')["vote"].sum("vote")
total_votes = total_votes.rename("Total_Votes")

# Text Analysis

# Length of Reviews Feature
reviewlength = x.groupby('asin')["reviewText"].apply(lambda x: x.str.split().str.len().mean())
reviewlength = reviewlength.fillna(0).rename("Review_Length")

summarylength = x.groupby('asin')["summary"].apply(lambda x: x.str.split().str.len().mean())
summarylength = summarylength.fillna(0).rename("Summary_Length")

# Percentage Uppercase Feature:
RpercentCap = x.groupby('asin')["reviewText"].apply(lambda x: (x.str.count("[A-Z]")/x.str.len()).mean())
RpercentCap = RpercentCap.fillna(0).rename("Review_Percent_Uppercase")

SpercentCap = x.groupby('asin')["summary"].apply(lambda x: (x.str.count("[A-Z]")/x.str.len()).mean())
SpercentCap = SpercentCap.fillna(0).rename("Summary_Percent_Uppercase")


# Sentiment Analysis
sia = sentiment.SentimentIntensityAnalyzer()
RavgSentiment = x.groupby('asin')["reviewText"].apply(lambda x: np.mean(sia.polarity_scores(x.to_string())["compound"]))
RavgSentiment = RavgSentiment.fillna(0).rename("Review_Avg_Sentiment")

SavgSentiment = x.groupby('asin')["summary"].apply(lambda x: np.mean(sia.polarity_scores(x.to_string())["compound"]))
SavgSentiment = SavgSentiment.fillna(0).rename("Summary_Avg_Sentiment")


# # Testing prep

# Feature vectors must have format: col 1 as 'asin'
# Currently using features:
# 
# Name                    |     Column Name
# 
# reviewCount                    Review_Count
# 
# ~~with_image_percentage          %_Image~~ (Not this)
# 
# percent_verified               %_Verified
# 
# total_votes                    Total_Votes
# 
# reviewlength                   Review_Length
# 
# summarylength                  Summary_Length
# 
# RpercentCap                    Review_%_Uppercase
# 
# SpercentCap                    Summary_%_Uppercase
# 
# ~~actualAwesomeness              Actual_Awesomeness~~ (Not this)
# 
# RavgSentiment                  Review_Avg_Sentiment
# 
# SavgSentiment                  Summary_Avg_Sentiment

#  
#combine all individual features into one dataFrame

#enter any features to be combined here! 
#   They must be pd dataFrames with the 'asin' column for this to work
features = [reviewCount,percent_verified,total_votes,RavgSentiment,SavgSentiment,reviewlength,summarylength,RpercentCap,SpercentCap]

z = x['asin']
for f in features:
    z = pd.merge(z,f,'inner','asin')

#combine resultant data with correct answers 
temp = pd.merge(z,y,'inner','asin') 
temp = temp.groupby("asin").mean()


#split into features (x) and awesomeness (y), which now row correspond
train_merged_x = temp.drop(['awesomeness'], axis=1)
train_merged_y = temp["awesomeness"]

# # Model
# Random Forest

from sklearn.ensemble import RandomForestClassifier

#number of trees, int
n_estimators=150

#Criteria to determine split quality
#“gini”, “entropy”, “log_loss”
criterion = 'log_loss'

#max depth of each tree, int or None for infinite
max_depth = None

#min #/% of samples to leave in a branch after a split
min_samples_split = 0.25

#min #/% of samples to be a leaf node
min_samples_leaf = 0.4

# param_grid = [{'n_estimators': [50,100,150],
#             'criterion': ['gini','entropy','log_loss'],
#             'max_depth':[None,10,15,20],
#             'min_samples_split':[2,0.01,0.05,0.1,0.2],
#             'min_samples_leaf':[2,0.01,0.05,0.1,0.2]}]
# -> {'criterion': 'log_loss', 'max_depth': None, 'min_samples_leaf': 0.2, 'min_samples_split': 0.2, 'n_estimators': 150}
# f1: 0.686935866983373

# param_grid = [{'n_estimators': [125,150,175,200],
#             'min_samples_split':[.1,.2,.3],
#             'min_samples_leaf':[.1,.2,.3]}]
# -> {'min_samples_leaf': 0.3, 'min_samples_split': 0.2, 'n_estimators': 150}
# f1: 0.6944863744763707
            
# param_grid = [{'n_estimators': [140,145,150,155,160],
#             'min_samples_split':[.15,.2,.25],
#             'min_samples_leaf':[.25,.3,.35,.4]}]
# -> {'min_samples_leaf': 0.4, 'min_samples_split': 0.25, 'n_estimators': 150}
# f1: 0.6944863744763707

# param_grid = [{'min_samples_split':[0.15,0.175,0.2,0.225,0.25,0.275,0.3]}]
# -> {'min_samples_split': 0.25}
# f1: 0.6944863744763707

randomforest_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1)


# # K-Fold Testing

#desired metrics
# scoring=['f1','accuracy','precision','recall']

# #number of "folds"
# k=10

# cv_results = cross_validate(randomforest_model, train_merged_x, train_merged_y, cv=k, scoring=scoring)

# # pretty print results
# hline = ("-"*40)

# #print means, stdevs
# print(hline+"\n\tAVG, STDEV TEST RESULTS\n"+hline)
# print("\t\tMEAN\t\tSTDEV")
# for m in scoring:
#         ind = ("test_"+m)
#         test = (m+":").ljust(10)
#         print(f"{test}\t{np.round(cv_results[ind].mean(),4)}\t\t{np.round(cv_results[ind].std(),4)}")


# #Training the model!
randomforest_model.fit(train_merged_x,train_merged_y)


# # Getting Prediction for Test Data

# Loading in test data
x = pd.read_json(os.path.join(current_folder, CDs_folder, 'test1', 'review_test.json'))
y = pd.read_json(os.path.join(current_folder, CDs_folder, 'test1', 'product_test.json'))

# ## Pre Processing

# Review Count
dropUnverified = x[x.verified == True]
reviewCount = x.groupby('asin')["reviewerID"].count()
reviewCount = reviewCount.rename("Review_Count")


# Percent verified
percent_verified = dropUnverified.groupby("asin")["reviewerID"].count()
percent_verified = percent_verified/reviewCount
percent_verified = percent_verified.apply(lambda x: x if x > 0  else 0)
percent_verified = percent_verified.rename("%_Verified")

# Total number of votes across reviews
x_copy = x.copy(deep=True)
total_review_num = x_copy.vote
total_review_num = total_review_num.apply(lambda x: float(x.replace(",", "")) if type(x) == str  else 0)
total_review_num = total_review_num.rename("vote").to_frame()
x_copy["vote"] = total_review_num["vote"]
total_votes = x_copy.groupby('asin')["vote"].sum("vote")
total_votes = total_votes.rename("Total_Votes")

# Text Analysis

# Length of Reviews Feature
reviewlength = x.groupby('asin')["reviewText"].apply(lambda x: x.str.split().str.len().mean())
reviewlength = reviewlength.fillna(0).rename("Review_Length")

summarylength = x.groupby('asin')["summary"].apply(lambda x: x.str.split().str.len().mean())
summarylength = summarylength.fillna(0).rename("Summary_Length")

# Percentage Uppercase Feature:
RpercentCap = x.groupby('asin')["reviewText"].apply(lambda x: (x.str.count("[A-Z]")/x.str.len()).mean())
RpercentCap = RpercentCap.fillna(0).rename("Review_Percent_Uppercase")

SpercentCap = x.groupby('asin')["summary"].apply(lambda x: (x.str.count("[A-Z]")/x.str.len()).mean())
SpercentCap = SpercentCap.fillna(0).rename("Summary_Percent_Uppercase")


# Sentiment Analysis
sia = sentiment.SentimentIntensityAnalyzer()
RavgSentiment = x.groupby('asin')["reviewText"].apply(lambda x: np.mean(sia.polarity_scores(x.to_string())["compound"]))
RavgSentiment = RavgSentiment.fillna(0).rename("Review_Avg_Sentiment")

SavgSentiment = x.groupby('asin')["summary"].apply(lambda x: np.mean(sia.polarity_scores(x.to_string())["compound"]))
SavgSentiment = SavgSentiment.fillna(0).rename("Summary_Avg_Sentiment")


#Again, combine all individual features into one dataFrame

features = [reviewCount,percent_verified,total_votes,RavgSentiment,SavgSentiment,reviewlength,summarylength,RpercentCap,SpercentCap]

test_merged_x = x['asin']
for f in features:
    test_merged_x = pd.merge(test_merged_x,f,'inner','asin')
test_merged_x = test_merged_x.groupby("asin").mean()

# Make prediction
predictions = randomforest_model.predict(test_merged_x) 

# Rejoin to create output
y.insert(1, "predictions", predictions)


# Output predictions to file!
  
#turn predictions into json
json_str = y.to_json()

#output json to file
with open('predictions.json', 'w') as f:
    f.write(json_str)