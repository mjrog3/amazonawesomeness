# **Imports**

# Data

import pandas as pd
import os

current_folder = os.getcwd()
CDs_folder = 'CDs_and_Vinyl'

# Open and load json training files
print("Importing Data...")
x = pd.read_json(os.path.join(current_folder, CDs_folder, 'train', 'review_training.json'))
y = pd.read_json(os.path.join(current_folder, CDs_folder, 'train', 'product_training.json'))

# Other imports  
print("Importing Required Modules...")
import time
import numpy as np
from nltk import sentiment
from sklearn.model_selection import cross_validate
from sklearn import metrics
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Patch to speed up sklearn (ONLY IF YOU HAVE AN INTEL CHIP)   
from sklearnex import patch_sklearn 
patch_sklearn()

# ## Pre Processing

# Review Count
print("Counting Reviews...")
tic = time.perf_counter()
dropUnverified = x[x.verified == True]
reviewCount = x.groupby('asin')["reviewerID"].count()
reviewCount = reviewCount.rename("Review_Count")
toc = time.perf_counter()
print(f"Reviews Counted in {toc - tic:0.4f} Seconds!" + "\n")

# Percent verified
print("Determining Percent Verified Reviews...")
tic = time.perf_counter()
percent_verified = dropUnverified.groupby("asin")["reviewerID"].count()
percent_verified = percent_verified/reviewCount
percent_verified = percent_verified.apply(lambda x: x if x > 0  else 0)
percent_verified = percent_verified.rename("%_Verified")
toc = time.perf_counter()
print(f"Percent Verified Reviews Determined in {toc - tic:0.4f} Seconds!" + "\n")

# Total number of votes across reviews
print("Counting Total Votes Across Reviews...")
tic = time.perf_counter()
x_copy = x.copy(deep=True)
total_review_num = x_copy.vote
total_review_num = total_review_num.apply(lambda x: float(x.replace(",", "")) if type(x) == str  else 0)
total_review_num = total_review_num.rename("vote").to_frame()
x_copy["vote"] = total_review_num["vote"]
total_votes = x_copy.groupby('asin')["vote"].sum("vote")
total_votes = total_votes.rename("Total_Votes")
toc = time.perf_counter()
print(f"Votes Counted in {toc - tic:0.4f} Seconds!" + "\n")

# Text Analysis

# Length of Reviews Feature
print("Counting Length of Reviews...")
tic = time.perf_counter()
reviewlength = x.groupby('asin')["reviewText"].apply(lambda x: x.str.split().str.len().mean())
reviewlength = reviewlength.fillna(0).rename("Review_Length")
toc = time.perf_counter()
print(f"Review Length Counted in {toc - tic:0.4f} Seconds!" + "\n")

print("Counting Length of Summaries...")
tic = time.perf_counter()
summarylength = x.groupby('asin')["summary"].apply(lambda x: x.str.split().str.len().mean())
summarylength = summarylength.fillna(0).rename("Summary_Length")
toc = time.perf_counter()
print(f"Summary Length Counted in {toc - tic:0.4f} Seconds!" + "\n")

# Percentage Uppercase Feature:
print("Calculating Percent Capital for Reviews...")
tic = time.perf_counter()
RpercentCap = x.groupby('asin')["reviewText"].apply(lambda x: (x.str.count("[A-Z]")/x.str.len()).mean())
RpercentCap = RpercentCap.fillna(0).rename("Review_Percent_Uppercase")
toc = time.perf_counter()
print(f"Percent Capital for Reviews Calculated in {toc - tic:0.4f} Seconds!" + "\n")

print("Calculating Percent Capital for Summaries...")
tic = time.perf_counter()
SpercentCap = x.groupby('asin')["summary"].apply(lambda x: (x.str.count("[A-Z]")/x.str.len()).mean())
SpercentCap = SpercentCap.fillna(0).rename("Summary_Percent_Uppercase")
toc = time.perf_counter()
print(f"Percent Capital for Summaries Calculated in {toc - tic:0.4f} Seconds!" + "\n")



# Sentiment Analysis

## Helper function for sentiment analysis testing
def review_sentiment(text):
    sia = sentiment.SentimentIntensityAnalyzer()
    return sia.polarity_scores(text)["compound"]

import string
def preprocess(text):
    if (text == None): return ""
    lemmatizer = WordNetLemmatizer()
    text = "".join([char for char in text.lower() if char not in string.punctuation])
    words = [word for word in word_tokenize(text) if word not in stopwords.words("english")]
    tokens = [lemmatizer.lemmatize(token) for token in words]
    return ' '.join(tokens)

print("Processing Text... \n")
print("This usually takes a while (~100 minutes)")
print("Currently removing stop words and lemmatizing from the review and review summaries.")
tic = time.perf_counter()

x["processedText"] = x["reviewText"].apply(preprocess)
x["processedSums"] = x["summary"].apply(preprocess)

toc = time.perf_counter()
print(f"Text Processed in {toc - tic:0.4f} Seconds!" + "\n")

def replaceWords(text, pos_words, neg_words):
    pos_message = "love best awesome"
    neg_message = "terrible disgusting awful"

    #replace
    for pw in pos_words:
        text = text.replace(" "+pw," "+pos_message)
    for nw in neg_words:
        text = text.replace(" "+nw," "+neg_message)

#spaces added to prevent replacing parts of words. 
# Still could replace ends of words, but probs not that much of an issue

    #append
    # for pw in pos_words:
    #     if (" "+pw) in text: text += " "+pos_message 
    # for nw in neg_words:
    #     if (" "+nw) in text: text += " "+neg_message 

    return text

def get_disjoint_elements(arr1,arr2):
    return [[e for e in arr1 if e not in arr2],[e for e in arr2 if e not in arr1]]

from sklearn.feature_extraction.text import CountVectorizer

#use count vectorizer on all reviews, summaries

def count_and_getMostCommon(textdf):
    cv = CountVectorizer(ngram_range=(1,2), max_features=50) #max_features itself gets most common, but not sorted
    #sum the resulting sparse matrices to get total counts
    counts = cv.fit_transform(textdf).toarray().sum(axis=0)
    #argsort to order by MOST COMMON 
    return cv.get_feature_names_out()[np.argsort(counts)[::-1]]

#split data into pos, neg reviews
data_w_awesomeness = x.merge(y,"inner","asin")
pos_revs = data_w_awesomeness.query("awesomeness==1")
neg_revs = data_w_awesomeness.query("awesomeness==0")

#get the words corresponding to the indexes
# put them into an array of pos_words, neg_words
rpos_words = count_and_getMostCommon(pos_revs["processedText"])
rneg_words = count_and_getMostCommon(neg_revs["processedText"])
spos_words = count_and_getMostCommon(pos_revs["processedSums"])
sneg_words = count_and_getMostCommon(neg_revs["processedSums"])

print("Review positive words: "+", ".join(rpos_words))
print("Review negative words: "+", ".join(rneg_words))

print("Summary positive words: "+", ".join(spos_words))
print("Summary negative words: "+", ".join(sneg_words))


#remove common words

[rpos_words,rneg_words] = get_disjoint_elements(rpos_words,rneg_words)
[spos_words,sneg_words] = get_disjoint_elements(spos_words,sneg_words)

print("UNIQUE Review positive words: "+", ".join(rpos_words))
print("UNIQUE Review negative words: "+", ".join(rneg_words))

print("UNIQUE Summary positive words: "+", ".join(spos_words))
print("UNIQUE Summary negative words: "+", ".join(sneg_words))



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