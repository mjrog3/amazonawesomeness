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

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
plt.style.use('Solarize_Light2')
import matplotlib.pyplot as plt
from sklearn import metrics
from nltk import sentiment
import numpy as np
import pickle
import string
import time
import re

# Patch to speed up sklearn (ONLY HELPS IF YOU HAVE AN INTEL CHIP)   
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

sia = sentiment.SentimentIntensityAnalyzer()
## Helper function for sentiment analysis testing
def review_sentiment(text):
    sia = sentiment.SentimentIntensityAnalyzer()
    #split at punctuation to do per-sentence analysis
    sentences = re.split("[!?.] ",text)
    score = 0
    #calculate sentiment for each sentence
    for s in sentences:
        scores = sia.polarity_scores(s)
        score += scores["compound"]
    #return average sentiment of sentences as this text block's sentiment
    return score/len(sentences)

def preprocess(text):
    if (text == None): return ""
    lemmatizer = WordNetLemmatizer()
    text = "".join([char for char in text.lower() if char not in string.punctuation])
    words = [word for word in word_tokenize(text) if word not in stopwords.words("english")]
    tokens = [lemmatizer.lemmatize(token) for token in words]
    return ' '.join(tokens)

print("Processing Text... \n")
print("FYI: This usually takes a while (~100 minutes)")
print("Currently removing stop words and lemmatizing the reviews and review summaries.")
tic = time.perf_counter()

x["processedText"] = x["reviewText"].apply(preprocess)
print("Reviews processed!")
x["processedSums"] = x["summary"].apply(preprocess)
print("Summaries processed!")

toc = time.perf_counter()
print(f"Text Processed in {toc - tic:0.4f} Seconds!" + "\n")


def replaceWords(text, pos_words, neg_words):
    pos_message = "LOVE BEST AWESOME !!!"
    neg_message = "TERRIBLE DISGUSTING AWFUL !!!"

    #replace
    for pw in pos_words:
        text = text.replace(" "+pw," "+pos_message)
    for nw in neg_words:
        text = text.replace(" "+nw," "+neg_message)

    #spaces added to prevent replacing parts of words. 
    # still could replace ends of words, but probs not that much of an issue

    return text

# returns the symmetric difference of arr1 and arr2 - elements of arr1 and arr2 that are not elements of the other
def get_disjoint_elements(arr1,arr2):
    return [[e for e in arr1 if e not in arr2],[e for e in arr2 if e not in arr1]]


#use count vectorizer on all reviews, summaries

#max_features itself gets most common, but not sorted
cv = CountVectorizer(ngram_range=(1,2), max_features=50)

def count_and_getMostCommon(textdf):
    #sum the resulting sparse matrices to get total counts
    counts = cv.fit_transform(textdf).toarray().sum(axis=0)
    #argsort to order by MOST COMMON 
    return cv.get_feature_names_out()[np.argsort(counts)[::-1]]

print("Finding most common words...")
tic = time.perf_counter()

#split data into pos, neg reviews
data_w_awesomeness = x.merge(y,"inner","asin")
pos_revs = data_w_awesomeness.query("awesomeness==1")
neg_revs = data_w_awesomeness.query("awesomeness==0")

#get the most common pos/neg words in reviews (r) and summaries (s)
# put them into an array of (r/s)pos_words, (r/s)neg_words

rpos_words = count_and_getMostCommon(pos_revs["processedText"])
rneg_words = count_and_getMostCommon(neg_revs["processedText"])
spos_words = count_and_getMostCommon(pos_revs["processedSums"])
sneg_words = count_and_getMostCommon(neg_revs["processedSums"])

print("Review positive words: "+", ".join(rpos_words))
print("Review negative words: "+", ".join(rneg_words) + "\n")

print("Summary positive words: "+", ".join(spos_words))
print("Summary negative words: "+", ".join(sneg_words) + "\n")


#remove common words

[rpos_words,rneg_words] = get_disjoint_elements(rpos_words,rneg_words)
[spos_words,sneg_words] = get_disjoint_elements(spos_words,sneg_words)

print("UNIQUE Review positive words: "+", ".join(rpos_words))
print("UNIQUE Review negative words: "+", ".join(rneg_words) + "\n")

print("UNIQUE Summary positive words: "+", ".join(spos_words))
print("UNIQUE Summary negative words: "+", ".join(sneg_words) + "\n")

toc = time.perf_counter()
print(f"Words found in {toc - tic:0.4f} Seconds!" + "\n")


print("Analyzing text sentiment...")
tic = time.perf_counter()

x["rsent"] = x["processedText"].apply(lambda y: review_sentiment(y))
# x.loc[:, ["rsent"]] = x.loc[:,['rsent']].multiply(x.loc[:, 'vote_weight'], axis="index")
# x.loc[:, ["rsent"]] = x.loc[:,['rsent']].multiply(x.loc[:, 'image_weight'], axis="index")
RavgSentiment = x.groupby("asin")["rsent"].mean()
RavgSentiment = RavgSentiment.rename("Review_Avg_Sentiment")

x['sumsent'] = x["processedSums"].apply(lambda y: review_sentiment(y))
# x.loc[:, ["sumsent"]] = x.loc[:,['sumsent']].multiply(x.loc[:, 'vote_weight'], axis="index")
# x.loc[:, ["sumsent"]] = x.loc[:,['sumsent']].multiply(x.loc[:, 'image_weight'], axis="index")
SavgSentiment = x.groupby("asin")["sumsent"].mean()
SavgSentiment = SavgSentiment.rename("Summary_Avg_Sentiment")

toc = time.perf_counter()
print(f"Words found in {toc - tic:0.4f} Seconds!" + "\n")


# # Testing prep

# Feature vectors must have format: col 1 as 'asin'
# Currently using features:
# 
# Name                    |     Column Name
# 
# reviewCount             |      Review_Count
# 
# ~~with_image_percentage |      %_Image~~ (Not this)
# 
# percent_verified        |      %_Verified
# 
# total_votes             |      Total_Votes
# 
# reviewlength            |      Review_Length
# 
# summarylength           |      Summary_Length
# 
# RpercentCap             |      Review_%_Uppercase
# 
# SpercentCap             |      Summary_%_Uppercase
# 
# ~~actualAwesomeness     |      Actual_Awesomeness~~ (Not this)
# 
# RavgSentiment           |      Review_Avg_Sentiment
# 
# SavgSentiment           |      Summary_Avg_Sentiment

#  
#combine all individual features into one dataFrame

print("Generating test dataframe...")
tic = time.perf_counter()

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

toc = time.perf_counter()
print(f"Dataframe generated in {toc - tic:0.4f} Seconds!" + "\n")


print("Fitting tfidf") .//////IM HEREEEEEEEEEEEEEEeee

# tfidf for reviews

tfidf = TfidfVectorizer(vocabulary=np.concatenate((rpos_words,rneg_words)))

tfidf_DF = x[["asin","processedText"]]

tfidf.fit_transform(tfidf_DF["processedText"])
Rfeature_names = [("R_"+e) for e in tfidf.get_feature_names_out()]

tfidf2 = TfidfVectorizer(vocabulary=tfidf.vocabulary_)
tfidfarray = tfidf2.fit_transform(tfidf_DF["processedText"]).toarray()

featuredf = pd.DataFrame(tfidfarray,columns = Rfeature_names)
tfidf_DF = pd.concat([tfidf_DF,featuredf],axis=1)
RtfidfFeature = tfidf_DF.groupby("asin")[Rfeature_names].mean()


###
# tfidf for summary
tfidf = TfidfVectorizer(vocabulary=np.concatenate((spos_words,sneg_words)))
# 
# x["processedText"] = revtext
tfidf_DF = x[["asin","processedSums"]]



# tfidf_DF["processedText"] = 
# tfidf_DF["processedTest"] = tfidf_DF["processedTest"].fillna("")
# tfidf_DF["processedRevs"] = tfidf_DF["reviewText"].fillna("").apply(preprocess)

tfidf.fit_transform(tfidf_DF["processedSums"])
# tfidffeatures.toarray()
Sfeature_names = [("S_"+e) for e in tfidf.get_feature_names_out()]

tfidf2 = TfidfVectorizer(vocabulary=tfidf.vocabulary_)
tfidfarray = tfidf2.fit_transform(tfidf_DF["processedSums"]).toarray()

featuredf = pd.DataFrame(tfidfarray,columns = Sfeature_names)
tfidf_DF = pd.concat([tfidf_DF,featuredf],axis=1)
StfidfFeature = tfidf_DF.groupby("asin")[Sfeature_names].mean()

StfidfFeature

###


# # Model
# Random Forest

# #number of trees, int
# n_estimators=150

# #Criteria to determine split quality
# #“gini”, “entropy”, “log_loss”
# criterion = 'log_loss'

# #max depth of each tree, int or None for infinite
# max_depth = None

# #min #/% of samples to leave in a branch after a split
# min_samples_split = 0.25

# #min #/% of samples to be a leaf node
# min_samples_leaf = 0.4

# randomforest_model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_jobs=-1)


# retrieve picked model
print("Getting pickled model...")
tic = time.perf_counter()

filename = "pickled_model"
randomforest_model = pickle.load(open(filename, 'rb'))

toc = time.perf_counter()
print(f"Model retrieved in {toc - tic:0.4f} Seconds!" + "\n")


# Make prediction
print("Making prediction...")
tic = time.perf_counter()

predictions = randomforest_model.predict(test_merged_x) 

toc = time.perf_counter()
print(f"Model prediction complete in {toc - tic:0.4f} Seconds!" + "\n")


# Rejoin with "asin" column to create output
y.insert(1, "predictions", predictions)


# Output predictions to file!

print("Outputting prediction to file...")
tic = time.perf_counter()

#turn predictions into json
json_str = y.to_json()

#output json to file
with open('predictions.json', 'w') as f:
    f.write(json_str)

toc = time.perf_counter()
print(f"Prediction output in {toc - tic:0.4f} Seconds! \n \n DONE!")
