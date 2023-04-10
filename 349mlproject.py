#Project by:
#Zachary Gerstenfeld....._______________

#enter the path to get to CDs_and_Vinyl folder here
PATH_TO_CDS = 'devided_dataset_v2\\CDs_and_Vinyl\\'

import json
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False, sort=True)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

# use sklearn's tfidf or just count

# Open json training files
# f_x_train = open(PATH_TO_CDS + 'train\\review_training.json') 
# f_y_train = open(PATH_TO_CDS + 'train\\product_training.json')

# Load training data
# x = json.load(f_x_train)
# y = json.load(f_y_train)

x = pd.read_json(PATH_TO_CDS + 'train\\review_training.json')
y = pd.read_json(PATH_TO_CDS + 'train\\product_training.json')

z = pd.merge(x,y,'inner','asin')

print(z.head)

# pandas function - try to join by asin
# then take awesomeness col and =y


# z.columns
z=z.dropna(subset="summary")
x = z["summary"]
y = z["awesomeness"]
y=y[:20000]
x=x[:20000]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!
#note: many reviews/item, so need to combine whatever numerical results from reviews -> 1 value, then use that to predict awesomeness for 1 row of y. 
# Y can't repeat if we're going for practical results!!! 

# x = v.fit_transform(x)
# print(type(x))
# y = v.fit_transform(y)

######################################
#if i could just get x, y to be numpy arrays, I can reshape them if necessary, so it'll probs work


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=2)

# make vocab = fit, transform -> numbers = transform
x_train = tfidf.fit_transform(x_train).toarray()
x_test = tfidf.transform(x_test).toarray()

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)
print(f"Number of mislabeled points out of a total {x_test.shape[0]} points : {(y_test != y_pred).sum()}")
