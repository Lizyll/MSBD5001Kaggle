import pandas as pd
import numpy as np
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import explained_variance_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler 

def ColSplit(lst):
	d = {}
	for item in lst:
		tmp = item.split(',')
		for i in tmp:
			d[i] = d.get(i, 0) + 1
	return d

def PlayOrNot(lst):
	l = []
	for i in lst:
		if i > 0:
			l.append(1)
		else:
			l.append(0)
	return l

def computeTFIDF(wordDict):
	# TF
	tfDict = {}
	bagOfWordsCount = len(wordDict)
	for word, count in wordDict.items():
		tfDict[word] = count / float(bagOfWordsCount)
    # IDF
	import math
	N = len(wordDict)
	idfDict = dict()
	for word, val in wordDict.items():
		idfDict[word] = idfDict.get(word, 0) + math.log(N / float(val))
    # TF-IDF
	tfidf = {}
	for word, val in tfDict.items():
		tfidf[word] = val * idfDict[word]
	return tfidf

def getTextWeight(wordDict, lst):
	weights = []
	for item in lst:
		tmp = item.split(',')
		s = 0.0
		for i in tmp:
			s += wordDict[i]
		weights.append(s)
	return weights

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

#the structure of NN
def create_model(x_features):
    model = tf.keras.models.Sequential()
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, input_shape=(x_features,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(19, input_shape=(41,)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1,))
    model.compile(optimizer='adam', loss='mse')
    return model

def dealwithminus(lst):
	l = []
	for i in lst:
		if i < 0:
			l.append(0)
		else:
			l.append(i)
	return l

if __name__ == '__main__':
	# read
	df = pd.read_csv('train.csv')

	# tags & categories text processing → convert to dicts and count 
	tagDict = ColSplit(df['tags'].tolist())
	categoryDict = ColSplit(df['categories'].tolist())
	genresDict = ColSplit(df['genres'].tolist())


	# convert date data
	df['purchase_date'] = df['purchase_date'].astype('datetime64[ns]')
	df['release_date'] = df['release_date'].astype('datetime64[ns]')
	df['diff'] = ((df['purchase_date'] - df['release_date']).dt.days).fillna(0)


	df['tags'] = getTextWeight(computeTFIDF(tagDict), df['tags'].tolist())
	#df['genres'] = getTextWeight(computeTFIDF(genresDict), df['genres'].tolist())
	#df['categories'] = getTextWeight(computeTFIDF(categoryDict), df['categories'].tolist())
	#df.to_csv('engineered.csv', encoding='utf-8', index=False)

	pl = (df['price'].to_numpy() / 1000)
	df['price'] = pl

	# NN to get categories and genres
	train_x = (df['categories'].str.get_dummies(',')).to_numpy().astype(np.float32)
	train_y = np.array(PlayOrNot(df['playtime_forever'].tolist())).astype(np.float32)
	train_x1 = (df['genres'].str.get_dummies(',')).to_numpy().astype(np.float32)

	# xgboost → logictic regression
	xgbc = XGBClassifier()
	xgbc.fit(train_x, train_y)
	print('xgboost score: ', explained_variance_score(train_y, xgbc.predict(train_x)))
	df['PlayOrNot_categories'] = xgbc.predict(train_x)

	xgbc1 = XGBClassifier()
	xgbc1.fit(train_x1, train_y)
	print('xgboost score: ', explained_variance_score(train_y, xgbc1.predict(train_x1)))
	df['PlayOrNot_genres'] = xgbc1.predict(train_x1)
	df.to_csv('engineered.csv', encoding='utf-8', index=False)

	
	# linear regression
	X_train = df[['price',  'PlayOrNot_categories', 'tags', 'total_positive_reviews', 'total_negative_reviews', 'diff']].fillna(0)
	Y_train = df['playtime_forever']
	X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.80, random_state=33)
	model_linear = LinearRegression()
	model_linear.fit(X_train, Y_train)
	print('y = ax + b |', 'b = ', model_linear.intercept_, 'a = ', model_linear.coef_)
	pre_y = np.asarray(model_linear.predict(X_test))
	print('Linear model RMSE: ', rmse(pre_y, Y_test.to_numpy()))
	print('linear model score: ', model_linear.score(X_test, Y_test))
	

	# test

	dftest = pd.read_csv('test.csv')

	# tags & categories text processing → convert to dicts and count 
	tagDict1 = ColSplit(dftest['tags'].tolist())


	# convert date data
	dftest['purchase_date'] = dftest['purchase_date'].astype('datetime64[ns]')
	dftest['release_date'] = dftest['release_date'].astype('datetime64[ns]')
	dftest['diff'] = ((dftest['purchase_date'] - dftest['release_date']).dt.days).fillna(0)
	dftest['tags'] = getTextWeight(computeTFIDF(tagDict1), dftest['tags'].tolist())

	pl = (dftest['price'].to_numpy() / 1000)
	dftest['price'] = pl

	# NN to get categories and genres
	a = (dftest['categories'].str.get_dummies(',')).to_numpy().astype(np.int)
	a1 = (dftest['genres'].str.get_dummies(',')).to_numpy().astype(np.int)
	N = len(dftest)
	m = a.shape[1]
	m1 = a1.shape[1]
	train_x_test = np.zeros((N,m+1))
	train_x_test[:,:-1] = a
	train_x1_test = np.zeros((N,m1+6))
	train_x1_test[:,:-6] = a1
	# xgboost → logictic regression
	dftest['PlayOrNot_categories'] = xgbc.predict(train_x_test)
	dftest['PlayOrNot_genres'] = xgbc1.predict(train_x1_test)
	#df.to_csv('engineered.csv', encoding='utf-8', index=False)

	
	# linear regression
	X_pre = dftest[['price', 'PlayOrNot_categories', 'tags', 'total_positive_reviews', 'total_negative_reviews', 'diff']].fillna(0)
	es_playtime = model_linear.predict(X_pre)
	dftest['playtime_forever'] = dealwithminus(es_playtime)
	dfresult = dftest[['id', 'playtime_forever']]
	dfresult.to_csv('preresult.csv', encoding='utf-8', index=False)

	
	

	





