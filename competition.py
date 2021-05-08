'''
DSCI 553 | Foundations and Applications of Data Mining
Competition Project
Matheus Schmitz

Method Description:
The main secret-sauce in my model s performance is morphing my XGBoost to be more Neural Network like. This is done by using training epochs in combination with additive and multiplicative noise to prevent overfitting, very similar to how modern image/audio neural networks are trained. In each epoch the model gets a new set of slightly noisy-ed features.
My XGBoost model also uses sampling for both datapoints and features in order to control overfitting, which was one of the biggest problems with my model, whose TRAIN rmse gets as low as 0.50 without overfitting control, while the VALID rmse stays awful (over 1.00).
A second cleaver implementation is figuring out from Yelp s website which is their official list of categories, there are 22 and they are used to extract the main business categories as predictors.
I have also built a Latent Semantic Analysis pipeline, which consist of a TF-IDF vectorization of the Tips file condensed with Singular Value Decomposition (SVD) to shrink the sparse matrix into something more useful/tenable for the XGBoost model.
Another feature engineering tool that improved results was creating interaction terms from the user/business features, to capture the particular relationship between each user-business pair.
By employing a lot of careful and intricate data type manipulation I can significantly speed up the feature construction of my rather large model, which then leaves me with more free time to run extra epochs in XGBoost, and also allows for a slower, more fine-grained learning rate to be used.

Error Distribution:
>=0 and <1: 102651
>=1 and <2: 32403
>=2 and <3: 6183
>=3 and <4: 806
>=4: 1

RMSE:
0.976579154429427

Execution Time:
1683s
'''

# Load packages
import sys
from pyspark import SparkContext, SparkConf
import time
import xgboost as xgb
import numpy as np
import json
import gc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from textblob import TextBlob
#from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
#from sklearn.impute import IterativeImputer
#import sklearn.ensemble


# The categories from the Yelp dataset are available at: https://blog.yelp.com/2018/01/yelp_category_list
yelp_categories = ["Active Life", "Arts & Entertainment", "Automotive", "Beauty & Spas", "Education", "Event Planning & Services", 
"Financial Services", "Food", "Health & Medical", "Home Services", "Hotels & Travel", "Local Flavor", "Local Services", "Mass Media", 
"Nightlife", "Pets", "Professional Services", "Public Services & Government", "Real Estate", "Religious Organizations", "Restaurants", "Shopping"]


def get_augmented_features_train(row):
	
	# Hyperparameters -- Apply dropout to a certain share of features (set by DROPOUT_RATE). For all non-dropped features, apply additive/multiplicative noise of varying levels.
	DROPOUT_RATE = 0.0
	NOISE_HIGH = 1.75 # 1.5 # 2.0
	NOISE_MEDIUM = 0.033 # 0.033 # 0.5 # 0.05
	NOISE_LOW = 0.033 # 0.1 # 0.025
	NOISE_NONE = 0.0

	### Averages ###
	try: 
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		user_avg_X = user_avg_rating[row[0][0]] + np.random.normal(loc=0, scale=NOISE_HIGH)
		user_avg_X = np.array([min(max(user_avg_X, 1.0), 5.0)], dtype=np.float32) # Prevent noisy features from going over the natural limits
		#else:
		#	user_avg_X = np.array([np.NaN], dtype=np.float32) # Dropout
	except:
		user_avg_X = np.array([np.NaN], dtype=np.float32)

	try: 
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		bizz_avg_X = bizz_avg_rating[row[0][1]] + np.random.normal(loc=0, scale=NOISE_HIGH)
		bizz_avg_X = np.array([min(max(bizz_avg_X, 1.0), 5.0)], dtype=np.float32) # Prevent noisy features from going over the natural limits
		#else:
		#	bizz_avg_X = np.array([np.NaN], dtype=np.float32) # Dropout
	except:
		bizz_avg_X = np.array([np.NaN], dtype=np.float32)

	### User & Business Complimentary Features ###
	try:
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		user_X = user_features[row[0][0]]
		user_X = np.array([min(max(user_X[0] + np.random.normal(loc=0, scale=NOISE_MEDIUM), 1.0), 5.0), user_X[1], user_X[2], user_X[3], user_X[4], user_X[5], user_X[6], user_X[7], user_X[8]], dtype=np.float32)
		#else:
		#	user_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32) # Dropout
	except:
		user_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		bizz_X = bizz_features[row[0][1]][0:5]
		bizz_X = np.array([min(max(bizz_X[0] + np.random.normal(loc=0, scale=NOISE_MEDIUM), 1.0), 5.0), bizz_X[1], bizz_X[2], bizz_X[3], bizz_X[4]], dtype=np.float32)
		#else:
		#	bizz_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32) # Dropout
	except:
		bizz_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		categories_X = np.array([cat in bizz_features[row[0][1]][5] for cat in yelp_categories], dtype=np.float32)
	except:
		categories_X = np.array([np.NaN for _ in range(len(yelp_categories))], dtype=np.float32)
	
	### Iem & User CF Features ###
	try: 
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		item_CF_X = item_CF_feature_train[row[0][0], row[0][1]] + np.random.normal(loc=0, scale=NOISE_HIGH)
		item_CF_X = np.array([min(max(item_CF_X, 1.0), 5.0)], dtype=np.float32) # Prevent noisy features from going over the natural limits
		#else:
		#	item_CF_X = np.array([np.NaN], dtype=np.float32) # Dropout
	except:
		item_CF_X = np.array([np.NaN], dtype=np.float32)
	'''
	try: 
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			user_CF_X = user_CF_feature_train[row[0][0], row[0][1]] + round(np.random.normal(loc=0, scale=NOISE_HIGH), 4)
			user_CF_X = [min(max(user_CF_X, 1.0), 5.0)] # Prevent noisy features from going over the natural limits
		else:
			user_CF_X = [np.NaN] # Dropout
	except:
		user_CF_X = [np.NaN]
	'''

	### Reviews Textual Features ###
	'''
	try:
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			review_user_sentiment_X = user_review_features[row[0][0]][0]
			review_user_sentiment_X = np.array([min(max(feat + round(np.random.normal(loc=0, scale=NOISE_LOW), 4) -1.0), 1.0) for feat in review_user_sentiment_X], dtype=np.float32)
		else:
			review_user_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32) # Dropout
	except:
		review_user_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			review_user_LSA_X = user_review_features[row[0][0]][1]
			#review_user_LSA_X = np.array([feat * np.random.normal(loc=1, scale=NOISE_LOW) for feat in review_user_LSA_X], dtype=np.float32)
		else:
			review_user_LSA_X = np.array([np.NaN for _ in range(reviews_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32) # Dropout
	except:
		review_user_LSA_X = np.array([np.NaN for _ in range(reviews_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32)

	try:
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			review_user_date_X = user_review_features[row[0][0]][2]
			review_user_date_X = np.array([feat * np.random.normal(loc=1, scale=NOISE_LOW) for feat in review_user_date_X], dtype=np.float32)
		else:
			review_user_date_X = np.array([np.NaN], dtype=np.float32) # Dropout
	except:
		review_user_date_X = np.array([np.NaN], dtype=np.float32)
	
	try:
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			review_bizz_sentiment_X = bizz_review_features[row[0][1]][0]
			review_bizz_sentiment_X = np.array([min(max(feat + np.random.normal(loc=0, scale=NOISE_LOW), 4), -1.0) for feat in review_bizz_sentiment_X], dtype=np.float32)
		else:
			review_bizz_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32) # Dropout
	except:
		review_bizz_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			review_bizz_LSA_X = bizz_review_features[row[0][1]][1]
			review_bizz_LSA_X = np.array([feat * np.random.normal(loc=1, scale=NOISE_LOW) for feat in review_bizz_LSA_X], dtype=np.float32)
		else:
			review_bizz_LSA_X = np.array([np.NaN for _ in range(reviews_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32) # Dropout
	except:
		review_bizz_LSA_X = np.array([np.NaN for _ in range(reviews_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32)
	
	try:
		if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
			review_bizz_date_X = user_review_features[row[0][1]][2]
			review_bizz_date_X = np.array([feat * np.random.normal(loc=1, scale=NOISE_LOW) for feat in review_bizz_date_X], dtype=np.float32)
		else:
			review_bizz_date_X = np.array([np.NaN], dtype=np.float32) # Dropout
	except:
		review_bizz_date_X = np.array([np.NaN], dtype=np.float32)
	'''
	
	### Tips Textual Features ###
	try:
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		tip_sentiment_X = tips_features[row[0][0], row[0][1]][0]
		tip_sentiment_X = np.array([min(max(feat + np.random.normal(loc=0, scale=NOISE_LOW), -1.0), 1.0) for feat in tip_sentiment_X.tolist()], dtype=np.float32)
		#else:
		#	tip_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32) # Dropout
	except:
		tip_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)
	
	try:
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		tip_LSA_X = tips_features[row[0][0], row[0][1]][1]
		#tip_LSA_X = np.array([feat * np.random.normal(loc=1, scale=NOISE_LOW) for feat in tip_LSA_X], dtype=np.float32)
		#else:
		#	tip_LSA_X = np.array([np.NaN for _ in range(tips_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32) # Dropout
	except:
		tip_LSA_X = np.array([np.NaN for _ in range(tips_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32)
	
	try:
		#if np.random.uniform(low=0.0, high=1.0) >= DROPOUT_RATE:
		tip_date_X = tips_features[row[0][0], row[0][1]][2]
		#else:
		#	tip_date_X = np.array([np.NaN], dtype=np.float32)
	except:
		tip_date_X = np.array([np.NaN], dtype=np.float32)

	### Interaction Terms ###
	try:
		interaction_X = np.array([user_X[0]*user_X[1], user_X[0]*user_X[2], user_X[0]*user_X[7], user_X[0]*bizz_X[0], user_X[0]*bizz_X[1], 
			                                           user_X[1]*user_X[2], user_X[1]*user_X[7], user_X[1]*bizz_X[0], user_X[1]*bizz_X[1], 
			                                           						user_X[2]*user_X[7], user_X[2]*bizz_X[0], user_X[2]*bizz_X[1],
			                                           											 user_X[7]*bizz_X[0], user_X[7]*bizz_X[1],
			                                                                                                          bizz_X[0]*bizz_X[1]], dtype=np.float32)
	except:
		interaction_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	X = np.concatenate([user_avg_X, bizz_avg_X, user_X, bizz_X, categories_X, item_CF_X, tip_sentiment_X, tip_LSA_X, tip_date_X, interaction_X]).astype(np.float32) 
	'''
	if row[0][0]%100000 == 0:
		print('Train sample: ', X)
	'''
	return X


def get_augmented_features_test(row):

	### Averages ###
	try:
		user_avg_X = np.array([user_avg_rating[row[0][0]]], dtype=np.float32)
	except:
		user_avg_X = np.array([np.NaN], dtype=np.float32)

	try:
		bizz_avg_X = np.array([bizz_avg_rating[row[0][1]]], dtype=np.float32)
	except:
		bizz_avg_X = np.array([np.NaN], dtype=np.float32)

	### User & Business Complimentary Features ###
	try:
		user_X = user_features[row[0][0]]
		user_X = np.array([user_X[0], user_X[1], user_X[2], user_X[3], user_X[4], user_X[5], user_X[6], user_X[7], user_X[8]], dtype=np.float32)
	except:
		user_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		bizz_X = np.array(bizz_features[row[0][1]][0:5], dtype=np.float32)
		bizz_X = np.array([bizz_X[0], bizz_X[1], bizz_X[2], bizz_X[3], bizz_X[4]], dtype=np.float32)
	except:
		bizz_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		categories_X = np.array([cat in bizz_features[row[0][1]][5] for cat in yelp_categories], dtype=np.float32)
	except:
		categories_X = np.array([np.NaN for _ in range(len(yelp_categories))], dtype=np.float32)
	
	### Iem & User CF Features ###
	try:
		item_CF_X = item_CF_feature_test[row[0][0], row[0][1]]
		item_CF_X = np.array([item_CF_X], dtype=np.float32)
		#item_CF_X = [min(max(item_CF_X, 1.0), 5.0)]
	except:
		item_CF_X = np.array([np.NaN], dtype=np.float32)
	'''
	try:
		user_CF_X = user_CF_feature_test[row[0][0], row[0][1]]
		user_CF_X = [user_CF_X]
		#user_CF_X = [min(max(user_CF_X, 1.0), 5.0)]
	except:
		user_CF_X = [np.NaN]
	'''
	### Reviews Textual Features ###
	'''
	try:
		review_user_sentiment_X = user_review_features[row[0][0]][0]
	except:
		review_user_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	try:
		review_user_LSA_X = user_review_features[row[0][0]][1]
	except:
		review_user_LSA_X = np.array([np.NaN for _ in range(reviews_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32)

	try:
		review_user_date_X = user_review_features[row[0][0]][2]
	except:
		review_user_date_X = np.array([np.NaN], dtype=np.float32)
	
	try:
		review_bizz_sentiment_X = bizz_review_features[row[0][1]][0]
	except:
		review_bizz_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)
	
	try:
		review_bizz_LSA_X = bizz_review_features[row[0][1]][1]
	except:
		review_bizz_LSA_X = np.array([np.NaN for _ in range(reviews_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32)
	
	try:
		review_bizz_date_X = user_review_features[row[0][1]][2]
	except:
		review_bizz_date_X = np.array([np.NaN], dtype=np.float32)
	'''
	### Tips Textual Features ###
	try:
		tip_sentiment_X = tips_features[row[0][0], row[0][1]][0]
	except:
		tip_sentiment_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)
	
	try:
		tip_LSA_X = tips_features[row[0][0], row[0][1]][1]
	except:
		tip_LSA_X = np.array([np.NaN for _ in range(tips_LSA_pipeline['svd'].components_.shape[0])], dtype=np.float32)
	
	try:
		tip_date_X = tips_features[row[0][0], row[0][1]][2]
	except:
		tip_date_X = np.array([np.NaN], dtype=np.float32)
	
	### Interaction Terms ###
	try:
		interaction_X = np.array([user_X[0]*user_X[1], user_X[0]*user_X[2], user_X[0]*user_X[7], user_X[0]*bizz_X[0], user_X[0]*bizz_X[1], 
			                                           user_X[1]*user_X[2], user_X[1]*user_X[7], user_X[1]*bizz_X[0], user_X[1]*bizz_X[1], 
			                                           						user_X[2]*user_X[7], user_X[2]*bizz_X[0], user_X[2]*bizz_X[1],
			                                           											 user_X[7]*bizz_X[0], user_X[7]*bizz_X[1],
			                                                                                                          bizz_X[0]*bizz_X[1]], dtype=np.float32)
	except:
		interaction_X = np.array([np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN], dtype=np.float32)

	X = np.concatenate([user_avg_X, bizz_avg_X, user_X, bizz_X, categories_X, item_CF_X, tip_sentiment_X, tip_LSA_X, tip_date_X, interaction_X]).astype(np.float32) 
	'''
	if row[0][0]%10000 == 0:
		print('Test sample: ', X)
	'''
	return X


def item_based_CF(user, 
				  bizz, 		
				  user_avg_rating,
				  bizz_avg_rating,
				  user_bizz_rating_dict,
				  bizz_user_rating_dict):
	
	### Ensure no errors in case a user and/or business doesn't have an average rating score ###
	# If both user and business have missing ratings, return the best guess, aka 3
	if user not in user_avg_rating and bizz not in bizz_avg_rating:
		return ((user, bizz), 3.75) # Based on real world knowledge I know that the overall average rating is somewhere between 3.5 and 4.0

	# If only the business has a missing value, we still cannot calculate similarity, so return the average for the associated user
	elif bizz not in bizz_avg_rating:
		return ((user, bizz), user_avg_rating[user])

	# If only the user has a missing value, we still cannot calculate similarity, so return the average for the associated business
	elif user not in user_avg_rating:
		return ((user, bizz), bizz_avg_rating[bizz])

	# If both user and business have ratings, proceed to calculating similarities
	similarities = list()

	# For each business rated by the current user, calculate the similarity between the current business and the comparison business
	bizz_rating_dict = user_bizz_rating_dict[user]
	for encoding in range(len(bizz_rating_dict)):
		pearson_corr = item_item_similarity(bizz, bizz_rating_dict[encoding][0], bizz_user_rating_dict)

		# Skip similarities of 0 to gain performenace
		if pearson_corr == 0:
			continue

		similarities.append((encoding, pearson_corr))
	
	# Calculate the person correlation to make a weighted prediction
	N = 0
	D = 0

	for (encoding, pearson_corr) in similarities:
		bizz_rating_tuple = bizz_rating_dict[encoding]
		business = bizz_rating_tuple[0]
		rating = bizz_rating_tuple[1]
		business_avg_rating = bizz_avg_rating[business]
		N += (rating - business_avg_rating) * pearson_corr
		D += abs(pearson_corr)
	
	prediction = float(bizz_avg_rating[bizz] + N/D) if N != 0 else 3.75 # Based on real world knowledge I know that the overall average rating is somewhere between 3.5 and 4.0

	return ((user, bizz), prediction)


def item_item_similarity(curr_bizz, 
						 comp_bizz,
						 bizz_user_rating_dict):

	# For each business get all pairs of user/rating
	curr_bizz_ratings = bizz_user_rating_dict[curr_bizz]
	comp_bizz_ratings = bizz_user_rating_dict[comp_bizz]

	# Get co-rated users (those who rated both businesses)
	corated_users = set(curr_bizz_ratings.keys()).intersection(set(comp_bizz_ratings.keys()))

	# If there are no co-rated users, its impossible to calculate similarity, so return a guess
	if len(corated_users) == 0:
		return 0.5 

	# Calculate the average rating given to the businesses by the co-rated users
	curr_bizz_total = 0
	comp_bizz_total = 0
	count = 0

	for user in corated_users:
		curr_bizz_total += curr_bizz_ratings[user]
		comp_bizz_total += comp_bizz_ratings[user]
		count += 1

	curr_bizz_avg = curr_bizz_total/count
	comp_bizz_avg = comp_bizz_total/count

	# Calculate the pearson correlation
	curr_x_comp_total = 0
	curr_norm_square = 0
	comp_norm_square = 0

	for user in corated_users:
		curr_x_comp_total += ((curr_bizz_ratings[user] - curr_bizz_avg) * (comp_bizz_ratings[user] - comp_bizz_avg))
		curr_norm_square += (curr_bizz_ratings[user] - curr_bizz_avg)**2
		comp_norm_square += (comp_bizz_ratings[user] - comp_bizz_avg)**2

	# Get the Pearson Correlation (Of guess a correlation if we cannot calculate the correlation for a given pair)
	pearson_corr = curr_x_comp_total/((curr_norm_square**0.5) * (comp_norm_square**0.5)) if curr_x_comp_total != 0 else 0.5
	
	return pearson_corr


def user_based_CF(user, 
				  bizz, 		
				  user_avg_rating,
				  bizz_avg_rating,
				  user_bizz_rating_dict,
				  bizz_user_rating_dict):
	
	### Ensure no errors in case a user and/or business doesn't have an average rating score ###
	# If both user and business have missing ratings, return the best guess, aka 3
	if user not in user_avg_rating and bizz not in bizz_avg_rating:
		return ((user, bizz), 3.75) # Based on real world knowledge I know that the overall average rating is somewhere between 3.5 and 4.0

	# If only the business has a missing value, we still cannot calculate similarity, so return the average for the associated user
	elif bizz not in bizz_avg_rating:
		return ((user, bizz), user_avg_rating[user])

	# If only the user has a missing value, we still cannot calculate similarity, so return the average for the associated business
	elif user not in user_avg_rating:
		return ((user, bizz), bizz_avg_rating[bizz])

	# If both user and business have ratings, proceed to calculating similarities
	similarities = list()

	# For each business rated by the current user, calculate the similarity between the current business and the comparison business
	user_rating_dict = bizz_user_rating_dict[bizz]
	for encoding in range(len(user_rating_dict)):
		pearson_corr = item_item_similarity(user, user_rating_dict[encoding][0], user_bizz_rating_dict)

		# Skip similarities of 0 to gain performenace
		if pearson_corr == 0:
			continue

		similarities.append((encoding, pearson_corr))
	
	# Calculate the person correlation to make a weighted prediction
	N = 0
	D = 0

	for (encoding, pearson_corr) in similarities:
		user_rating_tuple = user_rating_dict[encoding]
		user2 = user_rating_tuple[0]
		rating = user_rating_tuple[1]
		user2_avg_rating = user_avg_rating[user2]
		N += (rating - user2_avg_rating) * pearson_corr
		D += abs(pearson_corr)
	
	prediction = float(user_avg_rating[user] + N/D) if N != 0 else 3.75 # Based on real world knowledge I know that the overall average rating is somewhere between 3.5 and 4.0

	return ((user, bizz), prediction)


def user_user_similarity(curr_user, 
						 comp_user,
						 user_bizz_rating_dict):

	# For each business get all pairs of user/rating
	curr_user_ratings = user_bizz_rating_dict[curr_user]
	comp_user_ratings = user_bizz_rating_dict[comp_user]

	# Get co-rated users (those who rated both businesses)
	corated_bizzs = set(curr_user_ratings.keys()).intersection(set(comp_user_ratings.keys()))

	# If there are no co-rated users, its impossible to calculate similarity, so return a guess
	if len(corated_bizzs) == 0:
		return 0.5 

	# Calculate the average rating given to the businesses by the co-rated users
	curr_user_total = 0
	comp_user_total = 0
	count = 0

	for bizz in corated_bizzs:
		curr_user_total += curr_user_ratings[bizz]
		comp_user_total += comp_user_ratings[bizz]
		count += 1

	curr_user_avg = curr_user_total/count
	comp_user_avg = comp_user_total/count

	# Calculate the pearson correlation
	curr_x_comp_total = 0
	curr_norm_square = 0
	comp_norm_square = 0

	for bizz in corated_bizzs:
		curr_x_comp_total += ((curr_user_ratings[bizz] - curr_user_avg) * (comp_user_ratings[bizz] - comp_user_avg))
		curr_norm_square += (curr_user_ratings[bizz] - curr_user_avg)**2
		comp_norm_square += (comp_user_ratings[bizz] - comp_user_avg)**2

	# Get the Pearson Correlation (Of guess a correlation if we cannot calculate the correlation for a given pair)
	pearson_corr = curr_x_comp_total/((curr_norm_square**0.5) * (comp_norm_square**0.5)) if curr_x_comp_total != 0 else 0.5
	
	return pearson_corr


def merge_dicts(dict1, dict2):
	# If the first element is already a dict...
	if type(dict1) == dict:
		# Then append the second element
		dict1.update(dict2)
		return dict1
	# If it is not (because the reducer is comparing the null item to the first item)...
	else:
		# Then return the first item
		return dict2

if __name__ == "__main__":

	start_time = time.time()

	# Get user inputs
	folder_path = sys.argv[1]
	test_file_name = sys.argv[2]
	output_file_name = sys.argv[3]

	# Initialize Spark with 4 GB memory
	sc = SparkContext.getOrCreate(SparkConf().set("spark.executor.memory", "16g").set("spark.driver.memory", "16g"))
	sc.setLogLevel("ERROR")


	################################
	### Load Train and Test Data ###
	################################
	print(f'Loading Data...')
	stage_time = time.time()

	# Read the CSV skipping its header, and reshape it as ((user, bizz), rating)
	trainRDD = sc.textFile(folder_path+'yelp_train.csv', 8)
	trainHeader = trainRDD.first()
	trainRDD = trainRDD.filter(lambda row: row != trainHeader).map(lambda row: row.split(',')).map(lambda row: ((row[0],row[1]), float(row[2]))).persist() #.sample(False, 0.50)
	validRDD = sc.textFile(test_file_name, 8)
	validHeader = validRDD.first()
	validRDD = validRDD.filter(lambda row: row != validHeader).map(lambda row: row.split(',')).map(lambda row: ((row[0],row[1]), float(row[2]))).persist() #.sample(False, 0.50) #,float(row[2])
	print(f'Loading Data: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')


	###################################
	### Encode Users and Businesses ###
	###################################
	print(f'Generating ID Encodings...')
	stage_time = time.time()

	# Merge RDDs to get all IDs
	mergedRDD = sc.union([trainRDD,validRDD])

	# Get distinct users and businesses (over train and valid datasets)
	distinct_user = mergedRDD.map(lambda row: row[0][0]).distinct().sortBy(lambda user: user).collect()
	distinct_bizz = mergedRDD.map(lambda row: row[0][1]).distinct().sortBy(lambda bizz: bizz).collect()

	# Convert names to IDs (to optimize memory usage when holding the values)
	user_to_encoding, encoding_to_user = {}, {}
	for encoding, real_id in enumerate(distinct_user):
		user_to_encoding[real_id] = encoding
		#encoding_to_user[encoding] = real_id

	bizz_to_encoding, encoding_to_bizz = {}, {}
	for encoding, real_id in enumerate(distinct_bizz):
		bizz_to_encoding[real_id] = encoding
		#encoding_to_bizz[encoding] = real_id

	# Use the IDs to encode the RDD, which reduces memory requirements when holding itemsets, and keep the shape as ((user, bizz), rating)
	trainRDD_enc = trainRDD.map(lambda x: ((user_to_encoding[x[0][0]], bizz_to_encoding[x[0][1]]), x[1])).persist()
	validRDD_enc = validRDD.map(lambda x: ((user_to_encoding[x[0][0]], bizz_to_encoding[x[0][1]]), x[1])).persist()

	# Memory management
	trainRDD.unpersist()
	del trainRDD, trainHeader, validHeader, mergedRDD
	gc.collect()
	print(f'Generating ID Encodings: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')


	###############################
	### Average Rating Features ### 
	###############################
	print(f'Generating Average Rating Features...')
	stage_time = time.time()

	# Calculate average ratings 
	user_avg_rating = trainRDD_enc.map(lambda x: (x[0][0], [x[1]])).reduceByKey(lambda a, b: a + b).map(lambda row: (row[0], sum(row[1]) / len(row[1]))).collectAsMap()
	bizz_avg_rating = trainRDD_enc.map(lambda x: (x[0][1], [x[1]])).reduceByKey(lambda a, b: a + b).map(lambda row: (row[0], sum(row[1]) / len(row[1]))).collectAsMap()
	print(f'Generating Average Rating Features: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')

	
	#####################################
	### Item-Item Similarity Features ### 
	#####################################
	print(f'Generating Item-Item CF Features...')
	stage_time = time.time()

	# For each user/bizz, get a dict with all related bizz/user and the associated rating
	user_bizz_rating_dict = trainRDD_enc.map(lambda x: (x[0][0], [(x[0][1], x[1])])).reduceByKey(lambda dict1, dict2: dict1 + dict2).collectAsMap()
	bizz_user_rating_dict = trainRDD_enc.map(lambda x: (x[0][1], {x[0][0]:x[1]})).reduceByKey(lambda dict1, dict2: merge_dicts(dict1, dict2)).collectAsMap()

	# Generate augmented features
	item_CF_feature_train = trainRDD_enc.map(lambda row:  item_based_CF(row[0][0],
																		row[0][1],
																		user_avg_rating,
																		bizz_avg_rating,
																		user_bizz_rating_dict,
																		bizz_user_rating_dict)).collectAsMap()

	item_CF_feature_test = validRDD_enc.map(lambda  row:  item_based_CF(row[0][0],
																		row[0][1],
																		user_avg_rating,
																		bizz_avg_rating,
																		user_bizz_rating_dict,
																		bizz_user_rating_dict)).collectAsMap()
	
	# Memory management
	del user_bizz_rating_dict, bizz_user_rating_dict
	gc.collect()
	print(f'Generating Item-Item CF Features: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
	
	'''
	#####################################
	### User-User Similarity Features ### 
	#####################################
	print(f'Extracting User-User CF...')
	stage_time = time.time()

	# For each user/bizz, get a dict with all related bizz/user and the associated rating
	user_bizz_rating_dict = trainRDD_enc.map(lambda x: (x[0][0], {x[0][1]:x[1]})).reduceByKey(lambda dict1, dict2: merge_dicts(dict1, dict2)).collectAsMap()
	bizz_user_rating_dict = trainRDD_enc.map(lambda x: (x[0][1], [(x[0][0], x[1])])).reduceByKey(lambda dict1, dict2: dict1 + dict2).collectAsMap()

	# Generate augmented features
	user_CF_feature_train = trainRDD_enc.map(lambda row: user_based_CF(row[0][0],
															 	  row[0][1],
																  user_avg_rating,
															 	  bizz_avg_rating,
															 	  user_bizz_rating_dict,
															 	  bizz_user_rating_dict)).collectAsMap()

	user_CF_feature_test = validRDD_enc.map(lambda row: user_based_CF(row[0][0],
															 	 row[0][1],
																 user_avg_rating,
															 	 bizz_avg_rating,
															 	 user_bizz_rating_dict,
															 	 bizz_user_rating_dict)).collectAsMap()

	# Memory management
	del user_bizz_rating_dict, bizz_user_rating_dict
	gc.collect()
	print(f'Extracting User-User CF: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
	'''
	
	##############################################
	### User & Business Complimentary Features ### 
	##############################################
	print(f'Extracting User & Business Complimentary Features...')
	stage_time = time.time()

	# Read the user and business jsons, and load the features
	user_features = sc.textFile(folder_path+'user.json', 8).map(lambda row: json.loads(row)).filter(lambda row: row['user_id'] in distinct_user).map(lambda row: (user_to_encoding[row['user_id']], np.array([row['average_stars'], row['review_count'], int(row['yelping_since'][2:4]), row['useful'], row['funny'], row['cool'], row['fans'], len(row['friends']), len(row['elite'])], dtype=np.float16))).collectAsMap()
	bizz_features = sc.textFile(folder_path+'business.json', 8).map(lambda row: json.loads(row)).filter(lambda row: row['business_id'] in distinct_bizz).map(lambda row: (bizz_to_encoding[row['business_id']], [row['stars'], row['review_count'], row['is_open'], row['latitude'], row['longitude'], row['categories']])).collectAsMap()
	
	# Clean the business categories keeping only those words relevant for generating features
	for key, value in bizz_features.items():
		try:
			bizz_features[key][5] = ' '.join([category for category in bizz_features[key][5].split(', ') if category in yelp_categories])
		except:
			bizz_features[key][5] = ' '
	print(f'Extracting User & Business Complimentary Features: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')		
	
	'''
	########################################
	### Reviews Latent Semantic Analysis ### 
	#######################################
	print(f'Training Latent Semantic Analysis Pipeline on Reviews...')
	stage_time = time.time()

	# Extract all reviews
	all_review_texts = sc.textFile(folder_path+'review_train.json', 8).map(lambda row: json.loads(row.replace('\\n', ''))).map(lambda row: row['text']).collect()
	
	# Train the Tfidftips_vectorizer and TruncatedSVD (the LSA pipeline)
	reviews_vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=40000, stop_words=None)
	reviews_svd_tf_idf = TruncatedSVD(n_components=40)
	reviews_LSA_pipeline = Pipeline([('tfidf', reviews_vectorizer), 
                                     ('svd', reviews_svd_tf_idf)])
	
	# Train the LSA model
	reviews_LSA_pipeline.fit(all_review_texts)

	# Memory management
	del all_review_texts
	gc.collect()
	print(f'Training Latent Semantic Analysis Pipeline on Reviews: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
	print('reviews_LSA_pipeline size: ', len(reviews_LSA_pipeline['tfidf'].get_feature_names()))

	################################
	### Reviews Textual Features ### 
	################################
	print(f'Extracting Review Features...')
	stage_time = time.time()
	VADER_sentiment_analyzer = SentimentIntensityAnalyzer()
	
	# Extract Review Features with Users as keys
	user_review_features = sc.textFile(folder_path+'review_train.json', 8) \
		.map(lambda row: json.loads(row.replace('\\n', ''))) \
		.filter(lambda row: row['user_id'] in distinct_user) \
		.map(lambda row: (user_to_encoding[row['user_id']], (np.array(list(VADER_sentiment_analyzer.polarity_scores(row['text']).values()), dtype=np.float32), 
															 reviews_LSA_pipeline.transform([row['text']])[0].astype(np.float32),
															 np.array(int(row['date'][2:4]), dtype=np.float32),
															 1))) \
		.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3])) \
		.mapValues(lambda v: {0: v[0]/v[3], 1: v[1]/v[3], 2: v[2]/v[3]}) \
		.collectAsMap()
	
	# Extract Review Features with Businesses as keys
	bizz_review_features = sc.textFile(folder_path+'review_train.json', 8) \
		.map(lambda row: json.loads(row.replace('\\n', ''))) \
		.filter(lambda row: row['business_id'] in distinct_bizz) \
		.map(lambda row: (bizz_to_encoding[row['business_id']], (np.array(list(VADER_sentiment_analyzer.polarity_scores(row['text']).values()), dtype=np.float32), 
																 reviews_LSA_pipeline.transform([row['text']])[0].astype(np.float32),
																 1.0))) \
		.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])) \
		.mapValues(lambda v: {0: np.array(v[0]/v[2], dtype=np.float32), 
							  1: np.array(v[1]/v[2], dtype=np.float32)}) \
		.collectAsMap()

	print(f'Extracting Review Features: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
	'''

	#####################################
	### Tips Latent Semantic Analysis ### 
	#####################################
	print(f'Training Latent Semantic Analysis Pipeline on Tips...')
	stage_time = time.time()
	all_tip_texts = sc.textFile(folder_path+'tip.json', 8).map(lambda row: json.loads(row.replace('\\n', ''))).map(lambda row: row['text']).collect()
	
	# Train the Tfidftips_vectorizer and TruncatedSVD (the LSA pipeline)
	tips_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=60000, stop_words=None)
	tips_svd_tf_idf = TruncatedSVD(n_components=30)
	tips_LSA_pipeline = Pipeline([('tfidf', tips_vectorizer), 
								  ('svd', tips_svd_tf_idf)])
	
	# Train the LSA model
	tips_LSA_pipeline.fit(all_tip_texts)

	# Memory management
	del all_tip_texts
	gc.collect()
	print(f'Training Latent Semantic Analysis Pipeline on Tips: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
	print(f'tips_LSA_pipeline size: {len(tips_LSA_pipeline["tfidf"].get_feature_names())} -> {tips_LSA_pipeline["svd"].components_.shape[0]}')
	
	#############################
	### Tips Textual Features ### 
	#############################
	print(f'Extracting Tips Features...')
	stage_time = time.time()
	VADER_sentiment_analyzer = SentimentIntensityAnalyzer()

	# Extract tips Features with User/Businesses as keys
	tips_features = sc.textFile(folder_path+'tip.json', 8) \
		.map(lambda row: json.loads(row.replace('\\n', ''))) \
		.filter(lambda row: row['user_id'] in distinct_user) \
		.filter(lambda row: row['business_id'] in distinct_bizz) \
		.map(lambda row: ((user_to_encoding[row['user_id']], bizz_to_encoding[row['business_id']]), (np.array(list(VADER_sentiment_analyzer.polarity_scores(row['text']).values()), dtype=np.float16), 
																									 tips_LSA_pipeline.transform([row['text']])[0].astype(np.float32),
																									 np.array([int(row['date'][2:4])], dtype=np.int8)))) \
		.collectAsMap()

	print(f'Extracting Tips Features: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')


	######################
	### Build Datasets ### 
	######################
	print(f'Building Datasets...')
	stage_time = time.time()

	# Feature Engineering
	#mice_imputer = IterativeImputer()
	#polynomial_features = PolynomialFeatures(degree=2)
	#svd_all_features = TruncatedSVD(n_components=25)
	#standard_scaler = StandardScaler()

	# Get train data features to fit the feature transformers
	#X_train = trainRDD_enc.map(lambda row: get_augmented_features_train(row)).collect()
	#X_train = trainRDD_enc.map(lambda row: get_augmented_features_train(row, user_avg_rating, bizz_avg_rating, user_features, bizz_features, item_CF_feature_train, user_CF_feature_train, tipsRDD)).collect()
	#X_train = np.array(X_train)
	#X_train = mice_imputer.fit_transform(X_train)
	#X_train = polynomial_features.fit_transform(X_train)
	#X_train = svd_all_features.fit_transform(X_train)
	#X_train = standard_scaler.fit_transform(X_train)
	
	# Train labels
	y_train = trainRDD_enc.map(lambda row: row[1]).collect()
	y_train = np.array(y_train)

	# Get test data features
	X_test  = validRDD_enc.map(lambda row: get_augmented_features_test(row)).collect()
	#X_test  = validRDD_enc.map(lambda row: get_augmented_features_test(row, user_avg_rating, bizz_avg_rating, user_features, bizz_features, item_CF_feature_test, user_CF_feature_test, tipsRDD)).collect()
	X_test  = np.array(X_test)
	#X_test = mice_imputer.transform(X_test)
	#X_test = polynomial_features.transform(X_test)
	#X_test = svd_all_features.transform(X_test)
	#X_test = standard_scaler.transform(X_test)
	
	# Test labels
	y_test  = validRDD_enc.map(lambda row: row[1]).collect()
	y_test  = np.array(y_test)
	print(f'Building Datasets: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')


	###################
	### Train Model ### 
	###################
	print(f'Starting Training...')
	stage_time = time.time()
	# Instantiate model
	model = xgb.XGBRegressor(n_jobs = -1,
							 n_estimators = 165, 
							 learning_rate = 0.15,
							 num_parallel_tree = 1,
							 booster = 'gbtree',
							 #booster = 'dart',
							 #rate_drop = 0.25,
							 eval_metric = 'rmse',
							 min_child_weight = 0, 
							 min_split_loss = 0,
							 subsample = 0.5, 
							 colsample_bytree = 0.5,
							 max_depth = 5,
							 reg_lambda = 0.0, 
							 reg_alpha = 0.0) 

	# Train using a generator for the training data, to run rounds with noisy data
	EPOCHS = 8 ##7
	first_round = True

	for epoch in range(EPOCHS):
		print(f'    Epoch {epoch+1}/{EPOCHS}...')
		epoch_time = time.time()
		
		# Get train data features
		X_train = trainRDD_enc.map(lambda row: get_augmented_features_train(row)).collect()
		#X_train = trainRDD_enc.map(lambda row: get_augmented_features_train(row, user_avg_rating, bizz_avg_rating, user_features, bizz_features, item_CF_feature_train, user_CF_feature_train, tipsRDD)).collect()
		X_train = np.array(X_train)
		#X_train = mice_imputer.transform(X_train)
		#X_train = polynomial_features.transform(X_train)
		#X_train = svd_all_features.transform(X_train)
		#X_train = standard_scaler.transform(X_train)

		# Train
		if first_round:
			model.fit(X_train, y_train) #, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric = 'rmse')
			first_round = False
		else:
			model.fit(X_train, y_train, xgb_model = model.get_booster()) #, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric = 'rmse')

		print(f'    Epoch {epoch+1}/{EPOCHS}. Epoch Time: {time.time() - epoch_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
		
		# Memory management
		del X_train
		gc.collect()

	# Memory management
	del user_avg_rating, bizz_avg_rating, user_features, bizz_features
	gc.collect()
	print(f'Starting Training: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')	


	##################
	### Predicting ### 
	##################
	print(f'Writing Predictions...')
	stage_time = time.time()
	# Predict
	predictions = model.predict(X_test) #, ntree_limit=75*10)

	# Bind predictions to [1, 5]
	predictions = [min(max(pred, 1.0), 5.0) for pred in predictions]

	# Reverse the ID-Encoding dicts
	encoding_to_user = {v: k for k, v in user_to_encoding.items()}
	encoding_to_bizz = {v: k for k, v in bizz_to_encoding.items()}

	# Output predictions
	with open(output_file_name, "w") as fout:
		fout.write("user_id, business_id, prediction")
		for idx, row in enumerate(validRDD_enc.collect()):
			fout.write("\n" + f"{encoding_to_user[row[0][0]]},{encoding_to_bizz[row[0][1]]},"+str(predictions[idx]))
	print(f'Writing Predictions: Stage Time: {time.time() - stage_time:.0f} seconds. Total Time: {time.time() - start_time:.0f} seconds.')
			

	##############################
	### Evaluating Recommender ### 
	##############################
	print(f'Evaluating Recommender...')

	# Join the ground truth with predictions
	predictionsRDD = sc.textFile(output_file_name, 8)
	predictionsHeader = predictionsRDD.first()
	predictionsRDD = predictionsRDD.filter(lambda row: row != predictionsHeader).map(lambda row: row.split(',')).map(lambda row: ((row[0],row[1]), float(row[2]))).persist()
	evaluation = validRDD.join(predictionsRDD)

	# Report error distribution
	delta = evaluation.map(lambda row: abs(row[1][0] - row[1][1]))
	delta_0_1 = delta.filter(lambda abs_err:      abs_err < 1).count()
	delta_1_2 = delta.filter(lambda abs_err: 1 <= abs_err < 2).count()
	delta_2_3 = delta.filter(lambda abs_err: 2 <= abs_err < 3).count()
	delta_3_4 = delta.filter(lambda abs_err: 3 <= abs_err < 4).count()
	delta_4_5 = delta.filter(lambda abs_err: 4 <= abs_err    ).count()
	print('\n' + 'Error Distribution:')
	print(f'>=0 and <1: {delta_0_1}')
	print(f'>=1 and <2: {delta_1_2}')
	print(f'>=2 and <3: {delta_2_3}')
	print(f'>=3 and <4: {delta_3_4}')
	print(f'>=4: {delta_4_5}')
	
	# Report RMSE
	RMSE = (delta.map(lambda x: x ** 2).mean()) ** 0.5
	print('\n' + 'RMSE:')
	print(f'{RMSE}')

	# Close spark context
	sc.stop()

	# Measure the total time taken and report it
	time_elapsed = time.time() - start_time
	print('\n' + f'Duration: {time_elapsed}')