# Project Folders

Description of project folder including completed projects

## twitter_sentiment/

Working through Twitter Sentiment Analysis probem https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/, involving text cleaning, preliminary data analysis, followed by TfIdf and CBOW models for regression. Practice handling unbalanced classes. 

## bike_rental/

Given a dataset about bike rental rates, we would like to predict hourly bike demand given weather conditions, timestamps, etc. This project preliminarily involves data exploration, visualization, and data cleaning (fixing missing values, removing columns with correlated information).

Next, I perform some feature engineering by converting timestamp data into hour, day, month, and year categorical variables. I transform all categorical variables into one-hot encodings, perform hyperparmeter search on a range of models, and finally settle on a single model using a validation set.

I examine the feature importances of the model, and finally predict the test data set. 

## transformer/

First working implementation of a transformer, following http://peterbloem.nl/blog/transformers, applied to IMDB sentiment analysis. Achieved good accuracy (about 70% over 10 minutes training). Successfully setup local GPU usage, but memory was limited and had to limit the sequence length used.

## hugging_face/

Playing around with using and implementing the HuggingFace API as following the provided course https://huggingface.co/course