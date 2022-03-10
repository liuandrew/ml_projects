import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from wordcloud import WordCloud
from nltk.stem.porter import *
import warnings

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

train = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')
combi = train.append(test, ignore_index=True)

def remove_pattern(input_txt, pattern):
    """Use Regex to remove all instances of pattern matching

    Args:
        input_txt (str): String to remove pattern from
        pattern (str): Regex string

    Returns:
        str: Cleaned input_txt string
    """
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


'''
Clean text, remove punctuation and short words, stem words
'''
combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], '@[\w]*')
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
combi['tidy_tweet'] = tokenized_tweet.apply(lambda x: ' '.join(x))


combi['tidy_tweet_length'] = combi['tidy_tweet'].apply(len)
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub('\\n',' ',str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'\W',' ',str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'https\s+|www.\s+',r'', str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'http\s+|www.\s+',r'', str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'\s+[a-zA-Z]\s+',' ',str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'\^[a-zA-Z]\s+',' ',str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'\s+',' ',str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].str.lower()

combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\’", "\'", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"won\'t", "will not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"can\'t", "can not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"don\'t", "do not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"dont", "do not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"n\’t", " not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"n\'t", " not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'re", " are", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'s", " is", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\’d", " would", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\d", " would", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'ll", " will", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'t", " not", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'ve", " have", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'m", " am", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\n", "", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\r", "", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"[0-9]", "digit", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\'", "", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r"\"", "", str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'[?|!|\'|"|#]',r'', str(x)))
combi['tidy_tweet'] = combi['tidy_tweet'].map(lambda x: re.sub(r'[.|,|)|(|\|/]',r' ', str(x)))
'''
Generate word clouds
'''
def wordcloud_words(text):
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    

# all_words = ' '.join([text for text in combi['tidy_tweet']])
# normal_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
# negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])

# wordcloud_words(all_words)
# wordcloud_words(normal_words)
# wordcloud_words(negative_words)

'''
Extract and visualize hashtags
'''
def extract_hashtags(x):
    hashtags = []
    for i in x:
        ht = re.findall(r'#(\w+)', i)
        hashtags.append(ht)
        
    return hashtags

HT_regular = extract_hashtags(combi['tidy_tweet'][combi['label'] == 0])
HT_negative = extract_hashtags(combi['tidy_tweet'][combi['label'] == 1])

HT_regular = sum(HT_regular, [])
HT_negative = sum(HT_negative, [])

def freq_plot(hashtags):
    a = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(a.keys()),
                    'Count': list(a.values())})
    d = d.nlargest(columns='Count', n=10)
    plt.figure(figsize=(16, 5))
    ax = sns.barplot(data=d, x='Hashtag', y='Count')
    ax.set(ylabel='Count')
    plt.show()
    
# freq_plot(HT_regular)
# freq_plot(HT_negative)

'''
Modeling
'''
bow_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

# train_bow = bow[:31962, :]
# test_box = bow[31962:, :]

# xtrain_bow, xvalid_bow, ytrain_bow, yvalid_bow = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)
# reg = LogisticRegression()
# reg.fit(xtrain_bow, ytrain_bow)

# pred = reg.predict_proba(xvalid_bow)
# pred_int = pred[:, 1] => 0.3


# tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=1000, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
vec = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}', 
                      ngram_range=(1, 4), use_idf=1,smooth_idf=1,sublinear_tf=1, stop_words = 'english')
train, test = combi.iloc[:31962], combi.iloc[31962:]

xtrain, xvalid, ytrain, yvalid = train_test_split(train, train['label'], test_size=0.3)

# cheating by training tfidf on full dataset
# vec.fit(combi['tidy_tweet'])
# tfidf = vec.transform(xtrain['tidy_tweet'])

tfidf = vec.fit_transform(xtrain['tidy_tweet'])

clf = LogisticRegression(C=1000, class_weight='balanced').fit(tfidf, ytrain)
pred = clf.predict(vec.transform(xvalid['tidy_tweet']))

print(f1_score(pred, yvalid))