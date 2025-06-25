"""
# Understanding Text Classification in Python @ Datacamp

Discover what text classification is, how it works, and successful use cases. Explore end-to-end examples of how to build a text preprocessing pipeline followed by a text classification model in Python.
[Understanding Text Classification in Python @ Datacamp](https://www.datacamp.com/tutorial/text-classification-python)
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# definitions
NLTK_DOWNLOADS = './nltk_downloads/'
DATA_FOLDER = './data/'
SPAM_PKL = 'spam.pkl'
CLEAN_DATA_PKL = 'clean_data.pkl'

# prepare nltk
nltk.download('all',download_dir=NLTK_DOWNLOADS)

lemmatizer = WordNetLemmatizer()

# load data
data = pd.read_pickle(Path(DATA_FOLDER).joinpath(SPAM_PKL))

# begin cleanup
text = list(data['text'])
print(data['text'].head())

corpus = []
for i in range(len(text)):
    r = re.sub('[^a-zA-z]', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = ' '.join(r)
    corpus.append(r)

# replace data['text'] with corpus values <- leaving only meaningful words in text
data['text'] = corpus
print(data['text'].head())

data.to_pickle(Path(DATA_FOLDER).joinpath(CLEAN_DATA_PKL))