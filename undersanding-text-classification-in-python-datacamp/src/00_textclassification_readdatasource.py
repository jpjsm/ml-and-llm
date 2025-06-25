"""
# Understanding Text Classification in Python @ Datacamp

Discover what text classification is, how it works, and successful use cases. Explore end-to-end examples of how to build a text preprocessing pipeline followed by a text classification model in Python.
[Understanding Text Classification in Python @ Datacamp](https://www.datacamp.com/tutorial/text-classification-python)
"""
import pandas as pd
from pathlib import Path

# definitions
DATA_FOLDER = './data/'
SPAM_CSV = 'spam.csv'
SPAM_PKL = 'spam.pkl'

data = pd.read_csv('https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv', encoding='latin-1')

data.to_csv(Path(DATA_FOLDER).joinpath(SPAM_CSV))

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data.columns = ['label', 'text']

data.to_pickle(Path(DATA_FOLDER).joinpath(SPAM_PKL))