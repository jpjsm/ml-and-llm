"""
# Understanding Text Classification in Python @ Datacamp

Discover what text classification is, how it works, and successful use cases. Explore end-to-end examples of how to build a text preprocessing pipeline followed by a text classification model in Python.
[Understanding Text Classification in Python @ Datacamp](https://www.datacamp.com/tutorial/text-classification-python)
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# definitions
DATA_FOLDER = './data/'
CLEAN_DATA_PKL = 'clean_data.pkl'
X_TRAIN_PKL = 'x_train.pkl'
X_TEST_PKL = 'x_test.pkl'
Y_TRAIN_PKL = 'y_train.pkl'
Y_TEST_PKL = 'y_test.pkl'


# load data
clean_data = pd.read_pickle(Path(DATA_FOLDER).joinpath(CLEAN_DATA_PKL))

# Create Feature and Label sets

x = clean_data['text']

y = clean_data['label']
# train test split (66% train - 33% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=123)

print('Training Data :', x_train.shape)
print('Testing Data : ', x_test.shape)

for df, pkl in list([(x_train, X_TRAIN_PKL), (x_test, X_TEST_PKL), (y_train, Y_TRAIN_PKL), (y_test, Y_TEST_PKL)]):
    df.to_pickle(Path(DATA_FOLDER).joinpath(pkl))