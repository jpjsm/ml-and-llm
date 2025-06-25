"""
# Understanding Text Classification in Python @ Datacamp

Discover what text classification is, how it works, and successful use cases. Explore end-to-end examples of how to build a text preprocessing pipeline followed by a text classification model in Python.
[Understanding Text Classification in Python @ Datacamp](https://www.datacamp.com/tutorial/text-classification-python)
"""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# definitions
DATA_FOLDER = './data/'

X_TRAIN_PKL = 'x_train.pkl'
Y_TRAIN_PKL = 'y_train.pkl'

X_TEST_PKL = 'x_test.pkl'
Y_TEST_PKL = 'y_test.pkl'

# load train data
x_train = pd.read_pickle(Path(DATA_FOLDER).joinpath(X_TRAIN_PKL))
y_train = pd.read_pickle(Path(DATA_FOLDER).joinpath(Y_TRAIN_PKL))


# load test data
x_test = pd.read_pickle(Path(DATA_FOLDER).joinpath(X_TEST_PKL))
y_test = pd.read_pickle(Path(DATA_FOLDER).joinpath(Y_TEST_PKL))

# Train Bag of Words model
cv = CountVectorizer()

x_train_cv = cv.fit_transform(x_train)
print(x_train_cv.shape)

# Training Logistic Regression model

lr = LogisticRegression()
lr.fit(x_train_cv, y_train)

# transform X_test using CV
print(x_test.shape)
x_test_cv = cv.transform(x_test)

# generate predictions

predictions = lr.predict(x_test_cv)

print(predictions)

# confusion matrix
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['true-ham','true-spam'], columns=['predicted-ham','predicted-spam'])

print(df.info())
print(df.head())