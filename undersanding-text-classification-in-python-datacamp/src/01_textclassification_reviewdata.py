"""
# Understanding Text Classification in Python @ Datacamp

Discover what text classification is, how it works, and successful use cases. Explore end-to-end examples of how to build a text preprocessing pipeline followed by a text classification model in Python.
[Understanding Text Classification in Python @ Datacamp](https://www.datacamp.com/tutorial/text-classification-python)
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import keyboard

# definitions
DATA_FOLDER = './data/'
SPAM_PKL = 'spam.pkl'

# load data
data = pd.read_pickle(Path(DATA_FOLDER).joinpath(SPAM_PKL))

print(f"{'-'*10}  Check missing values  {'-'*10}")
print(data.isna().sum())

print(f"{'-'*10}  Check data shape  {'-'*10}")
print(data.shape)

print(f"{'-'*10}  Verify trained data relations  {'-'*10}")
#data['label'].value_counts(normalize = True).plot.bar()
#plt.show()
print(data['label'].value_counts(normalize = True))

