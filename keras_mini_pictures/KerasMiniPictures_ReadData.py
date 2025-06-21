from datetime import datetime, timedelta
from  pathlib import Path
import json
import pickle
import numpy as np

'''
Data file structure: [{mini-picture}, ...]

mini-picture:
{
        "Inputs": [],           # side x side elements, values between 0 .. 255: 0 == Black, 128 == Gray, 255 == White
        "InputsNormalized": [], # side x side elements, values between 0 .. 1.0: 0 == Black, 0.5 == Gray, 1.0 == White
        "ExpectedOutputs": [],  # number of labels elements, values are 
        "Label": "-",
        "LabelName": "dash",
        "Width": 3              # side
        "Height": 3             # side
}

'''
# Data location
mini_pictures_dir = Path('/mini-pictures')

# Defined labels
labels_to_number = {'-': 0, '\\': 1, '|': 2, '/': 3}
numbers_to_label = {v:k for k,v in labels_to_number.items()}
labels = labels_to_number.keys()

# mini-pictures resolution
side = 31

# data filename format: Labels_<number>-<test|train>-<dataset-size>-<side>-<side>-subset_<sequence-number>.json
with open("C:\\mini-pictures\\Labels_4-v1\\Test\\count_20000\\31x31\\Labels_4-v1-Test-31x31-20000-subset_000001.json", "r") as inputfile:
    test_data = json.load(inputfile)
    
print(f"{len(test_data)=}")

train_data = []
for s in [1]: ## [1, 2, 3]:
    filepath = f"C:\\mini-pictures\\Labels_4-v1\\Train\\count_100000\\31x31\\Labels_4-v1-Train-31x31-100000-subset_000001.json"

    with open(filepath, "r") as inputfile:
        train_data += json.load(inputfile)

print(f"{len(train_data)=}")

# Save state
with open("./state/Labels_4-test-20000-31x31.pkl", "wb") as outpickle:
    pickle.dump(test_data, outpickle)
    
with open("./state/Labels_4-train-100000-31x31.pkl", "wb") as outpickle:
    pickle.dump(train_data, outpickle)
    
# Extract data for model
train_images = np.array([ t["Inputs"] for t in train_data ])
train_labels = np.array([ labels_to_number[t["Label"]] for t in train_data ])

test_images = np.asarray([ t["Inputs"] for t in test_data ])
test_labels = np.asarray([ labels_to_number[t["Label"]] for t in test_data ])

print(f"{test_images.shape=}, {test_labels.shape=}")
print(f"{train_images.shape=}, {train_labels.shape=}")

# Save state
np.save("./state/Labels_4-train-100000-31x31-images.npy", train_images)
np.save("./state/Labels_4-train-100000-31x31-labels.npy", train_labels)

np.save("./state/Labels_4-test-20000-31x31-images.npy", test_images)
np.save("./state/Labels_4-test-20000-31x31-labels.npy", test_labels)
