from datetime import datetime, timedelta
import itertools as itools
from  pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import warnings
warnings.filterwarnings('ignore')

# Defined labels
labels_to_number = {'-': 0, '\\': 1, '|': 2, '/': 3}
numbers_to_label = {v:k for k,v in labels_to_number.items()}
labels = labels_to_number.keys()

# mini-pictures resolution
side = 31
InputShape =(side * side,)

# Import saved state
train_images = np.load("./state/Labels_4-train-100000-31x31-images.npy")
train_labels = np.load("./state/Labels_4-train-100000-31x31-labels.npy")


test_images = np.load("./state/Labels_4-test-20000-31x31-images.npy")
test_labels = np.load("./state/Labels_4-test-20000-31x31-labels.npy")

print(f"{train_images.shape=}, {train_labels.shape=}")
print(f"{test_images.shape=}, {test_labels.shape=}")


batch_sizes = [20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
epochs      = [3, 5, 8, 13]
nodes = [8, 16, 32, 64, 128, 256, 512]

best_accuracy = False
results = {}

output_layer = Dense(len(labels), activation='softmax', name='output')

for nodes_number, epoch_number, batch_size_number in itools.product(nodes, epochs, batch_sizes):
    
    print(f"Evaluating: (nodes:{nodes_number}, epochs:{epoch_number}, batch_size:{batch_size_number:,})")
    input_layer = Dense(nodes_number, activation='relu', input_shape=InputShape, name='first')

    model_layers = [
        Dense(nodes_number, activation='relu', input_shape=InputShape, name='first'),
        Dense(len(labels), activation='softmax', name='output')
    ]
    
    # Build the model
    model = Sequential(model_layers)

    # Compile the model
    model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['accuracy'],
    )

    # Train the model.

    # print("Train the model")

    train_start = datetime.now()

    model.fit(
      train_images,
      to_categorical(train_labels),
      epochs=epoch_number,
      batch_size=batch_size_number, 
      verbose=0
    )

    train_delta_secs = (datetime.now() - train_start).total_seconds()
    # print(f"{train_delta_secs=}")

    # Evaluate the model.
    # print("Evaluate the model")

    evaluate_start = datetime.now()
    model.evaluate(
      test_images,
      to_categorical(test_labels),
      batch_size=20,
      verbose=0
    )

    evaluate_delta_secs = (datetime.now() - evaluate_start).total_seconds()

    # print(f"{evaluate_delta_secs=}")

    # Load the model from disk later using:
    # model.load_weights('model.h5')

    # Test accuracy 
    # print("Test accuracy")
    test_start = datetime.now()

    sample_size = test_labels.shape[0]
    predictions = model.predict(test_images[:sample_size],verbose = 0)

    test_delta_secs = (datetime.now() - test_start).total_seconds()
    # print(f"{test_delta_secs=}")

    accuracy = np.count_nonzero(test_labels[:sample_size]==np.argmax(predictions, axis=1))/sample_size
    
    error_count = sample_size - np.count_nonzero(test_labels[:sample_size]==np.argmax(predictions, axis=1))
    # print(f"Empiric {accuracy=}, {error_count=}")
    
    if len(results.keys()) == 0 or (error_count < min(results.keys())):
        best_accuracy = True
        
    if error_count not in results:
        results[error_count] = []
        
    results[error_count].append((nodes_number, epoch_number, batch_size_number, error_count, accuracy, train_delta_secs, test_delta_secs))
    
    with open("./state/Labels_4-ErrorResults-31x31-v2.pkl", "wb") as outpickle:
        pickle.dump(results, outpickle)
        
    best_errors_number = min(results.keys())
    
    if best_accuracy:
        # Save the model to disk.
        model.save_weights('./state/keras_mini_pictures-Labels_4-31x31-v2-model.weights.h5')

        print(f"Min errors: {error_count:,}/{sample_size:,} (nodes_number, epoch_number, batch_size_number): {results[best_errors_number]}")
        best_accuracy = False

print("Job done!")