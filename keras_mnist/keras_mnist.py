# The full neural network code!
###############################
from datetime import datetime, timedelta
import numpy as np
import mnist_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

mnist_utils.temporary_dir = lambda: '/mnist_datasets'

train_images = mnist_utils.train_images()
train_labels = mnist_utils.train_labels()
test_images = mnist_utils.test_images()
test_labels = mnist_utils.test_labels()
print(f"{train_images.shape=}, {train_labels.shape=}, {test_images.shape=}, {test_labels.shape=}")
# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
  Dense(200, activation='relu', input_shape=(784,)),
  #Dense(200, activation='relu'),
  Dense(10, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Train the model.
print("Train the model")
train_start = datetime.now()
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=10,
  batch_size=20,
)

train_delta_secs = (datetime.now() - train_start).total_seconds()
print(f"{train_delta_secs=}")

# Evaluate the model.
print("Evaluate the model")
evaluate_start = datetime.now()
model.evaluate(
  test_images,
  to_categorical(test_labels),
  batch_size=20
)

evaluate_delta_secs = (datetime.now() - evaluate_start).total_seconds()
print(f"{evaluate_delta_secs=}")

# Save the model to disk.
model.save_weights('./keras_mnist-model.weights.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Test accuracy 
print("Test accuracy")
test_start = datetime.now()
sample_size = test_labels.shape[0]
# Predict on the first 5 test images.
predictions = model.predict(test_images[:sample_size])
test_delta_secs = (datetime.now() - test_start).total_seconds()
print(f"{test_delta_secs=}")

# Print our model's predictions.
#print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
#print(test_labels[:sample_size]) # [7, 2, 1, 0, 4]
accuracy = np.count_nonzero(test_labels[:sample_size]==np.argmax(predictions, axis=1))/sample_size
print(f"Empiric {accuracy=}")