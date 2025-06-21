import tensorflow
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

from my_utils import display_some_examples

model = tensorflow.keras.Sequential(

    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        
        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

if __name__ == '__main__':
    
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    display_some_examples(x_train, y_train)
    
    x_train = x_train.astype('float32') / 255
    x_test = x_train.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    
    model.evaluate(x_test, y_test, batch_size=64)

