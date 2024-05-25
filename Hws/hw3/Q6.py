import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input

# Assuming grayscale images with shape (height, width, channels)
input_shape = (28, 28, 64)

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
# Add more layers as needed for your specific task

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate a random example input
import numpy as np
example_input = np.random.rand(1, *input_shape)
print("Input shape: ", example_input.shape)

# Get the model prediction
result = model.predict(example_input)
print("Output Shape", result.shape)

