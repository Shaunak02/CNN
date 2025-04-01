# Import the necessary functions
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



# Load the dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the pixel values to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)


# Print the shape of the data
# print(f"Training data shape: {x_train.shape}")
# print(f"Test data shape: {x_test.shape}")

model = Sequential()
# First Conv Block
model.add(Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())  
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())  
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Second Conv Block
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

# Third Conv Block
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='softmax'))  # 100 classes

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=128, validation_data=(x_test, y_test))


model.save("cnn_cifar100_model.h5")
print("Model saved successfully!")