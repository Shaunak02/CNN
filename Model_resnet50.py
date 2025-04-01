# Step 1: Import Necessary Libraries
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical

# Step 2: Load CIFAR-100 Dataset and Preprocess Data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalize the pixel values to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)

# Step 3: Load Pre-trained ResNet50 Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Step 4: Add Custom Layers on Top of ResNet50
model = Sequential()
model.add(base_model)  # Add the ResNet50 base model

# Global Average Pooling to reduce spatial dimensions
model.add(GlobalAveragePooling2D())

# Fully connected layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Dropout layer to reduce overfitting (optional, add if needed)
model.add(Dropout(0.5))

# Output layer with 100 neurons (for CIFAR-100) and softmax activation
model.add(Dense(100, activation='softmax'))

# Step 5: Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(x_train, y_train, 
                    epochs=20, 
                    batch_size=64, 
                    validation_data=(x_test, y_test))

model.save('cnn_cifar100_resnet50_model.h5')