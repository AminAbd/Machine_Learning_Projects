
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers.experimental.preprocessing import RandomFlip, RandomZoom

# Data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomZoom(0.2),
])

# Directories for training, validation, and test sets
train_dir = 'Sports Classification/train'
validation_dir = 'Sports Classification/valid'
test_dir = 'Sports Classification/test'
# Convert integer labels to one-hot
def convert_to_one_hot(image, label):
    num_classes = 100  # Adjust based on your dataset
    one_hot_label = tf.one_hot(label, depth=num_classes)
    return image, one_hot_label

# Creating datasets
train_ds = keras.utils.image_dataset_from_directory(
    directory=train_dir,
    image_size=(224, 224),  # ResNet50 expects 224x224 images
    labels='inferred',
    label_mode='int',
    batch_size=32
)

valid_ds = keras.utils.image_dataset_from_directory(
    directory=validation_dir,
    image_size=(224, 224),  # ResNet50 expects 224x224 images
    labels='inferred',
    label_mode='int',
    batch_size=32
)

# Setting up the base model
base_model = ResNet50(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
for layer in base_model.layers:
    layer.trainable = False

def augment_data(image, label):
    # Apply the data_augmentation pipeline
    image = data_augmentation(image)
    return image, label

# Assuming train_ds is your training dataset
train_ds = train_ds.map(augment_data)
# Apply the conversion function to each dataset
train_ds = train_ds.map(convert_to_one_hot)
valid_ds = valid_ds.map(convert_to_one_hot)

# Building the model
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())  # Replace Flatten with GlobalAveragePooling2D
model.add(Dense(256, activation='relu'))
model.add(keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001))
model.add(keras.layers.Dropout(0.1))
model.add(Dense(100, activation='softmax'))  # Adjust based on your classes



# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_ds, epochs=10, validation_data=valid_ds)

# Extracting training and validation loss and accuracy
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Preparing the training results
training_results = "Epoch, Training Loss, Training Accuracy, Validation Loss, Validation Accuracy\n"
for epoch in range(len(train_loss)):
    training_results += f"{epoch+1}, {train_loss[epoch]:.4f}, {train_accuracy[epoch]*100:.2f}%, {val_loss[epoch]:.4f}, {val_accuracy[epoch]*100:.2f}%\n"

# Estimated FLOPs for ResNet50
flops_info = "Estimated FLOPs for ResNet50: 3.8 Billion (for a single forward pass)."

combined_text = training_results + "\n" + flops_info

# Define the file path
file_path = "Sports Classification/resnet50_training_performance.txt"

# Writing the results info to a file
with open(file_path, "w") as file:
    file.write(combined_text)

print(f"Training and validation performance metrics saved to {file_path}")
