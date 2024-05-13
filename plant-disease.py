import os
import zipfile
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Step 1: Extract the dataset
dataset_zip_path = 'C:/Users/saibalaji/Downloads/Plant-Leaf-Disease1.zip'
extract_path = './plant_disease_dataset'

with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 2: Data Preprocessing
train_data_dir = os.path.join(extract_path, 'Plant-Leaf-Disease-Prediction-023060fabd036c705a96b0f2b827b4702dd61fc4/Dataset/train')
val_data_dir = os.path.join(extract_path, 'Plant-Leaf-Disease-Prediction-023060fabd036c705a96b0f2b827b4702dd61fc4/Dataset/val')
test_data_dir = os.path.join(extract_path, 'Plant-Leaf-Disease-Prediction-023060fabd036c705a96b0f2b827b4702dd61fc4/Dataset/test')

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and augment training data
batch_size = 32
image_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 3: Model Building and Training
# Load MobileNetV2 base model
base_model = MobileNetV2(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks
checkpoint = ModelCheckpoint('plant_disease_model.h5', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
epochs = 20

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint, early_stopping]
)

# Step 4: Model Evaluation
# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 5: Inference
# You can use the trained model for inference on new plant images
# For example:
# new_image_path = 'path_to_new_image.jpg'
# load and preprocess the new image
# predictions = model.predict(new_image_preprocessed)
# You can interpret the predictions to detect plant diseases
