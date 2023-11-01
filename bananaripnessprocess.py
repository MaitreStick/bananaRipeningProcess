# -*- coding: utf-8 -*-
# Importing necessary classes and modules
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import pandas as pd
import numpy as np
import random
import glob
import zipfile
import shutil
import csv
from PIL import Image

# Reading a CSV file from a specified path in Google Drive using pandas
df = pd.read_csv("/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/train_classes.csv", index_col=0)

# Printing the shape of the DataFrame
print(df.shape)

# Printing the columns of the DataFrame
print(df.columns)

# Displaying the first few rows of the DataFrame
df.head()

# Path to the directory containing images
image_path = "/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/train"

# Variable to count the number of skipped images
num_skipped = 0

# Loop through each file in the directory
for fname in os.listdir(image_path):
    # Get the complete path of the file
    fpath = os.path.join(image_path, fname)
    try:
        # Open the file
        fobj = open(fpath, "rb")
        # Check if the file is a JPEG image
        is_jpg = tf.compat.as_bytes("JFIF") in fobj.peek(10)
    finally:
        # Close the file
        fobj.close()

    # If the file is not a JPEG image, delete it and update the count
    if not is_jpg:
        num_skipped += 1
        os.remove(fpath)

# Print the number of deleted images
print("Deleted %d images" % num_skipped)

# Defining image size
image_size = (224, 224)

# Defining batch size for training
batch_size = 16

# Number of classes in the dataset
num_classes = 5

# Number of channels in the image (e.g., 3 for RGB images)
num_channels = 3

# Creating input shape for the model
input_shape = image_size + (num_channels,)

# Parameters for image data generator
params = {'dim': image_size,        # Dimension of the images
          'batch_size': batch_size, # Batch size for training
          'n_classes': num_classes,  # Number of classes
          'n_channels': num_channels, # Number of channels
          'shuffle': False}          # Shuffling the data during training

# Directory containing the source images
IMAGE_SOURCE = "/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/train"

# Directory where images will be split into classes
SPLIT_DIR = "/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/classes"

# Read the CSV file containing class labels
df = pd.read_csv("/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/train_classes.csv")

# Extract the class label values
values = df[[' freshripe', ' freshunripe', ' overripe', ' ripe', ' rotten', ' unripe']]

# Determine the dominant class label for each image
labels = values.idxmax(axis=1)

# Print the values and labels of the first few rows
print(values)
print(labels[:5])

# Get the unique classes
classes = labels.unique()

# Create directories for each class if they don't exist
for class_name in classes:
    os.makedirs(os.path.join(SPLIT_DIR, class_name), exist_ok=True)

# Copy images to the respective directories based on their class
for index, row in df.iterrows():
    image_name = row["filename"]
    class_name = labels[index]
    shutil.copy(os.path.join(IMAGE_SOURCE, image_name), os.path.join(SPLIT_DIR, class_name, image_name))

# Path to the folder containing categorized images
image_folder = "/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/classes"

# Paths to specific folders for each class
freshripe_folder = os.path.join(image_folder, " freshripe")
freshunripe_folder = os.path.join(image_folder, " freshunripe")
overripe_folder = os.path.join(image_folder, " overripe")
ripe_folder = os.path.join(image_folder, " ripe")
rotten_folder = os.path.join(image_folder, " rotten")

# File paths to each class folder
freshripe_path = os.path.join(freshripe_folder, "*.jpg")
freshunripe_path = os.path.join(freshunripe_folder, "*.jpg")
overripe_path = os.path.join(overripe_folder, "*.jpg")
ripe_path = os.path.join(ripe_folder, "*.jpg")
rotten_path = os.path.join(rotten_folder, "*.jpg")

# Retrieving filenames using glob for each class
freshripe_filenames = glob.glob(freshripe_path)
freshunripe_filenames = glob.glob(freshunripe_path)
overripe_filenames = glob.glob(overripe_path)
ripe_filenames = glob.glob(ripe_path)
rotten_filenames = glob.glob(rotten_path)

# Printing the number of filenames in each class
print(len(freshripe_filenames), len(freshunripe_filenames), len(overripe_filenames), len(ripe_filenames), len(rotten_filenames))

# Dictionary containing the sizes of each class
class_sizes = {
    "freshripe": len(freshripe_filenames),
    "freshunripe": len(freshunripe_filenames),
    "overripe": len(overripe_filenames),
    "ripe": len(ripe_filenames),
    "rotten": len(rotten_filenames)
}

# Finding the class with the minimum number of samples
class_name = min(class_sizes, key=class_sizes.get)

# Getting the number of samples in the smallest class
min_samples = class_sizes[class_name]

# Printing the minimum number of samples
print(min_samples)

# Creating separate dataframes for each class
df_freshripe = pd.DataFrame()
df_freshunripe = pd.DataFrame()
df_overripe = pd.DataFrame()
df_ripe = pd.DataFrame()
df_rotten = pd.DataFrame()

# Trimming the image folder path from the filenames
for i in range(len(freshripe_filenames)):
    freshripe_filenames[i] = freshripe_filenames[i][len(image_folder)+1:]
for i in range(len(freshunripe_filenames)):
    freshunripe_filenames[i] = freshunripe_filenames[i][len(image_folder)+1:]
for i in range(len(overripe_filenames)):
    overripe_filenames[i] = overripe_filenames[i][len(image_folder)+1:]
for i in range(len(ripe_filenames)):
    ripe_filenames[i] = ripe_filenames[i][len(image_folder)+1:]
for i in range(len(rotten_filenames)):
    rotten_filenames[i] = rotten_filenames[i][len(image_folder)+1:]

# Shuffling the filenames within each class
random.shuffle(freshripe_filenames)
random.shuffle(freshunripe_filenames)
random.shuffle(overripe_filenames)
random.shuffle(ripe_filenames)
random.shuffle(rotten_filenames)

# Assigning filenames and labels to each class's dataframe
df_freshripe["filename"] = freshripe_filenames[:min_samples]
df_freshripe["label"] = 0

df_freshunripe["filename"] = freshunripe_filenames[:min_samples]
df_freshunripe["label"] = 1

df_overripe["filename"] = overripe_filenames[:min_samples]
df_overripe["label"] = 2

df_ripe["filename"] = ripe_filenames[:min_samples]
df_ripe["label"] = 3

df_rotten["filename"] = rotten_filenames[:min_samples]
df_rotten["label"] = 4

# Printing the lengths and first few rows of each dataframe
print(len(df_freshripe), len(df_overripe), len(df_ripe), len(df_rotten))
print(df_freshripe.head(5))
print(df_freshunripe.head(5))
print(df_overripe.head(5))
print(df_ripe.head(5))
print(df_rotten.head(5))

# Number of iterations for dataset splitting
num_iterations = 10

# Distribution of data among train, validation, and test sets
distribution = {"train": 0.70, "val": 0.15, "test": 0.15}

# Calculating lengths for each split
len_train = int(min_samples * distribution["train"])
len_val = int(min_samples * distribution["val"])
len_test = int(min_samples * distribution["test"])

# Iterating over the dataset splits
for i in range(num_iterations):

  # Generating the training set
  start = i * len_val
  end = start + len_train
  df_train = pd.concat([
      df_freshripe[start:end],
      df_freshunripe[start:end],
      df_overripe[start:end],
      df_ripe[start:end],
      df_rotten[start:end]
  ])

  # Checking if the training set size is smaller than the expected size
  if len(df_train) < (len_train*num_classes)-1:
    start = 0
    end = len_train - int(len(df_train)/ num_classes)
    df_train2 = pd.concat([
      df_freshripe[start:end],
      df_freshunripe[start:end],
      df_overripe[start:end],
      df_ripe[start:end],
      df_rotten[start:end]
    ])

    df_train = pd.concat([df_train, df_train2])

  # Printing information about the training set
  print(i, start, end, len(df_train))

  # Generating the validation set
  start = end
  if start >= min_samples-1:
    start = 0
  end = start + len_val
  df_val = pd.concat([
      df_freshripe[start:end],
      df_freshunripe[start:end],
      df_overripe[start:end],
      df_ripe[start:end],
      df_rotten[start:end]
  ])
  print(i, start, end, len(df_val))

  # Generating the test set
  start = end
  if start >= min_samples-1:
    start = 0
  end = start + len_test
  df_test = pd.concat([
      df_freshripe[start:end],
      df_freshunripe[start:end],
      df_overripe[start:end],
      df_ripe[start:end],
      df_rotten[start:end]
  ])
  print(i, start, end, len(df_test))

  # Creating filenames for the splits
  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  # Shuffling the dataframes
  df_train = df_train.sample(frac=1)
  df_val = df_val.sample(frac=1)
  df_test = df_test.sample(frac=1)

  # Saving the dataframes to CSV files
  df_train.to_csv(train_filename)
  df_val.to_csv(val_filename)
  df_test.to_csv(test_filename)

  # Printing a separator line
  print("-"*60)

for i in range(num_iterations):
  train_filename = "train_ds_" + str(i) + ".csv"
  val_filename = "val_ds_" + str(i) + ".csv"
  test_filename = "test_ds_" + str(i) + ".csv"

  df_train = pd.read_csv(train_filename)
  df_val = pd.read_csv(val_filename)
  df_test = pd.read_csv(test_filename)

  print(df_train.groupby(["label"])["label"].count())
  print(df_val.groupby(["label"])["label"].count())
  print(df_test.groupby(["label"])["label"].count())
  print("-"*60)
  print()

# Initializing the iteration number
i = 0

# Creating filenames for the datasets
train_filename = "train_ds_" + str(i) + ".csv"
val_filename = "val_ds_" + str(i) + ".csv"
test_filename = "test_ds_" + str(i) + ".csv"

# Reading the data from the CSV files into dataframes
df_train = pd.read_csv(train_filename)
df_val = pd.read_csv(val_filename)
df_test = pd.read_csv(test_filename)

# Creating an empty dictionary to store the partitions
partition = {}

# Storing the filenames for the training, validation, and test sets
partition["train"] = list(df_train["filename"])
partition["val"] = list(df_val["filename"])
partition["test"] = list(df_test["filename"])

# Printing the dictionary containing the partitions
print(partition)

# Creating an empty dictionary to store the filename-label mapping
labels = {}

# Combining all dataframes into a single dataframe
df_all = pd.concat([df_train, df_val, df_test])

# Iterating through the combined dataframe to populate the 'labels' dictionary
for index, row in df_all.iterrows():
    filename = row["filename"]
    label = row["label"]
    labels[filename] = label

# Printing the dictionary containing the filename-label mapping
print(labels)

def get_image(image_filename):
    # Opens an image in RGB mode
    im1 = Image.open(image_filename).convert("RGB")

    # Resizes the image to the specified image_size
    im1 = im1.resize(image_size)

    # Prints the type of the image
    print(type(im1))

    # Converts the image to a numpy array
    image = np.asarray(im1)

    # Converts the image array to type 'float32'
    image = np.array(image, dtype='float32')

    # Normalizes the pixel values to be in the range [0, 1]
    image = image / 255.0

    # Prints the shape of the resulting image
    print(image.shape)

    # Returns the normalized image
    return image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, batch_size=128, dim=(180,180), n_channels=3,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            image_filename = os.path.join(image_folder, ID)
            X[i,] = get_image(image_filename)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

# Creating a DataGenerator instance for the training set
train_generator = DataGenerator(partition['train'], labels, **params)

# Creating a DataGenerator instance for the validation set
val_generator = DataGenerator(partition['val'], labels, **params)

# Creating a DataGenerator instance for the test set
test_generator = DataGenerator(partition['test'], labels, **params)

# Number of epochs for training
epochs = 3

# Setting up the callback for early stopping
callbacks = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=3)
]

# Installing the keras-tuner package for hyperparameter tuning
!pip install -q -U keras-tuner

# Importing the keras_tuner library
import keras_tuner as kt

def model_builder(hp):
    # Choosing the model type
    hp_model_type = hp.Choice(
        "model_type", ["EfficientNetV2L", "ResNet152V2", "DenseNet201"], default="EfficientNetV2L"
    )

    # Choosing the learning rate
    hp_learning_rate = hp.Choice(
        'learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5]
    )

    # Choosing the optimizer
    hp_optimizer = hp.Choice(
        "optimizer", values=["Adadelta", "RMSprop", "Adam"], default="Adam"
    )

    # Configuring the model based on the chosen hyperparameters
    with hp.conditional_scope("model_type", ["EfficientNetV2L"]):
        if hp_model_type == "EfficientNetV2L":
            model = tf.keras.applications.EfficientNetV2L(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation="softmax",
            )

    with hp.conditional_scope("model_type", ["ResNet152V2"]):
        if hp_model_type == "ResNet152V2":
            model = tf.keras.applications.ResNet152V2(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation="softmax",
            )

    with hp.conditional_scope("model_type", ["DenseNet201"]):
        if hp_model_type == "DenseNet201":
            model = tf.keras.applications.DenseNet201(
                include_top=True,
                weights=None,
                input_shape=input_shape,
                classes=num_classes,
                classifier_activation="softmax",
            )

    with hp.conditional_scope("optimizer", ["Adadelta"]):
        if hp_optimizer == "Adadelta":
            optimizer = keras.optimizers.Adadelta(learning_rate=hp_learning_rate)

    with hp.conditional_scope("optimizer", ["RMSprop"]):
        if hp_optimizer == "RMSprop":
            optimizer = keras.optimizers.RMSprop(learning_rate=hp_learning_rate)

    with hp.conditional_scope("optimizer", ["Adam"]):
        if hp_optimizer == "Adam":
            optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Compiling the model with the chosen optimizer, loss function, and metrics
    model.compile(
        optimizer=hp_optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# Initializing a Hyperband tuner
tuner = kt.Hyperband(
    model_builder,  # The model_builder function defined earlier
    objective='val_accuracy',  # The objective to optimize for
    max_epochs=100,  # The maximum number of epochs to train the model
    factor=3,  # Reduction factor for the number of epochs and number of models
    overwrite=True,  # Whether to overwrite the results of a previous run
    directory='my_dir',  # Directory to store the tuning results
    project_name='intro_to_kt'  # Name of the project
)

# Initiating the hyperparameter search
tuner.search(
    train_generator,  # Data generator for training data
    epochs=epochs,  # Number of epochs for training
    callbacks=callbacks,  # Callbacks for monitoring the training process
    validation_data=val_generator,  # Data generator for validation data
)

# Retrieving the best hyperparameters from the completed tuner
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Printing the optimal learning rate for the optimizer
print(f"""
The hyperparameter search is complete. The optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Building the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Compiling the model with the specified configuration
model = tf.keras.applications.DenseNet201(
          include_top=True,
          weights=None,
          input_shape=input_shape,
          classes=num_classes,
          classifier_activation="softmax",
      )

model.compile(
  optimizer=keras.optimizers.Adadelta(learning_rate=1e-03),
  loss="categorical_crossentropy",
  metrics=["accuracy"],
)

# Training the model with the data generators and hyperparameters
history = model.fit(
    train_generator,
    epochs=100,
    callbacks=callbacks,
    validation_data=val_generator,
    initial_epoch=0
)

# Obtaining the best epoch based on validation accuracy
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# Defining the path for saving the trained model
model_save_path = '/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/model2.h5'

# Saving the model in .h5 format
model.save(model_save_path)

# Also saving the model in .keras format
model.save('/content/drive/MyDrive/artificial_intelligence/Banana_Ripening_Process/model2.keras')

# Printing a message to confirm that the model has been saved
print(f'Model saved in .keras and .h5')

# Importing the necessary library for visualization
import matplotlib.pyplot as plt

# Setting the label for the x-axis
plt.xlabel("# epoch")

# Setting the label for the y-axis
plt.ylabel("Loss Magnitude")

# Plotting the loss over the epochs
plt.plot(history.history["loss"])