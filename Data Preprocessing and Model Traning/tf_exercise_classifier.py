"""
    #############################################################
    --- Preprocess and prepare data ---
    #############################################################

    Snippet to import IMU data from csv files and input it to the training tensors. 
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os

print(f"TensorFlow version = {tf.__version__}\n")

# Set a fixed random seed value for reproducibility
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

# List of gestures for which data is available
GESTURES = [
    "name_first_gesture",
    "name_second_gesture",
    #...
]

# Directories containing training data
directories = [
    "/directory_1_name/",
    "/directory_2_name/",
    #...
]

# Define constants for the number of samples per gesture and the window size for the moving average filter
SAMPLES_PER_GESTURE = 60
NUM_GESTURES = len(GESTURES)
window_size = 10

# Function to apply a moving average filter to the data
def moving_average(data, window_size):
    data_filtered = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        data_filtered[i] = np.mean(data[start:end])
    return data_filtered

# Create a one-hot encoded matrix for the output labels
ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)

# Initialize lists to store input and output data
inputs = []
outputs = []

# Initialize dataframes for each gesture
dfs = [pd.DataFrame() for _ in range(NUM_GESTURES)]

# Load data from CSV files into the dataframes
for directory in directories:
    for gesture_index in range(NUM_GESTURES):
        gesture = GESTURES[gesture_index]
        file_path = os.path.join(directory, gesture + ".csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs[gesture_index] = pd.concat([dfs[gesture_index], df], ignore_index=True)

# Process each gesture data and prepare inputs and outputs
for gesture_index in range(NUM_GESTURES):
    gesture = GESTURES[gesture_index]
    print(f"Processing index {gesture_index} for gesture '{gesture}'.")

    # One-hot encoded output for the current gesture
    output = ONE_HOT_ENCODED_GESTURES[gesture_index]

    df = dfs[gesture_index]

    # Calculate the number of gesture recordings in the file
    num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)

    print(f"\tThere are {num_recordings} recordings of the {gesture} gesture.")

    # Apply moving average filter to the data
    for i in range(num_recordings):
        start_index = i * SAMPLES_PER_GESTURE
        end_index = (i + 1) * SAMPLES_PER_GESTURE
        df['aX'][start_index:end_index] = moving_average(df['aX'][start_index:end_index], window_size)
        df['aY'][start_index:end_index] = moving_average(df['aY'][start_index:end_index], window_size)
        df['aZ'][start_index:end_index] = moving_average(df['aZ'][start_index:end_index], window_size)
        df['gX'][start_index:end_index] = moving_average(df['gX'][start_index:end_index], window_size)
        df['gY'][start_index:end_index] = moving_average(df['gY'][start_index:end_index], window_size)
        df['gZ'][start_index:end_index] = moving_average(df['gZ'][start_index:end_index], window_size)
        df['mX'][start_index:end_index] = moving_average(df['mX'][start_index:end_index], window_size)
        df['mY'][start_index:end_index] = moving_average(df['mY'][start_index:end_index], window_size)
        df['mZ'][start_index:end_index] = moving_average(df['mZ'][start_index:end_index], window_size)

    # Normalize and prepare the data for each recording
    for i in range(num_recordings):
        tensor = []

        for j in range(SAMPLES_PER_GESTURE):
            index = i * SAMPLES_PER_GESTURE + j
            # Normalize the input data between 0 to 1
            tensor += [
                (df['aX'][index] + 4) / 8,
                (df['aY'][index] + 4) / 8,
                (df['aZ'][index] + 4) / 8,
                (df['gX'][index] + 2000) / 4000,
                (df['gY'][index] + 2000) / 4000,
                (df['gZ'][index] + 2000) / 4000,
                (df['mX'][index] + 400) / 800,
                (df['mY'][index] + 400) / 800,
                (df['mZ'][index] + 400) / 800
            ]

        inputs.append(tensor)
        outputs.append(output)

# Convert the lists to numpy arrays
inputs = np.array(inputs)
outputs = np.array(outputs)

print("Data set parsing and preparation complete.")

# Randomize the order of the inputs for even distribution during training, testing, and validation
num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)

inputs = inputs[randomize]
outputs = outputs[randomize]

# Split the data into training, testing, and validation sets
TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

print("Data set randomization and splitting complete.")


"""
    #############################################################
    --- Plot Data Distribution ---
    #############################################################
    
    Snippet to plot the data distribution of the previously prepared dataset. 
"""

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Perform PCA to reduce dimensionality for visualization
pca = PCA(n_components=3)
inputs_pca = pca.fit_transform(inputs)

# Plot the PCA results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(NUM_GESTURES):
    ax.scatter(inputs_pca[outputs[:, i] == 1, 0], inputs_pca[outputs[:, i] == 1, 1], inputs_pca[outputs[:, i] == 1, 2], label=GESTURES[i])
ax.set_title('PCA Visualization of Features Distribution')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()
plt.show()


"""
    #############################################################
    --- Train Model ---
    #############################################################
    
    Snippet to tune and train the machine learning model with the previously prepared dataset. 
"""

# Build the neural network model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(15, activation='relu'))
model.add(tf.keras.layers.Dense(NUM_GESTURES, activation='softmax'))  # Softmax is used as we expect only one gesture per input

# Compile the model
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'accuracy'])

# Train the model
history = model.fit(inputs_train, outputs_train, epochs=200, batch_size=1, validation_data=(inputs_validate, outputs_validate))


"""
    #############################################################
    --- Plot Loss ---
    #############################################################
    
    Snippet to plot the training and validation loss during the previous train process. 
"""

plt.rcParams["figure.figsize"] = (20, 10)

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(plt.rcParams["figure.figsize"])


"""
    #############################################################
    --- Plot Confusion Matrix ---
    #############################################################
    
    Snippet to plot plot the confusion matrix with the results of inferencing the model with the test dataset. 
"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Get predictions for the test set
predictions = model.predict(inputs_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(outputs_test, axis=1)

# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=GESTURES, yticklabels=GESTURES)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Display metrics
plt.figtext(0.5, -0.15, f'Accuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}', ha='center', fontsize=12)

plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


"""
    #############################################################
    --- TensorFlow Lite Conversion and Model to Byte Array ---
    #############################################################
    
    Snippet to compile the machine learning model to TensorFlow Lite format and export it into byte array. 
"""

# Convert the Keras model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
open("gesture_model.tflite", "wb").write(tflite_model)

# Generate the C header file from the TensorFlow Lite model
!echo "const unsigned char model[] = {" > /content/model.h
!cat gesture_model.tflite | xxd -i >> /content/model.h
!echo "};" >> /content/model.h

# Check the size of the model files
import os
basic_model_size = os.path.getsize("gesture_model.tflite")
print("Model is %d bytes" % basic_model_size)

model_h_size = os.path.getsize("model.h")
print(f"Header file, model.h, is {model_h_size:,} bytes.")
