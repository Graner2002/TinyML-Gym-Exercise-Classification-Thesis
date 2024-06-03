# --- Plot Data ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to apply a moving average filter to data
def moving_average(data, window_size=10):
    data_filtered = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        data_filtered[i] = np.mean(data[start:end])
    return data_filtered

# Function to plot gyroscope, accelerometer, and magnetometer data
def plot_data(df, index, samples_per_gesture, start_gesture, title):
    plt.rcParams["figure.figsize"] = (20, 10)

    # Plot gyroscope data
    plt.plot(index, df['gX'], color='tab:blue', label='gX', linestyle='solid', marker=',')
    plt.plot(index, df['gY'], color='tab:orange', label='gY', linestyle='solid', marker=',')
    plt.plot(index, df['gZ'], color='tab:green', label='gZ', linestyle='solid', marker=',')

    # Plot accelerometer data
    plt.plot(index, df['aX'], color='tab:red', label='aX', linestyle='solid', marker=',')
    plt.plot(index, df['aY'], color='tab:purple', label='aY', linestyle='solid', marker=',')
    plt.plot(index, df['aZ'], color='tab:brown', label='aZ', linestyle='solid', marker=',')

    # Plot magnetometer data
    plt.plot(index, df['mX'], color='tab:pink', label='mX', linestyle='solid', marker=',')
    plt.plot(index, df['mY'], color='tab:gray', label='mY', linestyle='solid', marker=',')
    plt.plot(index, df['mZ'], color='tab:cyan', label='mZ', linestyle='solid', marker=',')

    # Add vertical lines to indicate start of each gesture
    for i in range(1, len(index) // samples_per_gesture):
        plt.axvline(x=start_gesture * samples_per_gesture + i * samples_per_gesture, color='gray', linestyle='--')

    # Set title and labels
    plt.title(title)
    plt.xlabel("Sample #")
    plt.ylabel("Gyroscope (deg/s) / Accelerometer (g) / Magnetometer (uT)")
    plt.legend()
    plt.show()

# Load data from CSV file
filename = "filename.csv"
df = pd.read_csv("/directory_name/" + filename)

# Define constants for data processing
SAMPLES_PER_GESTURE = 60
NUM_OF_GESTURES = 6
start_gesture = 6

# Define range of indices for the current gesture data
start_index = SAMPLES_PER_GESTURE * start_gesture
end_index = SAMPLES_PER_GESTURE * (start_gesture + NUM_OF_GESTURES)
index = range(start_index, end_index)

# Select subset of data for visualization
df_subset = df.iloc[start_index:end_index]

# Apply moving average filter to the data
for column in df.columns:
    df[column] = moving_average(df[column].values)

# Select filtered data for visualization
filtered_data = df.iloc[start_index:end_index]

# Define exercise and plot title
exercise = "Exercise Name"
title = "Non-filtered Gyroscope / Accelerometer / Magnetometer Data for " + exercise

# Plot the data
plot_data(df_subset, index, SAMPLES_PER_GESTURE, start_gesture, title)

# --- Split Data ---

# Load data from CSV file
filename = "filename.csv"

directory = "/directory_name/"

# Read the CSV file
df = pd.read_csv(directory + filename)

# Split the dataframe into blocks of 60 samples
blocks = [df[i:i+60] for i in range(0, len(df), 60)]

# Define the first and second halves
first_half = blocks[::2]
second_half = blocks[1::2]

# Write the first half to a new CSV file
with open(directory + filename.replace(".csv", "_first_half.csv"), "w") as f:
    # Write the header
    f.write(",".join(df.columns) + "\n")
    for block in first_half:
        block.to_csv(f, index=False, header=False)
        f.write("\n")

# Write the second half to a new CSV file
with open(directory + filename.replace(".csv", "_second_half.csv"), "w") as f:
    # Write the header
    f.write(",".join(df.columns) + "\n")
    for block in second_half:
        block.to_csv(f, index=False, header=False)
        f.write("\n")