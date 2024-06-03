# TinyML Gym Exercise Classification Thesis

This repository holds the software code for the thesis on: CLASSIFICATION OF GYM EXERCISES USING TINY MACHINE LEARNING

The code is devided in three blocks: 
- Python Code to preprocess data and train the model.
- Arduino Code, both for data collection and model inference.
- AppInventor application design and programming in .aia format. 


#Data Preprocessing and Model Training

This folder contains two Python Codes. Both were designed using Colab environment, that's why they are devided in blocks. The first Python file is for plotting and splitting csv formatted 9-axis IMU data. The second file is for preparing the training data, traing the model, analyze its performance and deploy it into a byte arryay. 


#Data Collection and Inference

This folder contrains two Arduino Ide projects. The first one is a code to collect IMU exercise data. The second is to deploy the final application of the project; it uses the compiled TensorFlow model trained throught the previows Python code. 


#Smartphone Application

This folder contains the App Inventor Project used to develop the Android Application for the user interface. The file with .aia format has the project compiled, which can be opened through App Inventor. 
