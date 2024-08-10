## Diabetes Prediction using Neural Networks

This repository contains code for training and evaluating neural network models to predict diabetes based on the PIMA Indian Diabetes dataset. The models were trained with varying parameters, such as epochs, batch size, and the use of early stopping, to determine the most effective approach for accurate prediction.


## Project Overview

The primary goal of this project is to build a neural network model that can predict the likelihood of diabetes in a patient based on medical and demographic data. The dataset used is the PIMA Indian Diabetes dataset, which includes features such as glucose levels, blood pressure, BMI, and more.


## Table of Contents
Dataset
Models
Requirements
Usage
Result

## Dataset
The PIMA Indian Diabetes dataset is used in this project. It is publicly available and consists of the following features:

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skinfold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1) indicating the presence of diabetes

## Models
Four neural network models were trained with different parameters:

Model 1:

Epochs: 2500
Batch Size: 15

Model 2:

Epochs: 1500
Batch Size: 10

Model 3:

Epochs: 1200
Batch Size: 10

Model 4:

Epochs: 1200
Batch Size: 10
Early Stopping: Enabled

## Requirements
To run the code, you need the following dependencies:

Python 3.x
NumPy
Pandas
TensorFlow
Keras
Scikit-learn
Matplotlib (optional for plotting)
You can install the required libraries using the following command:

pip install -r requirements.txt

## Usage
Clone the repository:
git clone link 
cd diabetes-prediction

Install the required dependencies:
pip install -r requirements.txt

Run the code in ipynb file cell wise, it will save the model1 (model in github code) and model4 (contributed code), which can be used directly after loading it. 
The code will also output the test accuracy and loss for each model.

## Results
The best-performing model was Model 4, which achieved:

Test Loss: 0.1754
Test Accuracy: 75.49%
This model used early stopping, which helped prevent overfitting and resulted in the best performance among the four models.