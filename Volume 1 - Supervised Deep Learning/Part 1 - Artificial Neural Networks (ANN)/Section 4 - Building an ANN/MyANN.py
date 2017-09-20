# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 18:05:23 2017

@author: aakash.chotrani
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


#Take care of categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing ANN as sequence of layers
classifier = Sequential()

#Adding Input Layer and First hidden layer
#Step 1: Randomly initialize the weights to small numbers close to 0 but not 0
#classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu',input_dim = 11))
#first hidden layer with 6 output and 11 inputs. Activation is rectified linear and initialized weight with uniform distribution
classifier.add(Dense(6,activation = 'relu',kernel_initializer = 'uniform',input_shape=(11,)))

#Adding the second hidden layer
classifier.add(Dense(6,activation = 'relu',kernel_initializer = 'uniform'))

#Adding the output layer, changing the output dimension and the activation to sigmoid
classifier.add(Dense(1,activation = 'sigmoid',kernel_initializer = 'uniform'))

#compiling the ANN, choosing stochastic gradient descent algorithm = 'adam', it works on the loss function and how to evaluate the performance of the model.
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics= ['accuracy'])

#Fitting Ann to the training set results
classifier.fit(X_train,y_train,batch_size = 10, epochs = 100)



# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)