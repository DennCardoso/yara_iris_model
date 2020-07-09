#!/usr/bin/env python
# coding: utf-8

# Import dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import pickle


class logisticRegModel:
    def __init__(self):
        # Load de dataset 
        self.data = sns.load_dataset('iris')

        # Prepare the training
        # X = feature values, all the columns except the last column
        x = self.data.iloc[:, :-1]

        # y = target values, last column of the data frame
        y = self.data.iloc[:, -1]

        # Split the data into 80% training and 20% testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
        #Train the model
        model = LogisticRegression(max_iter=1000)

        # Training the model
        model.fit(self.x_train, self.y_train)
        pickle.dump(model, open('data/model.pkl', 'wb'))