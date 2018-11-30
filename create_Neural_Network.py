# -*- coding: utf-8 -*-
"""
Create Neural Nework
"""

from keras.models import Sequential
from keras.layers import Dense
from art.classifiers import KerasClassifier


def create_Neural_Network(min_, max_):
    ## Create simple 2 hidden layer 64 nodes neural network
    model = Sequential()
    model.add(Dense(64, input_dim=784, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    classifier = KerasClassifier((min_, max_), model=model)
    return classifier
    
    