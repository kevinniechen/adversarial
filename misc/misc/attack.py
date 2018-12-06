# -*- coding: utf-8 -*-

"""
CMSC 727 Final Project
Neural Network: simple 2 hidden layer 64 nodes neural network
Toolbox: IBM
Attack: CarliniL2Method
Defense: LabelSmoothing
"""


from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import abspath
import sys
sys.path.append(abspath('.'))

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from art.attacks import CarliniL2Method
from art.defences import LabelSmoothing
from art.classifiers import KerasClassifier
from art.utils import load_mnist_raw, preprocess, random_targets


# Read MNIST dataset (x_raw contains the original images):
(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist_raw()
x_train, y_train = preprocess(x_raw, y_raw)
x_test, y_test = preprocess(x_raw_test, y_raw_test)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


## Create simple 2 hidden layer 64 nodes neural network
model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
classifier = KerasClassifier((min_, max_), model=model)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)


## Create CarliniL2Method adversarial samples
print('Create CarliniL2Method attack \n')
adv_crafter = CarliniL2Method(classifier, targeted=True, max_iter=100, binary_search_steps=1, learning_rate=1, initial_const=10, decay=0)
params = {'y': random_targets(y_test, classifier.nb_classes)}
x_test_adv = adv_crafter.generate(x_test, **params)


print("Before LabelSmoothing\n")
# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on legit samples: %.2f%% \n" % (acc * 100))
# Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print('Accuracy on adversarial samples: %.2f%% \n' % (acc * 100))

from defense import Defense

print("Starting defense.")
d = Defense(x_train_adv, x_test_adv)
d.adversarial_training()

## LabelSmoothing
labelsmoother = LabelSmoothing()
x_train, y_train = labelsmoother(x_train, y_train, max_value=.8)
x_test, y_test = labelsmoother(x_test, y_test, max_value=.8)

## Create simple 2 hidden layer 64 nodes neural network
model = Sequential()
model.add(Dense(64, input_dim=784, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
classifier = KerasClassifier((min_, max_), model=model)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)


print("After LabelSmoothing \n")
# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Accuracy on legit samples: %.2f%% \n" % (acc * 100))
# Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print('Accuracy on adversarial samples: %.2f%% \n' % (acc * 100))



