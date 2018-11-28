# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:09:24 2018

@author: user
"""
from art.classifiers import KerasClassifier
import numpy as np

def adversarial_training(x_train_adv, 
                         x_test_adv,
                         x_train, 
                         y_train,
                         x_test, 
                         y_test, 
                         model,
                         min_,
                         max_,
                         out_file):
    ## Do adversarial training
    # Data augmentation: expand the training set with the adversarial samples
    x_train = np.append(x_train, x_train_adv, axis=0)
    y_train = np.append(y_train, y_train, axis=0)
    
    # Retrain the CNN on the extended dataset
    classifier = KerasClassifier((min_, max_), model=model)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)
    
    ## After adversarial training
    with open(out_file, 'a+') as f:
        preds = np.argmax(classifier.predict(x_train), axis=1)
        acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
        print("TRAIN: %.2f%% \n" % (acc * 100), file=f)
        
        preds = np.argmax(classifier.predict(x_train_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
        print("TRAIN-ADVERSARY: %.2f%% \n" % (acc * 100), file=f)
        
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print("TEST: %.2f%% \n" % (acc * 100), file=f)
        
        preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
        print('TEST-ADVERSARY: %.2f%% \n' % (acc * 100), file=f)
