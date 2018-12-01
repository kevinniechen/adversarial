# -*- coding: utf-8 -*-
"""
evaluate
"""
import numpy as np

def evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, y_train_adv, y_test_adv, classifier, file):
    preds = np.argmax(classifier.predict(x_train), axis=1)
    acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
    print("TRAIN: %.2f%% \n" % (acc * 100))
    file.write("  TRAIN: %.2f%% \n" % (acc * 100))
    
    preds = np.argmax(classifier.predict(x_train_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_train_adv, axis=1)) / y_train_adv.shape[0]
    print("TRAIN-ADVERSARY: %.2f%% \n" % (acc * 100))
    file.write("  TRAIN-ADVERSARY: %.2f%% \n" % (acc * 100))
    
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("TEST: %.2f%% \n" % (acc * 100))
    file.write("  TEST: %.2f%% \n" % (acc * 100))
    
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test_adv, axis=1)) / y_test_adv.shape[0]
    print("TEST-ADVERSARY: %.2f%% \n" % (acc * 100))
    file.write("  TEST-ADVERSARY: %.2f%% \n" % (acc * 100))
    