# -*- coding: utf-8 -*-
"""
Defense
"""

import numpy as np
from art.defences import SpatialSmoothing, FeatureSqueezing, GaussianAugmentation, LabelSmoothing
from art.utils import preprocess

from create_Neural_Network import create_Neural_Network
from evaluate import evaluate

def def_SpatialSmoothing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_):
    # reshape to smooth
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_train_adv = x_train_adv.reshape(60000, 28, 28, 1)
    x_test_adv = x_test_adv.reshape(10000, 28, 28, 1)
    # smooth
    smoother = SpatialSmoothing()
    x_train_smooth = smoother(x_train, window_size=3)
    x_test_smooth = smoother(x_test, window_size=3)
    x_train_adv_smooth = smoother(x_train_adv, window_size=3)
    x_test_adv_smooth = smoother(x_test_adv, window_size=3)
    # reshape back
    x_train_smooth = x_train_smooth.reshape(60000, 784)
    x_test_smooth = x_test_smooth.reshape(10000, 784)
    x_train_adv_smooth = x_train_adv_smooth.reshape(60000, 784)
    x_test_adv_smooth = x_test_adv_smooth.reshape(10000, 784)
    
    # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_smooth, y_train, nb_epochs=5, batch_size=50)
    
    # print result
    print("After SpatialSmoothing Defense\n")
    evaluate(x_train_smooth, x_test_smooth, y_train, y_test, x_train_adv_smooth, x_test_adv_smooth, classifier)
    
    
def def_AdversarialTraining(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_):
    # expand the training set with the adversarial samples
    x_train_aug = np.append(x_train, x_train_adv, axis=0)
    y_train_aug = np.append(y_train, y_train, axis=0)
    
    # retrain the Network on the extended dataset
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_aug, y_train_aug, nb_epochs=5, batch_size=50)
    
    # print result
    print("After Defense\n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    
    
def def_FeatureSqueezing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_):
    squeezer = FeatureSqueezing()
    x_train_squeeze = squeezer(x_train, bit_depth=2)
    x_test_squeeze = squeezer(x_test, bit_depth=2)
    x_train_adv_squeeze = squeezer(x_train_adv, bit_depth=2)
    x_test_adv_squeeze = squeezer(x_test_adv, bit_depth=2)
    
    # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_squeeze, y_train, nb_epochs=5, batch_size=50)
    
    # print result
    print("After Defense\n")
    evaluate(x_train_squeeze, x_test_squeeze, y_train, y_test, x_train_adv_squeeze, x_test_adv_squeeze, classifier)
    
    
def def_GaussianAugmentation(x_raw, x_raw_test, y_raw, y_raw_test, x_train_adv, x_test_adv, y_train, y_test, min_, max_):
    ga = GaussianAugmentation(sigma=150)
    x_train_aug, y_train_aug = ga(x_raw, y_raw)
    x_test_aug, y_test_aug = ga(x_raw_test, y_raw_test)
    x_train_aug, y_train_aug = preprocess(x_train_aug, y_train_aug)
    x_test_aug, y_test_aug = preprocess(x_test_aug, y_test_aug)
    x_train_aug = x_train_aug.reshape(120000, 784)
    x_test_aug = x_test_aug.reshape(20000, 784)
    
    # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_aug, y_train_aug, nb_epochs=5, batch_size=50)
    
    # print result
    print("After Defense\n")
    preds = np.argmax(classifier.predict(x_train_aug), axis=1)
    acc = np.sum(preds == np.argmax(y_train_aug, axis=1)) / y_train_aug.shape[0]
    print("TRAIN: %.2f%% \n" % (acc * 100))
    preds = np.argmax(classifier.predict(x_train_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_train, axis=1)) / y_train.shape[0]
    print("TRAIN-ADVERSARY: %.2f%% \n" % (acc * 100))
    preds = np.argmax(classifier.predict(x_test_aug), axis=1)
    acc = np.sum(preds == np.argmax(y_test_aug, axis=1)) / y_test_aug.shape[0]
    print("TEST: %.2f%% \n" % (acc * 100))
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print('TEST-ADVERSARY: %.2f%% \n' % (acc * 100))
    
def def_LabelSmoothing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_):
    labelsmoother = LabelSmoothing()
    x_train, y_train_smooth = labelsmoother(x_train, y_train, max_value=.8)
    x_test, y_test_smooth = labelsmoother(x_test, y_test, max_value=.8)
    
     # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train_smooth, nb_epochs=5, batch_size=50)
    
    # print result
    print("After Defense\n")
    evaluate(x_train, x_test, y_train_smooth, y_test_smooth, x_train_adv, x_test_adv, classifier)   
    