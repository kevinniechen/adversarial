# -*- coding: utf-8 -*-
"""
Defense
"""

import numpy as np
from art.defences import SpatialSmoothing, FeatureSqueezing, GaussianAugmentation, LabelSmoothing
from art.utils import preprocess

from create_Neural_Network import create_Neural_Network
from evaluate import evaluate

def def_SpatialSmoothing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_, file):
    train_num = 60000
    test_num = 10000
    # reshape to smooth
    x_train = x_train.reshape(train_num, 28, 28, 1)
    x_test = x_test.reshape(test_num, 28, 28, 1)
    x_train_adv = x_train_adv.reshape(5*train_num, 28, 28, 1)
    x_test_adv = x_test_adv.reshape(5*test_num, 28, 28, 1)
    # smooth
    smoother = SpatialSmoothing()
    x_train_smooth = smoother(x_train, window_size=3)
    x_test_smooth = smoother(x_test, window_size=3)
    x_train_adv_smooth = smoother(x_train_adv, window_size=3)
    x_test_adv_smooth = smoother(x_test_adv, window_size=3)
    # reshape back
    x_train_smooth = x_train_smooth.reshape(train_num, 784)
    x_test_smooth = x_test_smooth.reshape(test_num, 784)
    x_train_adv_smooth = x_train_adv_smooth.reshape(5*train_num, 784)
    x_test_adv_smooth = x_test_adv_smooth.reshape(5*test_num, 784)
    
    # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_smooth, y_train, nb_epochs=5, batch_size=50)
    
    # print result
    print("After SpatialSmoothing Defense\n")
    file.write("==== SpatialSmoothing Defense==== \n")
    for k in range (5):
        file.write("==== Attack %i ====\n" % (k))
        evaluate(x_train_smooth, x_test_smooth, y_train, y_test, x_train_adv_smooth[k*train_num:(k+1)*train_num], x_test_adv_smooth[k*test_num:(k+1)*test_num], y_train, y_test, classifier, file)
    
    
    
def def_AdversarialTraining(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_, file):
    # print result
    print("After AdversarialTraining Defense\n")
    file.write("==== AdversarialTraining Defense==== \n")
    train_num = 60000
    test_num = 10000
    for k in range (5):
        # expand the training set with the adversarial samples
        x_train_aug = np.append(x_train, x_train_adv[k*train_num:(k+1)*train_num], axis=0)
        y_train_aug = np.append(y_train, y_train, axis=0)
        # retrain the Network on the extended dataset
        classifier = create_Neural_Network(min_, max_)
        classifier.fit(x_train_aug, y_train_aug, nb_epochs=5, batch_size=50)
        
        file.write("==== Attack %i ====\n" % (k))
        evaluate(x_train, x_test, y_train, y_test, x_train_adv[k*train_num:(k+1)*train_num], x_test_adv[k*test_num:(k+1)*test_num], y_train, y_test, classifier, file)
    
    
def def_FeatureSqueezing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_, file):
    squeezer = FeatureSqueezing()
    x_train_squeeze = squeezer(x_train, bit_depth=2)
    x_test_squeeze = squeezer(x_test, bit_depth=2)
    x_train_adv_squeeze = squeezer(x_train_adv, bit_depth=2)
    x_test_adv_squeeze = squeezer(x_test_adv, bit_depth=2)
    
    # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_squeeze, y_train, nb_epochs=5, batch_size=50)
    
    # print result
    print("After FeatureSqueezing Defense\n")
    file.write("==== FeatureSqueezing Defense==== \n")
    train_num = 60000
    test_num = 10000
    for k in range (5):
        file.write("==== Attack %i ====\n" % (k))
        evaluate(x_train_squeeze, x_test_squeeze, y_train, y_test, x_train_adv_squeeze[k*train_num:(k+1)*train_num], x_test_adv_squeeze[k*test_num:(k+1)*test_num], y_train, y_test, classifier, file)
    
    
def def_GaussianAugmentation(x_raw, x_raw_test, y_raw, y_raw_test, x_train_adv, x_test_adv, y_train, y_test, min_, max_, file):
    train_num = 60000
    test_num = 10000
    # gaussian augmentation
    ga = GaussianAugmentation(sigma=150)
    x_train_aug, y_train_aug = ga(x_raw, y_raw)
    x_test_aug, y_test_aug = ga(x_raw_test, y_raw_test)
    x_train_aug, y_train_aug = preprocess(x_train_aug, y_train_aug)
    x_test_aug, y_test_aug = preprocess(x_test_aug, y_test_aug)
    x_train_aug = x_train_aug.reshape(2*train_num, 784)
    x_test_aug = x_test_aug.reshape(2*test_num, 784)
    
    # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train_aug, y_train_aug, nb_epochs=5, batch_size=50)
    
    # print result
    print("After GaussianAugmentation Defense\n")
    file.write("==== GaussianAugmentation Defense==== \n")
    for k in range (5):
        file.write("==== Attack %i ====\n" % (k))
        evaluate(x_train_aug, x_test_aug, y_train_aug, y_test_aug, x_train_adv[k*train_num:(k+1)*train_num], x_test_adv[k*test_num:(k+1)*test_num], y_train, y_test, classifier, file)
        
    
def def_LabelSmoothing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_, file):
    labelsmoother = LabelSmoothing()
    x_train, y_train_smooth = labelsmoother(x_train, y_train, max_value=.8)
    x_test, y_test_smooth = labelsmoother(x_test, y_test, max_value=.8)
    
     # train network
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train_smooth, nb_epochs=5, batch_size=50)
    
    # print result
    print("After LabelSmoothing Defense\n")
    file.write("==== LabelSmoothing Defense==== \n")
    train_num = 60000
    test_num = 10000
    for k in range (5):
        file.write("==== Attack %i ====\n" % (k))
        evaluate(x_train, x_test, y_train_smooth, y_test_smooth, x_train_adv[k*train_num:(k+1)*train_num], x_test_adv[k*test_num:(k+1)*test_num], y_train, y_test, classifier, file)   
    