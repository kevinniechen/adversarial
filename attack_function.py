# -*- coding: utf-8 -*-
"""
Attack
"""
from art.attacks import DeepFool, BasicIterativeMethod, FastGradientMethod, SaliencyMapMethod, CarliniL2Method
from art.utils import random_targets

from create_Neural_Network import create_Neural_Network
from evaluate import evaluate

def attack_DeepFool(x_train, x_test, y_train, y_test,  min_, max_):
    # train baseline netowrk
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)
    
    ## Create DeepFool adversarial samples
    print('Create DeepFool attack \n')
    adv_crafter = DeepFool(classifier, max_iter=20)
    x_train_adv = adv_crafter.generate(x_train)
    x_test_adv = adv_crafter.generate(x_test)
    
    print("Before Defense \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv


def attack_BasicIterativeMethod(x_train, x_test, y_train, y_test,  min_, max_):
    # train baseline netowrk
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)
    
    ## Create BasicIterativeMethod adversarial samples
    print('Create BasicIterativeMethod attack \n')
    adv_crafter = BasicIterativeMethod(classifier, eps=1, eps_step=0.1)
    x_train_adv = adv_crafter.generate(x_train)
    x_test_adv = adv_crafter.generate(x_test)
    
    print("Before Defense \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv


def attack_FastGradientMethod(x_train, x_test, y_train, y_test,  min_, max_):
    # train baseline netowrk
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)
    
    ## Create FastGradientMethod adversarial samples
    epsilon = 0.1
    print('Create FastGradientMethod attack \n')
    adv_crafter = FastGradientMethod(classifier)
    x_train_adv = adv_crafter.generate(x_train, eps=epsilon)
    x_test_adv = adv_crafter.generate(x_test, eps=epsilon)
    
    print("Before Defense \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv

def attack_JSMA(x_train, x_test, y_train, y_test,  min_, max_):
    # train baseline netowrk
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)
    
    ## Create JSMA adversarial samples
    print('Create JSMA attack \n')
    adv_crafter = SaliencyMapMethod(classifier, theta=1)
    x_train_adv = adv_crafter.generate(x_train)
    x_test_adv = adv_crafter.generate(x_test)
    
    print("Before Defense \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv

def attack_CarliniL2Method(x_train, x_test, y_train, y_test,  min_, max_):
    # train baseline netowrk
    classifier = create_Neural_Network(min_, max_)
    classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)
    
    ## Create CarliniL2Method adversarial samples
    print('Create CarliniL2Method attack \n')
    adv_crafter = CarliniL2Method(classifier, targeted=True, max_iter=100, binary_search_steps=1, learning_rate=1, initial_const=10, decay=0)
    params = {'y': random_targets(y_test, classifier.nb_classes)}
    x_train_adv = adv_crafter.generate(x_train, **params)
    x_test_adv = adv_crafter.generate(x_test, **params)
    
    print("Before Defense \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv



