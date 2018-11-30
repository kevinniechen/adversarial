# -*- coding: utf-8 -*-
"""
Attack Functions
"""
from art.attacks import DeepFool, BasicIterativeMethod, FastGradientMethod, SaliencyMapMethod, CarliniL2Method
from art.utils import random_targets

from evaluate import evaluate

def atk_DeepFool(x_train, x_test, y_train, y_test, classifier):
    #print('Create DeepFool attack \n')
    adv_crafter = DeepFool(classifier, max_iter=20)
    x_train_adv = adv_crafter.generate(x_train)
    x_test_adv = adv_crafter.generate(x_test)
    
    print("After DeepFool Attack \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv


def atk_BasicIterative(x_train, x_test, y_train, y_test, classifier):
    #print('Create BasicIterativeMethod attack \n')
    adv_crafter = BasicIterativeMethod(classifier, eps=1, eps_step=0.1)
    x_train_adv = adv_crafter.generate(x_train)
    x_test_adv = adv_crafter.generate(x_test)
    
    print("After BasicIterative Attack  \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv


def atk_FastGradient(x_train, x_test, y_train, y_test, classifier):
    epsilon = 0.1
    #print('Create FastGradientMethod attack \n')
    adv_crafter = FastGradientMethod(classifier)
    x_train_adv = adv_crafter.generate(x_train, eps=epsilon)
    x_test_adv = adv_crafter.generate(x_test, eps=epsilon)
    
    print("After FastGradient Attack  \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv

def atk_JSMA(x_train, x_test, y_train, y_test, classifier):
    #print('Create JSMA attack \n')
    adv_crafter = SaliencyMapMethod(classifier, theta=1)
    x_train_adv = adv_crafter.generate(x_train)
    x_test_adv = adv_crafter.generate(x_test)
    
    print("After JSMA Attack  \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv

def atk_CarliniAttack(x_train, x_test, y_train, y_test, classifier):
    #print('Create CarliniL2Method attack \n')
    adv_crafter = CarliniL2Method(classifier, targeted=True, max_iter=100, binary_search_steps=1, learning_rate=1, initial_const=10, decay=0)
    params = {'y': random_targets(y_test, classifier.nb_classes)}
    x_train_adv = adv_crafter.generate(x_train, **params)
    x_test_adv = adv_crafter.generate(x_test, **params)
    
    print("After CarliniAttack Attack  \n")
    evaluate(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, classifier)
    return x_test_adv, x_train_adv

