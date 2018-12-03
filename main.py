# -*- coding: utf-8 -*-
"""
main
"""

# import libraries
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import abspath
import sys
sys.path.append(abspath('.'))

import numpy as np
from art.utils import load_mnist_raw, preprocess

from create_Neural_Network import create_Neural_Network
from defense_function import def_SpatialSmoothing, def_AdversarialTraining, def_FeatureSqueezing, def_GaussianAugmentation, def_LabelSmoothing
from attack_function import atk_DeepFool, atk_BasicIterative, atk_FastGradient, atk_JSMA, atk_CarliniAttack, atk_NeutonFool


## Read MNIST dataset (x_raw contains the original images):
(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist_raw()
x_train, y_train = preprocess(x_raw, y_raw)
x_test, y_test = preprocess(x_raw_test, y_raw_test)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# create and train baseline netowrk
classifier = create_Neural_Network(min_, max_)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)


# create attacks
x_test_adv, x_train_adv = atk_BasicIterative(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = x_train_adv
x_test_adv_all = x_test_adv

x_test_adv, x_train_adv = atk_FastGradient(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

x_test_adv, x_train_adv = atk_JSMA(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

x_test_adv, x_train_adv = atk_CarliniAttack(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

x_test_adv, x_train_adv = atk_NeutonFool(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

x_test_adv, x_train_adv = atk_UniPerturb(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

x_test_adv, x_train_adv = atk_DeepFool(x_train, x_test, y_train, y_test, classifier)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

print("============ Finish Creating Attacks ============")

train_num = 60000
test_num = 10000
# utilize defenses
def_SpatialSmoothing(x_train, x_test, y_train, y_test, x_train_adv_all[:train_num, :], x_test_adv_all[:test_num, :], min_, max_)
def_AdversarialTraining(x_train, x_test, y_train, y_test, x_train_adv_all[train_num:2*train_num, :], x_test_adv_all[test_num:2*test_num, :], min_, max_)
def_FeatureSqueezing(x_train, x_test, y_train, y_test, x_train_adv_all[2*train_num:3*train_num, :], x_test_adv_all[2*test_num:3*test_num, :], min_, max_)
def_GaussianAugmentation(x_raw, x_raw_test, y_raw, y_raw_test, x_train_adv_all[3*train_num:4*train_num, :], x_test_adv_all[3*test_num:4*test_num, :], y_train, y_test, min_, max_)
def_LabelSmoothing(x_train, x_test, y_train, y_test, x_train_adv_all[4*train_num:5*train_num, :], x_test_adv_all[4*test_num:5*test_num, :], min_, max_)


print("============ Finish Utilizing Defenses ============")
