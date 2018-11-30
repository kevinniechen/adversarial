# -*- coding: utf-8 -*-
"""
main
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import abspath
import sys
sys.path.append(abspath('.'))

from art.utils import load_mnist_raw, preprocess
from defense_function import defense_SpatialSmoothing, defense_AdversarialTraining, defense_FeatureSqueezing, defense_GaussianAugmentation, defense_LabelSmoothing
from attack_function import attack_DeepFool, attack_BasicIterativeMethod, attack_FastGradientMethod, attack_JSMA, attack_CarliniL2Method


## Read MNIST dataset (x_raw contains the original images):
(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist_raw()
x_train, y_train = preprocess(x_raw, y_raw)
x_test, y_test = preprocess(x_raw_test, y_raw_test)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# DeepFool + SpatialSmoothing
x_test_adv, x_train_adv = attack_DeepFool(x_train, x_test, y_train, y_test, min_, max_)
defense_SpatialSmoothing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_)

# BasicIterativeMethod + AdversarialTraining
x_test_adv, x_train_adv = attack_BasicIterativeMethod(x_train, x_test, y_train, y_test, min_, max_)
defense_AdversarialTraining(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_)

# FastGradientMethod + FeatureSqueezing
x_test_adv, x_train_adv = attack_FastGradientMethod(x_train, x_test, y_train, y_test, min_, max_)
defense_FeatureSqueezing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_)


# JSMA + GaussianAugmentation
x_test_adv, x_train_adv = attack_JSMA(x_train, x_test, y_train, y_test, min_, max_)
defense_GaussianAugmentation(x_raw, x_raw_test, y_raw, y_raw_test, x_train_adv, x_test_adv, y_train, y_test, min_, max_)


# CarliniL2Method + LabelSmoothing
x_test_adv, x_train_adv = attack_CarliniL2Method(x_train, x_test, y_train, y_test, min_, max_)
defense_LabelSmoothing(x_train, x_test, y_train, y_test, x_train_adv, x_test_adv, min_, max_)


