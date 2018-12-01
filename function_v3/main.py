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
from attack_function import atk_DeepFool, atk_BasicIterative, atk_FastGradient, atk_JSMA, atk_CarliniAttack


## Read MNIST dataset (x_raw contains the original images):
(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist_raw()
x_train, y_train = preprocess(x_raw, y_raw)
x_test, y_test = preprocess(x_raw_test, y_raw_test)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# create and train baseline netowrk
classifier = create_Neural_Network(min_, max_)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=50)

# create txt files
baseline_file = open("baseline.txt", "w")
defense_1_file = open("defense_1.txt", "w")
defense_2_file = open("defense_2.txt", "w")
defense_3_file = open("defense_3.txt", "w")
defense_4_file = open("defense_4.txt", "w")
defense_5_file = open("defense_5.txt", "w")

# create attacks and append all the adversaril examples
# DeepFool
x_test_adv, x_train_adv = atk_DeepFool(x_train, x_test, y_train, y_test, classifier, baseline_file)
x_train_adv_all = x_train_adv
x_test_adv_all = x_test_adv
# BasicIterative
x_test_adv, x_train_adv = atk_BasicIterative(x_train, x_test, y_train, y_test, classifier, baseline_file)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)
# FastGradient
x_test_adv, x_train_adv = atk_FastGradient(x_train, x_test, y_train, y_test, classifier, baseline_file)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)
# JSMA
x_test_adv, x_train_adv = atk_JSMA(x_train, x_test, y_train, y_test, classifier, baseline_file)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)
# CarliniAttack
x_test_adv, x_train_adv = atk_CarliniAttack(x_train, x_test, y_train, y_test, classifier, baseline_file)
x_train_adv_all = np.append(x_train_adv_all, x_train_adv, axis=0)
x_test_adv_all = np.append(x_test_adv_all, x_test_adv, axis=0)

print("============ Finish Creating Attacks ============")


# utilize defenses
def_SpatialSmoothing(x_train, x_test, y_train, y_test, x_train_adv_all, x_test_adv_all, min_, max_, defense_1_file)
def_AdversarialTraining(x_train, x_test, y_train, y_test, x_train_adv_all, x_test_adv_all, min_, max_, defense_2_file)
def_FeatureSqueezing(x_train, x_test, y_train, y_test, x_train_adv_all, x_test_adv_all, min_, max_, defense_3_file)
def_GaussianAugmentation(x_raw, x_raw_test, y_raw, y_raw_test, x_train_adv_all, x_test_adv_all, y_train, y_test, min_, max_, defense_4_file)
def_LabelSmoothing(x_train, x_test, y_train, y_test, x_train_adv_all, x_test_adv_all, min_, max_, defense_5_file)
print("============ Finish Utilizing Defenses ============")

# close files
baseline_file.close()
defense_1_file.close()
defense_2_file.close()
defense_3_file.close()
defense_4_file.close()
defense_5_file.close()

print("============ Finish Program ============")


