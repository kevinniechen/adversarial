# Evaluating the Landscape of Adversarial Attacks and Defenses
**Authors:** Kevin Chen, Dexter Lee, Jacob Steen, Shivang Patel

![Sample](sample_fig.png)

## Description
Our project aimed to test two hypothesis about adver-sarial attacks and defenses.
**Hypothesis 1:** Because  distinct adversarial attacks employ different methods for generating adversarial examples, we  believe  that no defense method will allowclassifiers to be robust against all attacks.
**Hypothesis 2:** Because larger networks (deeper or more nodes) allow for more complex decision surfaces, adversarial attacks will decrease the accuracy of deeper networks more than they will for shallower networks.

![Cover](cover.png)

## Prerequisites
- python 3.6+
- Tensorflow
- Keras
- numpy
- pandas
- IBM Adversarial Training Robustness Toolbox

## Implementations
### Attacks:
- Fast Gradient Sign Method
- Jacobian Saliency Map Attack
- DeepFool
- NewtonFool
- Carlini L2 Attack
- Basic Iterative Method
- Universal Perturbation

### Defenses:
- Feature Squeezing
- Spacial Smoothing
- Gaussian Augmentation
- Label Smoothing
- Adversarial Training

### Neural Network Architectures:
- 2-Layer Fully-Connected Feed-Forward Neural Network
- 4-Layer Fully-Connected Feed-Forward Neural Network
- 7-Layer Fully-Connected Feed-Forward Neural Network
- 10-Layer Fully-Connected Feed-Forward Neural Network
- 1-Layer Convolutional Neural Nework with Maxpooling and Dropout
- 2-Layer Convolutional Neural Nework with Maxpooling and Dropout
