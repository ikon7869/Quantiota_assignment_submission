# Entropy-Based Classifier for Even/Odd MNIST Digits

Overview

This project implements a single-layer perceptron to classify MNIST digits as even or odd. Instead of using classical loss-based optimization (e.g., cross-entropy alone), the training updates are driven by an entropy gradient. Additionally, the model enforces a constraint on the change in the internal knowledge mapping (z) between iterations.

Implementation Strategy

Dual-Weight Structure and z Mapping:

  The model computes the knowledge mapping as:
  
  z = x dot (w1 + G1) + b1
  
  The activation is obtained using the sigmoid function:
  
  D = 1 / (1 + exp(-z))

Entropy Gradient and Parameter Updates:

  The entropy gradient is defined as:
  
  dH/dz = -1/ln(2) * z * D * (1 - D)
  
  Parameters are updated using both:
  
  The entropy gradient (unsupervised signal)
  
  The gradient from supervised binary cross-entropy loss
  
  The final update for each parameter is a weighted sum of the two gradients (alpha = 0.5).
  
z Constraint Enforcement:

  The model enforces the constraint:
  
  |z(new) - z(old)| < delta
  
  If an update would violate this constraint, the update is scaled down accordingly.

Key Differences from Classical Loss-Based Updates

Classical Approach:

  Uses a loss function (e.g., binary cross-entropy) to compute gradients.
  Parameter updates are performed using optimizers like SGD or Adam.
  
Entropy-Gradient Approach:

  Incorporates the gradient of an entropy functional as an additional (or alternative) update signal.
  Combines both supervised and unsupervised gradients, offering a unique perspective on training dynamics.
  
Training Dynamics and Observations

Stability:

  The enforced constraint on z helps to stabilize training by preventing abrupt changes in the model's internal representation.
  
Accuracy:

  The model achieves competitive accuracy on the even/odd classification task.
  Training dynamics show steady improvement with a dynamic learning rate schedule.  
  
Execution Instructions

Dependencies:

Python 

TensorFlow 2.17.1

NumPy 1.26.4

Matplotlib 3.7.5

Installation:

Install the dependencies using pip:

pip install tensorflow numpy matplotlib

Running the Code:

Save the source code in a file.
python entropy_classifier.py

The script will load the MNIST dataset, train the classifier, plot the training accuracy, and display sample predictions.
