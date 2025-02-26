# Quantiota_assignment_submission

Entropy-Based Classifier for Even/Odd MNIST Digits
Overview
This project implements a single-layer perceptron for classifying MNIST digits as even or odd. Unlike classical training methods based on loss minimization (e.g., cross-entropy), this implementation uses an alternative update mechanism driven by the gradient of an entropy functional. The classifier also enforces a constraint on the change in the "knowledge" mapping 
𝑧
z between iterations.

Implementation Strategy
Dual-Weight Structure & z Mapping:
The knowledge mapping is computed as:

𝑧
=
∑
𝑗
=
1
𝑛
(
𝑤
1
,
𝑗
+
𝐺
1
,
𝑗
)
𝑥
𝑗
+
𝑏
1
z= 
j=1
∑
n
​
 (w 
1,j
​
 +G 
1,j
​
 )x 
j
​
 +b 
1
​
 
where 
𝑥
𝑗
x 
j
​
  are the input features, 
𝑤
1
,
𝑗
w 
1,j
​
  and 
𝐺
1
,
𝑗
G 
1,j
​
  are two sets of weights, and 
𝑏
1
b 
1
​
  is the bias. The activation 
𝐷
D is then obtained using the sigmoid function:

𝐷
=
𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
D=σ(z)= 
1+e 
−z
 
1
​
 
Entropy Gradient and Parameter Updates:
The entropy functional is defined such that its gradient with respect to 
𝑧
z is:

∂
𝐻
(
𝑧
)
∂
𝑧
=
−
1
ln
⁡
2
𝑧
 
𝐷
 
(
1
−
𝐷
)
∂z
∂H(z)
​
 =− 
ln2
1
​
 zD(1−D)
This gradient is used to update the parameters:

𝑤
1
,
𝑗
←
𝑤
1
,
𝑗
−
𝜂
 
∂
𝐻
(
𝑧
)
∂
𝑧
 
𝑥
𝑗
w 
1,j
​
 ←w 
1,j
​
 −η 
∂z
∂H(z)
​
 x 
j
​
 
𝐺
1
,
𝑗
←
𝐺
1
,
𝑗
−
𝜂
 
∂
𝐻
(
𝑧
)
∂
𝑧
 
𝑥
𝑗
G 
1,j
​
 ←G 
1,j
​
 −η 
∂z
∂H(z)
​
 x 
j
​
 
𝑏
1
←
𝑏
1
−
𝜂
 
∂
𝐻
(
𝑧
)
∂
𝑧
b 
1
​
 ←b 
1
​
 −η 
∂z
∂H(z)
​
 
Additionally, supervised gradients computed from binary cross-entropy loss are combined with the entropy-based updates using a weighting factor (
𝛼
=
0.5
α=0.5).

z Constraint Enforcement:
To ensure stable training, the update is constrained so that the change in 
𝑧
z between iterations satisfies:

∣
𝑧
𝑖
+
1
−
𝑧
𝑖
∣
<
𝛿
∣z 
i+1
​
 −z 
i
​
 ∣<δ
If this condition is violated, the updates are scaled down accordingly.

Vectorization and Performance Tuning:
All updates are vectorized to fully leverage GPU parallelism. In particular, the outer product between the input features and the entropy gradient is computed in a single matrix multiplication, replacing per-sample loops.

Key Differences from Classical Loss-Based Updates
Classical Approach:
Typically uses a loss function (e.g., cross-entropy) to compute gradients, and then an optimizer (like Adam or SGD) to update parameters.

Entropy-Gradient Approach:
In this implementation, updates are driven by the gradient of an entropy functional, providing an alternative perspective on model training. The supervised loss is still used but is combined with the entropy gradient.

Training Dynamics and Observations
Training Stability:
The enforced constraint on 
𝑧
z (
𝛿
δ) ensures that updates remain stable over iterations, preventing abrupt changes in the model's internal knowledge representation.

Accuracy:
The model reaches competitive accuracy on the even/odd classification task. Observations during training show steady improvements and stable convergence.

Execution Instructions
Dependencies:

Python 3.x
TensorFlow 2.x
NumPy
Matplotlib
Installation:

Install dependencies via pip:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
Running the Code:

Save the source code in a file (e.g., entropy_classifier.py).
Run the script:
bash
Copy
Edit
python entropy_classifier.py
The script will load the MNIST dataset, train the classifier, and display the training accuracy over epochs along with sample predictions.
Optional:

For GPU usage, ensure you have a compatible GPU and the necessary CUDA/cuDNN libraries installed.
3. Execution Instructions Summary
Dependencies:
Python 3.x, TensorFlow 2.x, NumPy, Matplotlib

Installation Command:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
Run the Script:

bash
Copy
Edit
python entropy_classifier.py
