# Adversarial Learning in Credit Card Fraud Detection
Increasing rates of credit card fraud have resulted in significant monetary losses for financial institutions. While many fraud detection systems have been successful in flagging genuine fraudulent transactions, few incorporate an adversary's potential strategies in order to adapt. In order to create an adaptive model, knowledge of an adversary's most effective strategy is beneficial.
The contributions of this project are the following:
1. Introducing an adaptive fraud detection system that utilizes repeated games in the form of a feedforward model and incorporates the synthetic minority oversampling technique (SMOTE) to mitigate class imbalance.
2. Utilizing Gaussian Mixture Models (GMMs) to segment the distribution space of continuous attributes as a means to find possible adversarial strategies.

## Instructions: 
- Run `python adversarial_learning.py` in order to train the model
- Once the model is trained, you can output the ROC curves by using `python curves.py`

### Dependencies:
- Python 3.5

```
conda install pandas 
conda install numpy
conda install scipy
conda install -c glemaitre imbalanced-learn 
conda install matplotlib
conda install pyqt=4
conda install pytables
```

