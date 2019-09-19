# Supervised Machine Learning
Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output.

# KNeighbors Regressor with wave dataset
In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. 

![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/images/knn.PNG)


# Linear Models
 Linear Regression
Linear regression, or ordinary least squares (OLS), is the simplest and most classic linear method for regression. Linear regression finds the parameters w and b that minimize the mean squared error between predictions and the true regression targets, y, on the training set. The mean squared error is the sum of the squared differences between the predictions and the true values. Linear regression has no parameters, which is a benefit, but it also has no way to control model complexity. By using wave dataset :

 ![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/images/linear_regression.PNG)

# Ridge Regression
Ridge regression is also a linear model for regression, so the formula it uses to make predictions is the same one used for ordinary least squares. In ridge regression, though, the coefficients (w) are chosen not only so that they predict well on the training data, but also to fit an additional constraint. We also want the magnitude of coefficients to be as small as possible; in other words, all entries of w should be close to zero. Intuitively, this means each feature should have as little effect on the outcome as possible (which translates to having a small slope), while still predicting well. This constraint is an example of what is called regularization. Regularization means explicitly restricting a model to avoid overfitting. The particular kind used by ridge regression is known as L2 regularization.

![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/images/ridge_regression.PNG)

# Linear Regression VS Ridge Regression
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/images/LR_vs_RR.PNG) 

# Lasso
An alternative to Ridge for regularization linear regression is lasso. As with ridge regression, using lasso also restricts coeficients to be close to zero but in a slightly different way, called L1 regularization.8 The consequence of L1 regularization is that when using the lasso, some coefficients are exactly zero. This means some features are entirely ignored by the model. This can be seen as a form of automatic feature selection. Having some coefficients be exactly zero often makes a model easier to interpret, and can reveal the most important features of your model.

![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/images/lasso.PNG)

# Linear model for multiclass classification
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img1.PNG)

* Let’s visualize the lines given by the three binary classifiers:-
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img2.PNG)

* Final result
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img3.PNG)


# Decision tree classifier
* Dataset visualization of Cancer dataset

![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img4.PNG)

* Prediction
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img5.PNG)

# Neural Networks (Deep Learning)
A family of algorithms known as neural networks has recently seen a revival under the name “deep learning.” While deep learning shows great promise in many machine learning applications, deep learning algorithms are often tailored very carefully to a specific use case. Here, we will only discuss some relatively simple methods, namely multilayer perceptrons for classification and regression, that can serve as a starting point for more involved deep learning methods. Multilayer perceptrons (MLPs) are also known as (vanilla) feed-forward neural networks, or sometimes just neural networks.

# The neural network model
Remember that the prediction by linear regression is given as:
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img6.PNG)

* Visualization of logistic regression , where input feature and prediction are shown as ndes and the cofficients are connections between the nodes.
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img7.PNG)


# Activation Functions
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img8.PNG)

# Multilayer perceptron
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img9.PNG)

# MLP Classifier on make_moon dataset
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img10.PNG)

* Comparision on differnt size of hidden layers units
![alt text](https://github.com/paramjeet2005/Introduction-to-machine-learning-with-python/blob/master/2.%20Supervised%20Learning/images/img11.PNG)

