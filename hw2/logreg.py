import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import logging

np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# LEARNING RATE
STEP_SIZE=0.0001
MAX_ITERS=1000

def main(train_file,test_file):

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData(train_file,test_file)

  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  t0 = time.time()
  logging.info("Training logistic regression model")
  X_train_with_bias = dummyAugment(X_train)
  w, losses = trainLogistic(X_train_with_bias,y_train,MAX_ITERS,STEP_SIZE)
  y_pred_train = X_train_with_bias@w >= 0
  t1 = time.time()

  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  logging.info("Elapsed time = {:2f}".format(t1-t0))
  logging.info("\n---------------------------------------------------------------------------\n")

  # plt.figure(figsize=(16,9))
  # plt.plot(range(len(losses)), losses, label="Batch Gradient Descent")
  # plt.plot(range(len(losses_SGD)), losses_SGD, label="Stochastic Gradient Descent")
  # plt.plot(range(len(losses_MBGD)), losses_MBGD, label="Minibatch Gradient Descent")
  # plt.title("Logistic Regression Training Curve")
  # plt.xlabel("Epoch")
  # plt.ylabel("Negative Log Likelihood")
  # plt.legend()
  # plt.show()

  t0 = time.time()
  max_iters = 10
  w_SGD, losses_SGD = trainSGDLogistic(X_train_with_bias,y_train,max_iters,STEP_SIZE)
  y_pred_train = X_train_with_bias@w_SGD >= 0
  t1 = time.time()
  logging.info("Training SGD logistic regression model")
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w_SGD]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  logging.info("Elapsed time = {:2f}".format(t1-t0))
  
  logging.info("\n---------------------------------------------------------------------------\n")
  t0 = time.time()
  max_iters = 100
  w_MBGD, losses_MBGD = trainMiniBatchGDLogistic(X_train_with_bias,y_train,max_iters,STEP_SIZE)
  y_pred_train = X_train_with_bias@w_MBGD >= 0
  t1 = time.time()
  
  logging.info("Training MiniBatchGD logistic regression model")
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w_MBGD]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  logging.info("Elapsed time = {:2f}".format(t1-t0))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="Batch Gradient Descent")
  plt.plot(range(len(losses_SGD)), losses_SGD, label="Stochastic Gradient Descent")
  plt.plot(range(len(losses_MBGD)), losses_MBGD, label="Minibatch Gradient Descent")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  #return
  logging.info("\n---------------------------------------------------------------------------\n")

######################################################################
# dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X --  a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):
  ones = np.ones((len(X), 1))
  aug_X = np.hstack((ones, X))
  return aug_X
 
######################################################################
# logistic 
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input: 
#   z --   a n-by-1 vector
#
# Output:
#   logit_z --  a n-by-1 vector where logit_z[i] is the result of 
#               applying the logistic function to z[i]
######################################################################
def logistic(z):
  # This works for a single integer too
  # sigma(z) = 1 / (1 + e^-z)
  logit_z = np.negative(z)
  logit_z = np.exp(logit_z)
  logit_z = logit_z + 1
  logit_z = np.power(logit_z, -1)

  return logit_z


######################################################################
# calculateNegativeLogLikelihood 
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input: 
#   X --   a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example plus a bias term
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   w --    a (d+1)-by-1 weight vector
#
# Output:
#   nll --  the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X,y,w):
  # wT xi is the dot product of w and xi
  nll = 0

  for index in range(len(X)):
    x_vector = X[index]
    # print(x_vector)
    # print("w: " + str(w))
    wt = np.transpose(w)
    weighted_product = wt @ x_vector
    # print(weighted_product)
    logit = logistic(weighted_product)
    term_1 = y[index] * np.log(logit + 0.0000001)
    term_2 = (1 - y[index]) * np.log(1 - logit + 0.0000001)
    nll += term_1
    nll += term_2

  nll *= -1

  return nll

######################################################################
# trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of 
# iterations max_iters, and step size step_size -- run max_iters of 
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input: 
#   X --   a n-by-(d+1) matrix of examples where each row
#          corresponds to a single (d+1)-dimensional example where (d+1)
#          is the number of dimensions (d) plus a bias term
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   max_iters --   the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w --  the (d+1)-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################
def trainLogistic(X,y, max_iters, step_size):
    # Initialize our weights with zeros
    w = np.zeros( (X.shape[1],1) )
    
    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X,y,w)]
    
    # Take up to max_iters steps of gradient descent
    for i in range(max_iters):

        # Make a variable to store our gradient
        w_grad = np.zeros( (X.shape[1],1) )

        # Compute the gradient over the dataset and store in w_grad
        weighted_X = X @ w
        logit_wx = logistic(weighted_X)
        w_grad = np.transpose(X) @ (logit_wx - y)#second_term
               
        # This is here to make sure your gradient is the right shape
        assert(w_grad.shape == (X.shape[1],1))

        # Take the update step in gradient descent
        w = w - step_size*w_grad 
        
        # Calculate the negative log-likelihood with the 
        # new weight vector and store it for plotting later
        losses.append(calculateNegativeLogLikelihood(X,y,w))
        
    return w, losses

######################################################################
# trainSGDLogistic
######################################################################
def trainSGDLogistic(X,y, max_iters, step_size):
  raise Exception('Student error: You haven\'t implemented the trainSGDLogistic function yet.')
  return w, losses

######################################################################
# trainMiniBatchGDLogistic
######################################################################
def trainMiniBatchGDLogistic(X,y, max_iters, step_size):
  raise Exception('Student error: You haven\'t implemented the trainMiniBatchLogistic function yet.')
  return w, losses

##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Loads the train and test splits, passes back x/y for train and just x for test
def loadData(train_file,test_file):
  train = np.loadtxt(train_file, delimiter=",",skiprows=1)
  test = np.loadtxt(test_file, delimiter=",", skiprows=1)
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.

if __name__ == "__main__":
    """main"""
   
    if len(sys.argv) != 3:
        print("Usage: logreg.py <training data file> <test data file>")
        sys.exit(1)
    main(sys.argv[1],sys.argv[2])