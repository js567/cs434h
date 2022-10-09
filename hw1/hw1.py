import numpy as np
import sys
import time
import nn
import rplsh_nn
import multiprocessing

def main(training_file,test_file,mode):
  sanity_check(training_file,test_file,mode)
  # Commented out cross validation for testing
  run_cross_validation_test(training_file,test_file,mode)

######################################################################
# sanity_check
######################################################################
# Runs some sanity checks on your code once you finish the homework.
# You can use this to test your code.
#
# Input: 
#   training_file -- the name of the training dataset
#   test_file --  the name of the test dataset
#   mode -- set to '0' for regular kNN and set to '1' for LSH kNN. Note
#           that this is a string, not an int
#    
# Output:
#   None 
######################################################################
def sanity_check(training_file, test_file, mode):
   
  #############################################################
  # These first bits are just to help you develop your code
  # and have expected ouputs given. All asserts should pass.
  ############################################################

  # I made up some random 3-dimensional data and some labels for us
  example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],
                                 [ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
  example_train_Y = np.array([[0], [1], [1], [1], [0], [1]])
  classifier = nn.NearestNeighbor(example_train_x, example_train_Y)
  
  #########
  # Sanity Check 1: If I query with examples from the training set 
  # and k=1, each point should be its own nearest neighbor
    
  for i in range(len(example_train_x)):
    assert([i] == classifier.get_nearest_neighbors(example_train_x[i], 1))

  # for i in range(len(example_train_x)):
  #   print(classifier.get_nearest_neighbors(example_train_x[i], 3))

  print("Passed Sanity Check 1")

  #########
  # Sanity Check 2: See if neighbors are right for some examples (ignoring order)
  nn_idx = classifier.get_nearest_neighbors(np.array( [ 1, 4, 2] ), 2)
  assert(set(nn_idx).difference(set([4,3]))==set())

  nn_idx = classifier.get_nearest_neighbors(np.array( [ 1, -4, 2] ), 3)
  assert(set(nn_idx).difference(set([1,0,2]))==set())

  nn_idx = classifier.get_nearest_neighbors(np.array( [ 10, 40, 20] ), 5)
  assert(set(nn_idx).difference(set([4, 3, 0, 2, 1]))==set())

  print("Passed Sanity Check 2")

  #########
  # Sanity Check 3: Neighbors for increasing k should be subsets
  query = np.array( [ 10, 40, 20] )
  p_nn_idx = classifier.get_nearest_neighbors(query, 1)
  for k in range(2,7):
    nn_idx = classifier.get_nearest_neighbors(query, k)
    assert(set(p_nn_idx).issubset(nn_idx))
    p_nn_idx = nn_idx

  print("Passed Sanity Check 3.0")
   
  #########
  # Test out our prediction code
  queries = np.array( [[ 10, 40, 20], [-2, 0, 5], [0,0,0]] )
  # print(str(classifier.get_nearest_neighbors([10, 40, 20], 3)))
  # print(str(classifier.get_nearest_neighbors([-2, 0, 5], 3)))
  # print(str(classifier.get_nearest_neighbors([0, 0, 0], 3)))

  # print("\n")

  pred = classifier.classify_dataset(queries, 3)
  assert( np.all(pred == np.array([[0],[1],[0]])))

  print("Passed Sanity Check 3.1")

  #########
  # Test our our accuracy code
  true_y = np.array([[0],[1],[2],[1],[1],[0]])
  pred_y = np.array([[5],[1],[0],[0],[1],[0]])                    
  assert( compute_accuracy(true_y, pred_y) == 3/6)

  pred_y = np.array([[5],[1],[2],[0],[1],[0]])                    
  assert( compute_accuracy(true_y, pred_y) == 4/6)

  print("Passed Sanity Check 3.2")

######################################################################
# run_cross_validation_test
######################################################################
# Runs the cross validation part of the homework
#
# Input: 
#   training_file -- the name of the training dataset
#   test_file --  the name of the test dataset
#   mode -- set to '0' for regular kNN and set to '1' for LSH kNN. Note
#           that this is a string, not an int
#    
# Output:
#   None but will save the predictions to a file
######################################################################
def run_cross_validation_test(training_file, test_file,mode):
  # Load training and test data as numpy matrices 
  traindata = np.genfromtxt(training_file, delimiter=',')[1:, 1:]
  # train_X = traindata[:, :100]
  # train_Y = traindata[:, :100]
  train_X = traindata[:, :-1]
  train_Y = traindata[:, -1]
  train_Y = train_Y[:,np.newaxis]
  if( mode == "0" ):
    classifier = nn.NearestNeighbor(train_X, train_Y)
  elif (mode == "1"):
    classifier = rplsh_nn.RPLSHNearestNeighbor(train_X, train_Y,4,2)
  else:
    sys.exit("mode must be 0 or 1")
  test_X = np.genfromtxt(test_file, delimiter=',')[1:, 1:]
  
  ####################################################################
  # Q3 Hyperparameter Search
  ####################################################################
  # Search over possible settings of k
  print("Performing 4-fold cross validation")

  k_selection_set = []

  for k in [1,3,5,7,9]:#,99]:#,999,8000]:
    t0 = time.time()

    # Compute train accuracy using whole set
    # print("shape")
    # print(train_X.shape[0])

    predicted_labels = classifier.classify_dataset(train_X[:100], k)
    train_acc = compute_accuracy(predicted_labels[:100], train_Y)

    # Compute 4-fold cross validation accuracy
    val_acc, val_acc_var = cross_validation(mode, train_X, train_Y, 4, k)
        
    t1 = time.time()
    print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))

    k_selection_set.append((val_acc, k))
      
  # TODO set your best k value and then run on the test set
  # SHOULD CALCULATE THIS BASED OFF OF VALIDATION ACCURACY PROBABLY
  # How should we interpret variance in this situation?

  # Sort k values by best validation accuracy - maybe add an arbitrary variance cutoff point?
  sorted_k_selection_set = sorted(k_selection_set, key=lambda tup: tup[0], reverse=True)
  best_k = sorted_k_selection_set[0][1]

  print("Best K: " + str(best_k))

  # Make predictions on test set
  classifier.train(train_X, train_Y) #WHAT DOES THIS MEAN????
  pred_test_y = classifier.classify_dataset(test_X, best_k)    
    
  # add index and header then save to file
  test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=int), axis=1), pred_test_y), axis=1)
  header = np.array([["id", "income"]])
  test_out = np.concatenate((header, test_out))
  np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')


######################################################################
# cross_validation 
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input: 
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#   num_folds -- the number of folds (should be 4 for this assn)
#   k -- the number of nearest neighbors to look at
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   varr_val_acc --      the variance of validation accuracy across the folds
######################################################################
def cross_validation(mode, train_X, train_Y, num_folds=4, k=1):

  # Split train_X and train_Y into K folds for computation
  # TODO - CONVERT THIS IN TERMS OF NUM_FOLDS FOR CONTINUITY
  split_X = np.split(train_X, [1999, 3999, 5999])
  split_Y = np.split(train_Y, [1999, 3999, 5999])

  # avg_val_acc = 0
  accuracy_array = []

  for i in range(num_folds):

    # Test - 1/4 of dataset
    fold_test_X = split_X[i]
    fold_test_Y = split_Y[i]

    # Create 3/4 X dataset
    fold_train_X_uncombined = np.delete(split_X, i)
    fold_train_X_combined = np.vstack(fold_train_X_uncombined)

    # Create 3/4 Y labels
    fold_train_Y_uncombined = np.delete(split_Y, i)
    fold_train_Y_combined = np.vstack(fold_train_Y_uncombined)

    # Create new classifier trained on each small set of data
    if( mode == "0" ):
      fold_classifier = nn.NearestNeighbor(fold_train_X_combined, fold_train_Y_combined)
    elif (mode == "1"):
      fold_classifier = rplsh_nn.RPLSHNearestNeighbor(fold_train_X_combined, fold_train_Y_combined,4,2)

    # Train classifier if necessary - RPLSH only
    fold_classifier.train(fold_train_X_combined, fold_train_Y_combined) #WHAT DOES THIS MEAN????

    # Predict from fold test set
    fold_pred = fold_classifier.classify_dataset(fold_test_X, k)  
    fold_acc = compute_accuracy(fold_pred, fold_test_Y)

    # Add accuracy result to array to compute mean and variance
    accuracy_array.append(fold_acc)

  # Calculate mean and variance from accuracy array
  avg_val_acc = np.average(accuracy_array)
  varr_val_acc = np.var(accuracy_array)

  print("k-fold accuracies: " + str(accuracy_array))

  return(avg_val_acc, varr_val_acc)


##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   true_y --  a n-by-1 vector where each value corresponds to 
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match 
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy
    
if __name__ == "__main__":
    """main"""
   
    if len(sys.argv) != 4:
        print("Usage: hw1.py <training data file> <test data file> <mode 0=regular kNN, 1=RPLSH kNN>")
        sys.exit(1)
    main(sys.argv[1],sys.argv[2],sys.argv[3])