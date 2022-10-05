import numpy as np

class NearestNeighbor:
    'K-Nearest Neighbor classifier'
######################################################################
# __init___
######################################################################
# Constructor for the kNN classifier. Note that you have to pass in
# the training set (features and class labels) when you construct it.
#
# Input: 
#
#   self --     the instance of the class
#   train_X --  an n-by-d 2D Numpy array. Each row contains the n 
#               features of a single d-dimensional data point
#   train_Y --  a n-by-1 numpy array of the class labels for each
#               data instance in train_X
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################
    def __init__(self, train_X, train_Y) -> None:
        #TODO
        self.train_X = train_X
        self.train_Y = train_Y
        # pass


######################################################################
# get_nearest_neighbors 
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the 
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out 
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy. 
#
# Input: 
#   self --     the instance of the class
#   query --    a 1-by-d vector representing a single example
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by-1 list of indices for the nearest k
#                       neighbors of the query point
######################################################################
    def get_nearest_neighbors(self, query, k):

        adjusted_matrix = self.train_X - query
        length_array = []

        for row in range(adjusted_matrix.shape[0]):
            length_array.append((np.linalg.norm(adjusted_matrix[row]), row))

        sorted_length_array = sorted(length_array, key=lambda tup: tup[0])

        idx_of_nearest = []

        for neighbor in range(k):
            idx_of_nearest.append(sorted_length_array[neighbor][1])

        return idx_of_nearest  
    

######################################################################
# classify 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of nearest neighbors to look at
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################
    def classify(self, query, k):
        # TODO
        examples_X = self.train_X
        examples_Y = self.train_Y

        nn = self.get_nearest_neighbors(query, k)
        # print(nn)

        sum_zeros = 0
        sum_ones = 0

        for neighbor in range(len(nn)):
            # print(examples_Y[nn][neighbor])
            if examples_Y[nn][neighbor] == 0:
                sum_zeros += 1
            else: 
                sum_ones += 1

        # print(sum_ones)
        # print(sum_zeros)

        if sum_ones > sum_zeros:
            predicted_label = 1
        else:
            predicted_label = 0

        # print("predicted_label: " + str(predicted_label))
        return predicted_label


######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input: 
#   queries_X --    a m-by-d matrix representing a set of queries 
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################
    def classify_dataset(self, queries_X, k): 

        predicted_y = []

        for query_idx in range(queries_X.shape[0]):
            predicted_y.append([self.classify(queries_X[query_idx], k)])

        # print(predicted_y)
        # return np.zeros(len(queries_X))
        return predicted_y
