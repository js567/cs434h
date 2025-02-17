import numpy as np
import time

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
        self.train_X = train_X
        self.train_Y = train_Y


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

        time0 = time.time()

        # Broadcasting operation - find difference between query and all rows
        timec = time.time()
        adjusted_matrix = self.train_X - query
        timed = time.time()
        # print("broadcast: " + str(timed-timec))

        # timecc = time.time()
        length_array = []
        # timedd = time.time()
        # print("create array: " + str(timedd - timecc))
        
        timee = time.time()

        for row in range(adjusted_matrix.shape[0]):
            length_array.append((np.linalg.norm(adjusted_matrix[row]), row))
 
        # timef = time.time()
        # print("norm: " + str(timef-timee))

        # timea = time.time()
        # # Sort vectors by length - room for improvement here?
        sorted_length_array = sorted(length_array, key=lambda tup: tup[0])
        # timeb = time.time()
        # print("sorting: " + str(timeb-timea))

        idx_of_nearest = []

        timeg = time.time()

        if k == len(sorted_length_array): 
            # print("LEN" + str(k) + str(len(sorted_length_array)))
            for item in range(len(sorted_length_array)):
                idx_of_nearest.append(sorted_length_array[item][1])
        else:
            for neighbor in range(k):
                idx_of_nearest.append(sorted_length_array[neighbor][1])

        timeh = time.time()
        # print("append: " + str(timeh - timeg))

        time1 = time.time()
        # print(time1 - time0)
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

        time0 = time.time()

        nn = self.get_nearest_neighbors(query, k)

        # Assign the label of the majority of neighbors
        n_zeros = np.count_nonzero(self.train_Y[nn]==0)

        if n_zeros > len(nn) / 2:
            predicted_label = 0
        else:
            predicted_label = 1

        # sum_zeros = 0
        # sum_ones = 0

        # for neighbor in range(len(nn)):
        #     # print(examples_Y[nn][neighbor])
        #     if examples_Y[nn][neighbor] == 0:
        #         sum_zeros += 1
        #     else: 
        #         sum_ones += 1

        # if sum_ones > sum_zeros:
        #     predicted_label = 1
        # else:
        #     predicted_label = 0

        # print("predicted_label: " + str(predicted_label))

        time1 = time.time()
        # print("classify: " + str(time1 - time0))

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

        return predicted_y


#######
# train function to allow interoperation with rplsh
#######

    def train(self, data1, data2):
        pass

    
