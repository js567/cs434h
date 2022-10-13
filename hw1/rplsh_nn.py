import numpy as np
import nn
import rplsh
import random

# Inherits from NearestNeighbor
class RPLSHNearestNeighbor(nn.NearestNeighbor):
    'Random Projection Locality Sensitive Hashing Nearest Neighbor Classifier'

    ######################################################################
    # __init__
    ######################################################################
    # Constructor for RPLSH class
    #
    # Input: 
    #   self --     the instance of the class
    #   train_X --  an n-by-d 2D Numpy array. Each row contains the n 
    #               features of a single d-dimensional data point
    #   train_Y --  a n-by-1 numpy array of the class labels for each
    #               data instance in train_X
    #   num_projections -- these are the number of random projections (i.e. 
    #                   random hyperplanes)
    #   num_hash_tables -- This is the number of hash tables to use (we call this
    #                       L in the class notes)
    #
    # Output:
    #   None
    ######################################################################
    def __init__(self, train_X, train_Y, num_projections, num_hash_tables) -> None:
        self.train_X = train_X
        self.train_Y = train_Y
        self.num_projections = num_projections
        self.num_hash_tables = num_hash_tables

        # Initiallize parent class with super()
        super().__init__(train_X, train_Y)

        # Create a variable to store the hyperplane vectors
        # Here, the object known as a 'hyperplane' is a vector normal to the hyperplane
        #   through the origin. This makes it easy to calculate the dot product later.
        # self.hyperplanes = [] # is this the correct variable type?
        # self.generate_hyperplanes()

        self.rplsh_functions = rplsh.RPLSH(train_X, train_Y, num_projections, num_hash_tables) 

        self.rplsh_functions.hash_dataset(train_X)
        # might move hyperplane construction to 
        # rplsh because of multiple hash tables being constructed

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
        print("method successfully overridden\n")

        # print(self.rplsh_functions.get_k_hash_entries(query, k))
        idx_of_all_nearest = self.rplsh_functions.get_k_hash_entries(query, k) # for now, will update this

        # create a dataset that consists of k neighbors 
        # if k don't exist in the bucket, go to the next buckets over, and etc.
        # run super().get_nearest_neighbors or just copy some logic from original implementation

        length_array = []

        for i in range(len(idx_of_all_nearest)):
            # for j in range(len(idx_of_all_nearest[i])):

                # distance_vector = self.train_X[idx_of_all_nearest[i][j]] - query
                distance_vector = self.train_X[idx_of_all_nearest[i]] - query
                length_array.append((np.linalg.norm(distance_vector), idx_of_all_nearest[i]))

        
        sorted_length_array = sorted(length_array, key=lambda tup: tup[0])
        print(sorted_length_array)

        idx_of_nearest = []

        if k == len(sorted_length_array): 
            for item in range(len(sorted_length_array)):
                idx_of_nearest.append(sorted_length_array[item][1])
        else:
            for neighbor in range(k):
                idx_of_nearest.append(sorted_length_array[neighbor][1])

        print("idx_of_nearest = " + str(idx_of_nearest) + "\n")
        return idx_of_nearest
        

if __name__ == '__main__':
    pass
    