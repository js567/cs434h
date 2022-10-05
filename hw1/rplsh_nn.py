import numpy as np
import nn
import rplsh

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
    def get_nearest_neighbors(self,query,k):
        # TODO
        return idx_of_nearest
        
if __name__ == '__main__':
    pass
    