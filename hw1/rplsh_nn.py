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

        # Do we need self as an argument here?
        # Initiallize parent class with super()
        super().__init__(train_X, train_Y)

        # Create a variable to store the hyperplane vectors
        # Here, the object known as a 'hyperplane' is a vector normal to the hyperplane
        #   through the origin. This makes it easy to calculate the dot product later.
        self.hyperplanes = [] # is this the correct variable type?
        self.generate_hyperplanes()

        rplsh_functions = rplsh.RPLSH(self.hyperplanes)


    def generate_hyperplanes(self):
        vector_length = (self.train_X.shape[1]) - 1 # Don't want to use index as a variable
        print("Vector length: " + str(vector_length))

        # Generate i hyperplanes with vector length j
        for i in range(self.num_projections):
            new_hyperplane = []
            for j in range(vector_length):
                new_hyperplane.append(random.gauss(0, 1))
            self.hyperplanes.append(new_hyperplane)
            # print(new_hyperplane)

        self.hyperplanes = np.array(self.hyperplanes)

        # print(self.num_projections)


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
        # TODO
        # return idx_of_nearest
        print("method successfully overridden")
        print()
        return [0, 1, 2]
        pass
        
if __name__ == '__main__':
    pass
    