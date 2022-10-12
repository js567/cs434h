import numpy as np

class RPLSH:
    'Random Projection Locality Sensitive Hashing class'

    # Need access to the hyperplanes from rplsh_nn
    def __init__(self, hyperplanes):
        self.hyperplanes = hyperplanes


######################################################################
# get_hash_code
######################################################################
# Creates the hash code for the data instance x.
#
# Input: 
#   self -- a reference to the instance of the class
#   x -- a 1-by-Ddnumpy array representing the data instance, where d = number of features
#
# Output:
#   returns a string representing the hash code for the data instance x
######################################################################

    def get_hash_code(self, x):
        #TODO
        # 


        # return(hashcode)

        pass

######################################################################
# hash_dataset
######################################################################
# Takes a dataset and inserts all the data instances into its internal
# hash table. The hash code is computed using the random hyperplanes.
#
# Input: 
#   self -- a reference to the instance of the class
#   dataset -- an n-by-d numpy array representing the dataset, where n
#       is the number of data points and d is the number of features.
#
# Output:
#   None
######################################################################
    def hash_dataset(self,dataset):
        # TODO
        pass

######################################################################
# get_hash_entries
######################################################################
# Returns a numpy array of indices corresponding to data points in the
# training data that are approximately close to x. More precisely, these
# indices are the data points that map to the same hash bucket as x does.
#
# Input: 
#   self -- a reference to the instance of the class
#   x -- a 1-by-d numpy array representing the data instance, where d = number of features
#
# Output:
#   returns a list of the indices of data instances that map to the hash
#   bucket.
######################################################################
    def get_hash_entries(self,x):
        # TODO
        return []

if __name__ == '__main__':
    pass