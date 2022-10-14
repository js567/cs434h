from operator import index
import numpy as np
import random

class RPLSH:
    'Random Projection Locality Sensitive Hashing class'

    # Need access to the hyperplanes from rplsh_nn
    def __init__(self, train_X, train_Y, num_projections, num_hash_tables):

        self.train_X = train_X
        self.train_Y = train_Y
        self.num_projections = num_projections
        self.num_hash_tables = num_hash_tables

        # TEMPORARY - GOING TO DEVELOP USING JUST ONE HASH TABLE
        self.hash_table = []

        # Fill hash table with enough empty lists (these will be turned into np arrays later)
        self.hash_table = [[] for _ in range(pow(2, self.num_projections))]
        # print(self.hash_table)

        # create extra hash tables if necessary
        self.extra_hash_tables = []
        if self.num_hash_tables > 1:
            for h in range(num_hash_tables - 1):
                self.extra_hash_tables.append([[] for _ in range(pow(2, self.num_projections))])

        self.hyperplanes = self.generate_hyperplanes()

        self.hash_dataset(train_X)


    # Need to generate a set of hyperplanes for each hash table
    def generate_hyperplanes(self):

        # Hyperplane vectors will have the same length as data vectors
        vector_length = (self.train_X.shape[1])
        # print("Vector length: " + str(vector_length))

        hyperplanes = []

        for h in range(self.num_hash_tables):
            # Generate i hyperplanes with vector length j
            h_hyperplanes = []

            for i in range(self.num_projections):

                new_hyperplane = []

                for j in range(vector_length):
                    new_hyperplane.append(random.gauss(0, 1))

                # Converting each hyperplane into an np array for dot product calculation
                new_hyperplane = np.array(new_hyperplane)
                h_hyperplanes.append(new_hyperplane)
                # print(new_hyperplane)

            hyperplanes.append(h_hyperplanes)

        # print(hyperplanes)
        return hyperplanes


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

    # only going to call this from RPLSH?
    # maybe pass hyperplanes to this function to more quickly insert into hash table
    def get_hash_code(self, x, hyperplane=0):

        hashcode = ""
        int_hashcode = int()

        # Getting this to work first for one hash table - CHANGE THIS LATER OR REDESIGN FUNCTION
        # if self.num_hash_tables == 1:
        for i in range(len(self.hyperplanes)):
            # print("1 hash table")
            hashcode = ""
            for hyperplane in range(len(self.hyperplanes[i])):
                dot_product = np.dot(self.hyperplanes[i][hyperplane], x)
                # print("Dot Product: " + str(dot_product))
                if dot_product >= 0: # if point is on 'top' of plane
                    hashcode += "1"
                elif dot_product < 0: # if point is on 'bottom' of plane
                    hashcode += "0"

            # print("hash code: " + hashcode)

            # Converting hash code from binary into integer for easy storage in hash table
            int_hashcode = int(hashcode, 2)
            # print("int hash code: " + str(int_hashcode))

        return(int_hashcode)


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

    def hash_dataset(self, dataset):

        # For each entry in the dataset, generate hash code and insert into hash table
        for i in range(self.num_hash_tables):
            for index in range(dataset.shape[0]):
                if i == 0:
                    new_hashcode = self.get_hash_code(dataset[index])
            # print(dataset[index])
            # print("index: " + str(index))
            # print("hashcode from has_dataset: " + str(new_hashcode))
                    self.hash_table[new_hashcode].append(index)#append(dataset[index])
            # print(self.hash_table[new_hashcode])
            # print("\n")
            # print(data_shape)
                else:
                    new_hashcode = self.get_hash_code(dataset[index], i)
                    self.extra_hash_tables[i-1].append(index)

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

    def get_hash_entries(self, x):

        x_hash_code = self.get_hash_code(x)
        return self.hash_table[x_hash_code]


    # Similar premise to function above, but this will return k indices and will
    # pull values from adjacent buckets if needed

    def get_kl_hash_entries(self, query, k, l=1):

        index_list = []
        query_hash_code = self.get_hash_code(query) 

        index_list = index_list + self.hash_table[query_hash_code]

        # print("index list: " + str(index_list))

        j = 1
        while len(index_list) < k: 
            if query_hash_code - j >= 0:
                for n in range(len(self.hash_table[query_hash_code - j])):
                    index_list = index_list + [self.hash_table[query_hash_code - j][n]]
            if query_hash_code + j < len(self.hash_table):
                for l in range(len(self.hash_table[query_hash_code + j])):
                    index_list = index_list + [self.hash_table[query_hash_code + j][l]]
            j += 1

        return index_list



if __name__ == '__main__':
    pass