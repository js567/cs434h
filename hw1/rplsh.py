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

        # self.hash_tables = []

        # TEMPORARY - GOING TO DEVELOP USING JUST ONE HASH TABLE
        self.hash_table = []

        # add sufficient np arrays to hash table
        print("Necessary number of hash buckets: " + str(pow(2, self.num_projections)))
        # array_shape = (train_X.shape[1], 1)
        # for i in range(pow(2, num_projections)):
        #     print(i) # garbage
        #     self.hash_table.append(np.zeros(array_shape))
        # print(self.hash_table)

        # Fill hash table with enough empty lists (these will be turned into np arrays later)
        self.hash_table = [[] for _ in range(pow(2, self.num_projections))]
        print(self.hash_table)

        self.hyperplanes = self.generate_hyperplanes()

    # Need to generate a set of hyperplanes for each hash table
    def generate_hyperplanes(self):

        # Hyperplane vectors will have the same length as data vectors
        vector_length = (self.train_X.shape[1])
        print("Vector length: " + str(vector_length))

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

        print(hyperplanes)
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
    def get_hash_code(self, x):

        hashcode = ""

        # Getting this to work first for one hash table - CHANGE THIS LATER OR REDESIGN FUNCTION
        if self.num_hash_tables == 1:
            # print("1 hash table")
            hashcode = ""
            for hyperplane in range(len(self.hyperplanes[0])):
                dot_product = np.dot(self.hyperplanes[0][hyperplane], x)
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
        for index in range(dataset.shape[0]):
            new_hashcode = self.get_hash_code(dataset[index])
            print()
            print(dataset[index])
            print("index: " + str(index))
            print("hashcode from has_dataset: " + str(new_hashcode))
            self.hash_table[new_hashcode].append(index)#append(dataset[index])
            print(self.hash_table[new_hashcode])
            print("\n")
        
        # Convert each index of the hash table to a 2D numpy array
        # for i in range(pow(2, self.num_projections)):
        #     self.hash_table[i] = np.array(self.hash_table[i])

        print("hash table ---: " + str(self.hash_table))

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


if __name__ == '__main__':
    pass