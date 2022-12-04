import matplotlib.pyplot as plt
import numpy as np
import random


# Toy problem with 3 clusters for us to verify k-means is working well
def toyProblem():
  # Generate a dataset with 3 cluster
  X = np.random.randn(150,2)*1.5
  X[:50,:] += np.array([1,4])
  X[50:100,:] += np.array([15,-2])
  X[100:,:] += np.array([5,-2])

  # Randomize the seed
  np.random.seed()

  # Apply kMeans with visualization on
  k = 3
  max_iters=20
  centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False, smartInitialization=False)
  plotClustering(centroids, assignments, X, title="Final Clustering")
  
  # Print a plot of the SSE over training
  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.show()

  #############################
  # Q4 Randomness in Clustering
  #############################
  k = 5
  max_iters = 20

  SSE_rand = []
  # Run the clustering with k=5 and max_iters=20 fifty times and 
  # store the final sum-of-squared-error for each run in the list SSE_rand.
  # raise Exception('Student error: You haven\'t implemented the randomness experiment for Q5.')
  
  for i in range(50):
    centroids, assignments, SSE = kMeansClustering(X, k=k, max_iters=max_iters, visualize=False, smartInitialization=False)
    SSE_rand.append(SSE[0])

  # Plot error distribution
  plt.figure(figsize=(8,8))
  plt.hist(SSE_rand, bins=20)
  plt.xlabel("SSE")
  plt.ylabel("# Runs")
  plt.show()

  ########################
  # Q5 Error vs. K
  ########################

  SSE_vs_k = []
  # Run the clustering max_iters=20 for k in the range 1 to 150 and 
  # store the final sum-of-squared-error for each run in the list SSE_vs_k.

  for k in range(1, 150):
    centroids, assignments, SSE = kMeansClustering(X, k, max_iters=20, visualize=False, smartInitialization=False)
    SSE_vs_k.append(SSE[0])

  # Plot how SSE changes as k increases
  plt.figure(figsize=(16,8))
  plt.plot(SSE_vs_k, marker="o")
  plt.xlabel("k")
  plt.ylabel("SSE")
  plt.show()

  ####################################
  # Smart initialization 
  ####################################
  centroids, assignments, SSE = kMeansClustering(X, k=3, max_iters=max_iters, visualize=False, smartInitialization=True)
  # centroids, assignments, SSE = kMeansClustering(X, k=5, max_iters=max_iters, visualize=True, smartInitialization=True)
  plotClustering(centroids, assignments, X, title="Final Clustering")
  
  # Print a plot of the SSE over training
  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.show()

def imageProblem():
  np.random.seed()
  # Load the images and our pre-computed HOG features
  data = np.load("img.npy")
  img_feats = np.load("hog.npy")


  # Perform k-means clustering
  k=4
  centroids, assignments, SSE = kMeansClustering(img_feats, k, 30, min_size=0, smartInitialization=True)

  # Visualize Clusters
  for c in range(len(centroids)):
    # Get images in this cluster
    members = np.where(assignments==c)[0].astype(int)
    imgs = data[np.random.choice(members,min(50, len(members)), replace=False),:,:]
    
    # Build plot with 50 samples
    print("Cluster "+str(c) + " ["+str(len(members))+"]")
    _, axs = plt.subplots(5, 10, figsize=(16, 8))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img,plt.cm.gray)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # Fill out plot with whitespace if there arent 50 in the cluster
    for i in range(len(imgs), 50):
      axs[i].axes.xaxis.set_visible(False)
      axs[i].axes.yaxis.set_visible(False)
    plt.show()

  # Print a plot of the SSE over training
  plt.figure(figsize=(16,8))
  plt.plot(SSE, marker='o')
  plt.xlabel("Iteration")
  plt.ylabel("SSE")
  plt.text(k/2, (max(SSE)-min(SSE))*0.9+min(SSE), "k = "+str(k))
  plt.show()

##########################################################
#
# Q6 Smart Initialization
#  
###########################################################
# smartInitializeCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   k --  integer number of clusters to make
#
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#
# Initializes the centroids in a smart way using an 
# iterative furthest-from centroids heuristic
##########################################################
def smartInitializeCentroids(dataset, k):

  # n is the original number of points in the dataset
  n = dataset.shape[0]

  # First centroid is a random point from the dataset
  first_centroid_index = random.randrange(n)
  # print(first_centroid_index)
  centroids = np.array([dataset[first_centroid_index]])

  # Remove new centroid from current dataset
  remaining_points = dataset
  np.delete(remaining_points, first_centroid_index, 0)

  eucli_dist = np.linalg.norm(remaining_points - centroids[0], axis=1).reshape((dataset.shape[0], 1))

  # total_distance = np.zeros((150, 1))

  # For every new centroid:
  for i in range(0, k-1):
    
    eucli_dist += np.linalg.norm(remaining_points - centroids[i], axis=1).reshape((dataset.shape[0], 1))
    new_centroid_index = np.argmax(eucli_dist)
    new_centroid = np.array([dataset[new_centroid_index]])
    np.append(centroids, new_centroid)
    centroids = np.vstack((centroids, new_centroid))
    np.delete(eucli_dist, new_centroid_index, 0)

  return centroids

  #   for j in range(len(centroids)):

  #     # Compute distance from all points to the first centroid
  #     eucli_dist = np.linalg.norm(dataset - centroids[0], axis=1).reshape((dataset.shape[0], 1))

  # for index in range(1, centroids.shape[0]):
  #   # Add a vector containing a new set of centroid distances for every centroid
  #   eucli_dist = np.hstack((eucli_dist, np.linalg.norm(dataset - centroids[index], axis=1).reshape((dataset.shape[0], 1))))

  # # After all distances are computed, find the argmin indices for each point
  # assignments = np.argmin(eucli_dist, axis=1)

      
  # Sample k random indices and select the corresponding points to use as centroids
  # indices = random.sample(range(0, n), k)
  # centroids = dataset[indices[0]]

    # Add remaining centroids to centroid array
  # for index in range(1, k):
  #   centroids = np.vstack((centroids, dataset[indices[index]]))

##########################################################
# initializeCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   k --  integer number of clusters to make
#   smartInitialization -- if true, run the smart 
#                          initialization otherwise pick 
#                          the initial centroids randomly
#                          from the data
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
##########################################################

def initializeCentroids(dataset, k, smartInitialization=False):

  if smartInitialization:
    return smartInitializeCentroids(dataset, k)

  else:
    n = dataset.shape[0]

    # Sample k random indices and select the corresponding points to use as centroids
    indices = random.sample(range(0, n), k)
    centroids = dataset[indices[0]]

    # Add remaining centroids to centroid array
    for index in range(1, k):
      centroids = np.vstack((centroids, dataset[indices[index]]))

    return centroids

##########################################################
# computeAssignments
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#
# Outputs:
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
##########################################################

def computeAssignments(dataset, centroids):

  # Compute distance from all points to the first centroid
  eucli_dist = np.linalg.norm(dataset - centroids[0], axis=1).reshape((dataset.shape[0], 1))

  for index in range(1, centroids.shape[0]):
    # Add a vector containing a new set of centroid distances for every centroid
    eucli_dist = np.hstack((eucli_dist, np.linalg.norm(dataset - centroids[index], axis=1).reshape((dataset.shape[0], 1))))

  # After all distances are computed, find the argmin indices for each point
  assignments = np.argmin(eucli_dist, axis=1)

  return assignments


##########################################################
# updateCentroids
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j after being updated
#                 as the mean of assigned points
#   counts -- k x 1 matrix where the j'th entry is the number
#             points assigned to cluster j
##########################################################

def updateCentroids(dataset, centroids, assignments):

  new_centroids = []

  for index in range(centroids.shape[0]):
    # Reduce indices to those matching the current centroid
    assignment_indices = np.where(assignments == index)

    # Select the according datapoints from the dataset
    assignment_array = dataset[assignment_indices]

    # Calculate the mean of this array and make a new centroid at this value
    mean_value = np.mean(assignment_array, axis=0)
    new_centroids.append(mean_value)

  centroids = np.array(new_centroids)

  # Reassess the dataset based on new centroids
  assignments = computeAssignments(dataset, centroids)

  counts = []

  # Count the number of values in the new assignments array that match each centroid
  for index in range(centroids.shape[0]):
    counts.append(np.count_nonzero(assignments == index))

  counts = np.array(counts)

  return centroids, counts

##########################################################
# calculateSSE
#
# Inputs:
#   datasets -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#   assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
# Outputs:
#   sse -- the sum of squared error of the clustering
##########################################################

def calculateSSE(dataset, centroids, assignments):

  sse = 0

  for index in range(centroids.shape[0]):
    # Reduce indices to those matching the current centroid
    assignment_indices = np.where(assignments == index)

    # Select the according datapoints from the dataset
    assignment_array = dataset[assignment_indices]

    # Find the distance between each point and the centroid
    eucli_dist = np.linalg.norm(assignment_array - centroids[index], axis=1).reshape((assignment_array.shape[0], 1))

    # Sum these distances and add to SSE
    sse += np.sum(eucli_dist, axis=0)

  return sse
  

########################################
# Instructor Code: Don't need to modify 
# beyond this point but should read it
# 
########################################
##########################################################
# calculateSSE
#
# Inputs:
#   dataset -- n x d matrix of dataset points where the
#               i'th row represents x_i
#   k -- the number of clusters
#   max_iters -- the number of iterations of k-means
#   min_size -- the minimum size of a cluster
#   visualize -- true if you want to plot each iteration
#   smartInitialization -- set to false if you want a random
#                          initialization of cluster 
#                          centroids, true if you want a 
#                          smart initialization                            
# 
# Outputs:
#  centroids -- k x d matrix of centroid points where the
#                 j'th row represents c_j
#  assignments -- n x 1 matrix of indexes where the i'th 
#                  value is the id of the centroid nearest
#                  to the i'th datapoint
#  sse -- the sum of squared error of the clustering
##########################################################
def kMeansClustering(dataset, k, max_iters=10, min_size=0, visualize=False, smartInitialization=False):
  
  # Initialize centroids
  centroids = initializeCentroids(dataset, k, smartInitialization)

  # Keep track of sum of squared error for plotting later
  SSE = []

  # Main loop for clustering
  for i in range(max_iters):

    # Update Assignments Step
    assignments = computeAssignments(dataset, centroids)
    if visualize:
      plotClustering(centroids, assignments, dataset, "Iteration "+str(i))  

    # Update Centroids Step
    centroids, counts = updateCentroids(dataset, centroids, assignments)

    # Re-initalize any cluster with fewer then min_size points
    for c in range(k):
      if counts[c] <= min_size:
        centroids[c] = initializeCentroids(dataset, 1)
    
    SSE.append(calculateSSE(dataset,centroids,assignments))

    # Get final assignments
    assignments = computeAssignments(dataset, centroids)

  return centroids, assignments, SSE

def plotClustering(centroids, assignments, dataset, title=None):
  plt.figure(figsize=(8,8))
  plt.scatter(dataset[:,0], dataset[:,1], c=assignments, edgecolors="k", alpha=0.5)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=5, edgecolors="k", s=250)
  plt.scatter(centroids[:,0], centroids[:,1], c=np.arange(len(centroids)), linewidths=2, edgecolors="w", s=200)
  if title is not None:
    plt.title(title)
  plt.show()


if __name__=="__main__":
  toyProblem()
  imageProblem()
