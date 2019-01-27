import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set() # for plot styling

from sklearn.datasets.samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50)

####################################
#  k-mean clustering algorithm :
#  * The "cluster center" is the arithmetic mean of all the points beloinging to the cluster.
#  * Each point is closer to its own cluster center than to other cluster centers.
####################################
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


###################################
#  k-means Algorithm : Expectation-Maximization
#  * Guess some cluster centers
#  * Repeat until converged
#     1. E-Step : assign points to the nearest cluster center
#     2. M-Step : set the cluster centers to the mean
###################################

#print("random = ", rng, rng.permutation(300))

from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_clusters, rseed=2):
    #1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    #Can be used to get the random number up to certain limit as list

    centers = X[i] #X[a,b,c,d] => X[a], X[b], X[c], X[d]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels==i].mean(0) for i in range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

centers, labels = find_clusters(X, 4)

plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()


#############################################
#   k-means is limited to linear cluster boundaries
#   Way to use the same trick to allow k-means to discover non-linear boundaries
#   One version of this kernelized k-means is implmented in scikit-learn within SpectralClustering estimator.
#   It uses the graph of nearest neighbors to compute a higher-dimensional representation of the data,
#   and then assigns labels using a k-means algorithm
###############################################

from sklearn.datasets import make_moons
X, y = make_moons(200, noise=0.05, random_state=0)

labels = KMeans(2, random_state=0).fit_predict(X)

plt.figure(3)
plt.subplot(1,2,1)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(X)
plt.subplot(1,2,2)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
