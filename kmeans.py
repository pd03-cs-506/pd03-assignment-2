import numpy as np
from PIL import Image as im
import matplotlib
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import random 

matplotlib.use('Agg') 

class KMeans():
    def __init__(self, data, k, init_method):
        # randomly generated dataset of points
        self.data = data

        # number of centers
        self.k = k

        # center initialization method, string of either 'random' , 'farthest_first', 'k-means-pp', or 'manual'
        self.init_method = init_method

        # assigment for each point to a cluster (defined as a center)
        self.assignment = [-1 for _ in range(len(data))]

        self.snaps = []

    def snap(self, centers):
        TEMPFILE = "temp.png"

        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        ax.scatter(centers[:, 0], centers[:, 1], c='r', marker='x')
        fig.savefig(TEMPFILE)
        plt.close()
        self.snaps.append(im.fromarray(np.asarray(im.open(TEMPFILE))))

    def isunassigned(self, i):
        # check if data point is not assigned to a center
        return self.assignment[i] == -1

    def initialize(self):
        # INITIALIZATION METHODS
        # need to implement: random, farthest-first, k-means-pp, and user input (manual)

        # farthest first: select random first point, then select ones furthest away, and so on
        if self.init_method == 'farthest-first':
            init_index = np.random.choice(len(self.data), size=1, replace=False)

        # weighted by distance
        elif self.init_method == 'k-means-pp':
            init_index = np.random.choice(len(self.data), size=1, replace=False)
            indices = []

        # user click on points
        elif self.init_method == 'manual':
            indices = []

        else:
            # k random indices of centers
            indices = np.random.choice(len(self.data), size=self.k, replace=False)
        return self.data[indices]

    def make_clusters(self, centers):
        # assign each data point to its closest center
        for i in range(len(self.data)):
            min_dist = float('inf') # large number

            # loop through each center
            for j in range(self.k):
                dist = self.dist(centers[j], self.data[i])
                # find min dist
                if dist < min_dist:
                    min_dist = dist
                    self.assignment[i] = j

    def compute_centers(self):
        # compute new centers as the mean of all assigned points
        new_centers = []
        for i in range(self.k):
            cluster = [self.data[j] for j in range(len(self.assignment)) if self.assignment[j] == i]
            if cluster:
                new_centers.append(np.mean(cluster, axis=0))
            else:
                new_centers.append(np.random.randn(2))  # if no points, assign random point
        return np.array(new_centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        # Check if the centers have changed
        return not np.array_equal(centers, new_centers)

    def dist(self, x, y):
        # Euclidean distance
        return np.sqrt(np.sum((x - y) ** 2))

    def lloyds(self):
        centers = self.initialize()

        # Assign each point to centers
        self.make_clusters(centers)

        # Compute center average
        new_centers = self.compute_centers()

        self.snap(new_centers)

        # Compare if new center is same as before
        while self.are_diff(centers, new_centers):
            self.unassign()  # Unassign at the beginning
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)

        return

# centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
dataset = np.array([[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(300)])

def generate_image(dataset, k, init_method, reset_data, final):
    if (reset_data == 1):
        # centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
        dataset = np.array([[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(300)])

    kmeans = KMeans(dataset, k, init_method)
    kmeans.lloyds()
    images = kmeans.snaps

    if (final == 1): 
        return images[-1].save(
            'static/kmeans.gif',
            optimize=False,
            save_all=False,
            format='PNG'
        )
    else: 
        return images[0].save(
            'static/kmeans.gif',
            optimize=False,
            save_all=True,
            append_images=images[1:],
            duration=500
        )

generate_image(dataset, 2, 'random', 0, 1)