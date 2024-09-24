import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

centers = [[0, 0], [2, 2], [-3, 2], [2, -4]]
X, _ = datasets.make_blobs(n_samples=300, centers=centers, cluster_std=1, random_state=0)

class KMeans():

    def __init__(self, data, k):
        self.data = data
        self.k = k
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
        return self.assignment[i] == -1

    def initialize(self):
        indices = np.random.choice(len(self.data), size=self.k, replace=False)
        return self.data[indices]

    def make_clusters(self, centers):
        # Assign each data point to the closest center
        for i in range(len(self.data)):
            min_dist = float('inf')
            for j in range(self.k):
                dist = self.dist(centers[j], self.data[i])
                if dist < min_dist:
                    min_dist = dist
                    self.assignment[i] = j

    def compute_centers(self):
        # Compute new centers as the mean of all assigned points
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


kmeans = KMeans(X, 4)
kmeans.lloyds()
images = kmeans.snaps

images[0].save(
    'kmeans.gif',
    optimize=False,
    save_all=True,
    append_images=images[1:],
    loop=0,
    duration=500
)
