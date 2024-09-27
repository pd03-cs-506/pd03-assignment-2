import numpy as np
from PIL import Image as im
import matplotlib
import matplotlib.pyplot as plt
import random
from io import BytesIO  # Import BytesIO for in-memory operations

matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI rendering

class KMeans:
    def __init__(self, data, k, init_method, centroids=None):
        self.data = data
        self.k = k
        self.init_method = init_method
        self.centroids = centroids  # For manual initialization
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []

    def snap(self, centers):
        # Create an in-memory file using BytesIO
        buffer = BytesIO()

        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        ax.scatter(centers[:, 0], centers[:, 1], c='r', marker='x')

        # Save the plot to the in-memory file
        fig.savefig(buffer, format='png')
        plt.close(fig)  # Close the plot to avoid memory leaks

        # Seek to the beginning of the in-memory file and open it as an image
        buffer.seek(0)
        self.snaps.append(im.open(buffer))

    def initialize(self):
        if self.centroids is not None and len(self.centroids) == self.k:
            print("applying manual init")
            centers = np.array(self.centroids)
            print(f"centers are {centers}")

        elif self.init_method == 'farthest-first':
            centers = [[random.uniform(-10, 10)] * 2] 
            for _ in range(self.k - 1):
                dists = np.array([min([self.dist(point, center) for center in centers]) for point in self.data])
                next_center = self.data[np.argmax(dists)]
                centers.append(next_center)
                
        elif self.init_method == 'k-means-pp':
            centers = [self.data[random.randint(0, len(self.data) - 1)]]  # First random center
            for _ in range(self.k - 1):
                dists = np.array([min([self.dist(point, center) for center in centers]) for point in self.data])
                prob = dists / dists.sum()
                next_center = self.data[np.random.choice(len(self.data), p=prob)]
                centers.append(next_center)

        else:  # random initialization
            centers = np.array([[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(self.k)])

        return np.array(centers)

    def make_clusters(self, centers):
        for i in range(len(self.data)):
            min_dist = float('inf')
            for j in range(self.k):
                dist = self.dist(centers[j], self.data[i])
                if dist < min_dist:
                    min_dist = dist
                    self.assignment[i] = j

    def compute_centers(self):
        new_centers = []
        for i in range(self.k):
            cluster = [self.data[j] for j in range(len(self.assignment)) if self.assignment[j] == i]
            if cluster:
                new_centers.append(np.mean(cluster, axis=0))
            else:
                new_centers.append(np.random.randn(2))  # Handle empty clusters
        return np.array(new_centers)

    def unassign(self):
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        return not np.array_equal(centers, new_centers)

    def dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def lloyds(self):
        centers = self.initialize()
        self.make_clusters(centers)
        new_centers = self.compute_centers()
        self.snap(new_centers)

        while self.are_diff(centers, new_centers):
            self.unassign()
            centers = new_centers
            self.make_clusters(centers)
            new_centers = self.compute_centers()
            self.snap(new_centers)
        return

def new_dataset():
    return np.array([[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(300)])


def generate_image(dataset, k, init_method, reset_data, final, centroids=None):
    if reset_data == 1:
        dataset = new_dataset()

    # Pass user-selected centroids if provided (for manual initialization)
    kmeans = KMeans(dataset, k, init_method, centroids)
    kmeans.lloyds()  # Run the K-Means algorithm
    images = kmeans.snaps

    # Save the final convergence or the steps in a GIF
    if final == 1:
        images[-1].save(
            'static/kmeans.gif',
            format='PNG'
        )
    else:
        images[0].save(
            'static/kmeans.gif',
            save_all=True,
            append_images=images[1:],
            duration=500,
            loop=0
        )