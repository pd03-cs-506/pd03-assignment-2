import numpy as np
from PIL import Image as im
import matplotlib
import matplotlib.pyplot as plt
import random
from io import BytesIO  

matplotlib.use('Agg') 

class KMeans:
    def __init__(self, data, k, init_method, centroids=None):
        self.data = data
        self.k = k
        self.init_method = init_method
        self.centroids = centroids  # For manual initialization
        self.assignment = [-1 for _ in range(len(data))]
        self.snaps = []

    def snap(self, centers):
        buffer = BytesIO()

        fig, ax = plt.subplots(figsize=(6.6, 6.6))
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.assignment)
        ax.scatter(centers[:, 0], centers[:, 1], c='r', marker='x')

        fig.savefig(buffer, format='png')
        plt.close(fig) 

        buffer.seek(0)
        self.snaps.append(im.open(buffer))

    def initialize(self):
        if self.centroids is not None and len(self.centroids) == self.k:
            print("applying manual init")
            centers = np.array(self.centroids)
            print(f"centers are {centers}")

        elif self.init_method == 'farthest-first':
            centers = [[random.uniform(-10, 10)] * 2] # first random center
            for _ in range(self.k - 1):
                dists = np.array([min([self.dist(point, center) for center in centers]) for point in self.data])
                next_center = self.data[np.argmax(dists)]
                centers.append(next_center)
                
        elif self.init_method == 'k-means-pp':
            centers = [[random.uniform(-10, 10)] * 2] # first random center
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
                new_centers.append(np.random.randn(2))
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

    kmeans = KMeans(dataset, k, init_method, centroids)
    kmeans.lloyds() 
    images = kmeans.snaps

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