import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
import time
from dataclasses import field
import heapq
from math import dist
from tqdm import tqdm
from dataclasses import dataclass

NOISE_ID = -999
UNVISITED_ID = -1

@dataclass(slots=True)
class Point:
    value: list[list[float]]
    index: int
    cluster_id: int

class DataBase:
    def __init__(self):
        self.index_store = NearestNeighbors()
        self.points: list[Point] = []

    def fit(self, data_points: list[list[float]]):
        self.index_store = self.index_store.fit(data_points)
        self.points = [Point(value, i, UNVISITED_ID) for i, value in
                                    enumerate(data_points)]

    def radius_neighbors(self, point: list[list[int]], radius: float) -> list[Point]:
        # print(point)
        distances, indeces = self.index_store.radius_neighbors([point], radius)
        # print(indeces)
        near_points = [self.points[i] for i in indeces[0]]
        return near_points

    def _count_occurrences(self, clusters: list[int]) -> dict[int, int]:
        occurrence_dict = {}
        for num in clusters:
            if num in occurrence_dict:
                occurrence_dict[num] += 1
            else:
                occurrence_dict[num] = 1
        return dict(sorted(occurrence_dict.items(), key=lambda item: item[1], reverse=True))


    def __getitem__(self, index: int):
        return self.points[index]

    @property
    def size(self) -> int:
        return len(self.points)

    def stats(self) -> None:
        clusters = [point.cluster_id for point in self.points]
        occurences = self._count_occurrences(clusters)
        for key, value in occurences.items():
            print(f"Cluster {key} has {value} points in it")

    def plot_clusters(self, xlabel: str, ylabel: str, col1: int, col2: int) -> None:
        points = np.array([point.value for point in self.points])
        clusters = [point.cluster_id for point in self.points]
        unique_clusters = np.unique(clusters)

        colors = {cluster: (random.random(), random.random(), random.random()) for cluster in unique_clusters}

        # Convert points to x and y coordinates
        x_coords = np.array(points)[:, col1]
        y_coords = np.array(points)[:, col2]

        # Create a scatter plot with small and half-visible points, assigning random colors based on clusters
        for cluster_id in unique_clusters:
            mask = np.array(clusters) == cluster_id
            cluster_points = points[np.array(clusters) == cluster_id]


            plt.scatter(cluster_points[:, col1], cluster_points[:, col2],
                        color=colors[cluster_id], label=f'Cluster {cluster_id}',
                        s=50, alpha=0.99, edgecolor='k')

        # Create a discrete legend
        plt.legend(title='Clusters')

        # Add labels and title
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Scatter Plot of Points with Random Colors per Cluster')

        # Show the plot
        plt.show()

# first value in db's points is index of a point

def _expand_cluster(db, point, point_index: int, cluster_id: int, eps_radius: float, min_pts_density: int) -> bool:
    # Returns False if point is noise and True if it is a member of cluster
    seeds = db.radius_neighbors(point.value, radius=eps_radius)
    if len(seeds) < min_pts_density:
        point.cluster_id = NOISE_ID
        return False
    else:
        # all points in seeds are density-reachable from *point*
        for seed in seeds:
            seed.cluster_id = cluster_id

        for current_point in seeds:
            result = db.radius_neighbors(current_point.value, eps_radius)

            if len(result) >= min_pts_density:
                # resultP
                for neighbor_point in result:
                    if neighbor_point.cluster_id in (NOISE_ID, UNVISITED_ID):
                        if neighbor_point.cluster_id == UNVISITED_ID:
                            seeds.append(neighbor_point)
                        neighbor_point.cluster_id = cluster_id
        return True


def dbscan(db, eps_radius: float, min_pts_density: int):
    cluster_id = 0
    for i in range(db.size):
        point = db[i]
        if point.cluster_id == UNVISITED_ID:
            if _expand_cluster(db, point, i, cluster_id ,eps_radius, min_pts_density):
                cluster_id += 1

    return db
