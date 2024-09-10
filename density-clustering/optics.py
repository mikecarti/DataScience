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

from dataclasses import field
import heapq
from math import dist

from dbscan import Point, DataBase

@dataclass(slots=True)
class OptPoint(Point):
    reachability_distance: float = field(default=None)  # default to infinity (undefined)
    processed: bool = field(default=False)

class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.entry_finder = {}  # Mapping of points to their entries in the heap
        self.REMOVED = '<removed-point>'
        self.counter = 0

    def add_point(self, point: OptPoint, priority=0.0):
        """Add a new point or update the priority of an existing point"""
        if point.index in self.entry_finder:
            self.remove_point(point)
        entry = [priority, self.counter, point]
        self.entry_finder[point.index] = entry
        heapq.heappush(self.heap, entry)

        self.counter += 1

    def remove_point(self, point: OptPoint):
        """Mark an existing point as REMOVED"""
        entry = self.entry_finder.pop(point.index)
        entry[-1] = self.REMOVED

    def move_up(self, point: OptPoint, new_priority):
        """Update the priority of a point and move it up"""
        self.add_point(point, new_priority)  # Re-insert with updated priority

    def pop_point(self):
        """Remove and return the lowest priority point"""
        while self.heap:
            priority, count, point = heapq.heappop(self.heap)
            if point is not self.REMOVED:
                del self.entry_finder[point.index]
                return point
        raise KeyError('pop from an empty priority queue')


class OpticsDataBase(DataBase):
    def __init__(self):
        super().__init__()

    def fit(self, data_points: list[list[float]]):
        self.index_store = self.index_store.fit(data_points)
        self.points = [OptPoint(value, i, UNVISITED_ID) for i, value in
                                    enumerate(data_points)]
        self.ordering = None

    def min_pts_distance(self, point: list[float], k: int):
        index_store: NearestNeighbors = self.index_store
        dist, neighbors = index_store.kneighbors([point], k, return_distance=True)
        largest_dist = dist.max()
        return largest_dist

    def get_reachability_distances(self) -> tuple[list[list[float]], list[float]]:
        # returns ordered points
        points = [point.value for point in self.ordering]
        reach_dist = [point.reachability_distance for point in self.ordering]

        return points, reach_dist

    def plot_reachability(self):
        _, reachability = self.get_reachability_distances()

        undefined_value = round(4 * max([r for r in reachability if r is not None]))

        # Replace None values with the undefined_value (for UNDEFINED)
        reachability = [undefined_value if r is None else r for r in reachability]

        # Create the cluster order based on the number of reachability values
        cluster_order = np.arange(len(reachability))

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_order, y=reachability, color='black')

        # Labels and title
        plt.xlabel("Cluster-order of the objects")
        plt.ylabel("Reachability-distance")
        plt.title("Reachability Plot")

        # Mark y=undefined_value as UNDEFINED
        plt.axhline(y=undefined_value, color='gray', linestyle='--')
        plt.text(len(cluster_order) - 1, undefined_value, 'UNDEFINED', va='bottom', ha='right', color='gray')

        # Remove x-tick labels (but keep the axis for structure)
        plt.xticks([])

        # Add legend with eps and min_pts
        plt.legend([f"eps = {self.eps}, min_pts = {self.min_pts}"], loc='upper right')

        # Display the plot
        plt.show()

def calc_core_distance(db: OpticsDataBase, point: list[float], eps_radius: float, min_pts: int) -> float:
    epsilon_cardinality = len(db.radius_neighbors(point, eps_radius))

    if epsilon_cardinality < min_pts:
        return None # UNDEFINED
    else:
        return db.min_pts_distance(point, min_pts) #min_pts-distance 

def update_seeds(db: OpticsDataBase, neighbors: np.ndarray, point: OptPoint, seeds: list, eps_radius: float, min_pts: int):
    core_dist = calc_core_distance(db, point.value, eps_radius, min_pts)

    for neighbor in neighbors:
        if neighbor.processed is False:
            new_reach_dist = max(core_dist, dist(point.value, neighbor.value))
            # print(new_reach_dist)

            if neighbor.reachability_distance == None: # o s not in seeds
                neighbor.reachability_distance = new_reach_dist
                seeds.add_point(neighbor, new_reach_dist)
            else:
                if new_reach_dist < neighbor.reachability_distance:
                    neighbor.reachability_distance = new_reach_dist
                    seeds.move_up(neighbor, new_reach_dist)


def expand_cluster_order(db: OpticsDataBase, point: OptPoint, eps_radius: float, min_pts_density: int):
    neighbors = db.radius_neighbors(point.value, radius=eps_radius)
    
    point.processed = True
    point.reachability_distance = None # UNDEFINED
    point_core_distance = calc_core_distance(db, point.value, eps_radius, min_pts_density)
    db.ordering.append(point)

    if point_core_distance != None: 
        seeds = PriorityQueue()
        update_seeds(db, neighbors, point, seeds, eps_radius, min_pts_density)

        for entry in seeds.heap:
            _, _, other_point = entry
            if other_point == seeds.REMOVED:
                continue

            other_neighbors = db.radius_neighbors(other_point.value, radius=eps_radius)
            other_point.processed = True

            db.ordering.append(other_point)
            
            if calc_core_distance(db, other_point.value, eps_radius, min_pts_density) != None:
                update_seeds(db, other_neighbors, other_point, seeds, eps_radius, min_pts_density)
            

def optics(db: OpticsDataBase, eps_radius: float, min_pts_density: int):
    db.ordering: list[int] = list()

    db.eps = eps_radius
    db.min_pts = min_pts_density

    for i in tqdm(range(db.size), desc="Processing Data Points"):
        point = db[i]
        if not point.processed:
            expand_cluster_order(db, point, eps_radius, min_pts_density)

    return db
