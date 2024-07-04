import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random
import time


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0  # For RRT*


class RRTBase:
    def __init__(self, start, goal, obstacles, map_size, step_size=5, goal_sample_rate=0.1, max_iter=10000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.map_size = map_size
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.nodes = []
        self.kdtree = None

    def get_random_point(self):
        if random.random() > self.goal_sample_rate:
            return [random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1])]
        return [self.goal.x, self.goal.y]

    def get_nearest_node(self, point):
        if not self.kdtree or len(self.nodes) > len(self.kdtree.data):
            self.kdtree = KDTree([[node.x, node.y] for node in self.nodes])
        _, idx = self.kdtree.query(point)
        return self.nodes[idx]

    def steer(self, from_node, to_point):
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist > self.step_size:
            dx = dx * self.step_size / dist
            dy = dy * self.step_size / dist
        new_node = Node(from_node.x + dx, from_node.y + dy)
        new_node.parent = from_node
        return new_node

    def check_collision(self, node):
        if self.is_collision(node.x, node.y):
            return True
        if node.parent:
            for t in np.arange(0, 1, 0.1):
                x = node.parent.x + t * (node.x - node.parent.x)
                y = node.parent.y + t * (node.y - node.parent.y)
                if self.is_collision(x, y):
                    return True
        return False

    def is_collision(self, x, y):
        for obs in self.obstacles:
            if self.is_point_in_obstacle(x, y, obs):
                return True
        return False

    @staticmethod
    def is_point_in_obstacle(x, y, obstacle):
        if obstacle[0] == 'circle':
            return np.sqrt((x - obstacle[1]) ** 2 + (y - obstacle[2]) ** 2) <= obstacle[3]
        elif obstacle[0] == 'rectangle':
            return (obstacle[1] <= x <= obstacle[1] + obstacle[3] and
                    obstacle[2] <= y <= obstacle[2] + obstacle[4])
        elif obstacle[0] == 'triangle':
            return RRTBase.point_in_triangle(x, y, obstacle[1:])
        return False

    @staticmethod
    def point_in_triangle(x, y, triangle):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign((x, y), triangle[0], triangle[1])
        d2 = sign((x, y), triangle[1], triangle[2])
        d3 = sign((x, y), triangle[2], triangle[0])

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def generate_path(self, node):
        path = [[self.goal.x, self.goal.y]]
        while node.parent:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path[::-1]

    @staticmethod
    def dist(node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


class RRT(RRTBase):
    def plan(self):
        self.nodes = [self.start]
        for _ in range(self.max_iter):
            rnd = self.get_random_point()
            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)

            if not self.check_collision(new_node):
                self.nodes.append(new_node)
                if self.dist(new_node, self.goal) <= self.step_size:
                    final_node = self.steer(new_node, [self.goal.x, self.goal.y])
                    if not self.check_collision(final_node):
                        return self.generate_path(final_node)
        return None


import numpy as np

class BiRRT(RRTBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]

    def plan(self):
        for _ in range(self.max_iter):
            if len(self.start_tree) > len(self.goal_tree):
                self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
                is_start_tree = False
            else:
                is_start_tree = True

            rnd = self.get_random_point()
            new_node = self.extend(self.start_tree, rnd)
            if new_node:
                nearest_in_goal = self.get_nearest_node_from_tree(self.goal_tree, [new_node.x, new_node.y])
                if self.try_connect(new_node, nearest_in_goal):
                    if is_start_tree:
                        return self.generate_bidirectional_path(new_node, nearest_in_goal)
                    else:
                        return self.generate_bidirectional_path(nearest_in_goal, new_node)

        return None

    def extend(self, tree, point):
        nearest_node = self.get_nearest_node_from_tree(tree, point)
        new_node = self.steer(nearest_node, point)
        if not self.check_collision(new_node):
            new_node.parent = nearest_node
            tree.append(new_node)
            return new_node
        return None

    def try_connect(self, node1, node2):
        while True:
            new_node = self.steer(node1, [node2.x, node2.y])
            if self.check_collision(new_node):
                return False
            new_node.parent = node1
            if self.dist(new_node, node2) <= self.step_size:
                node2.parent = new_node
                return True
            node1 = new_node

    def get_nearest_node_from_tree(self, tree, point):
        distances = [self.dist(node, Node(point[0], point[1])) for node in tree]
        return tree[np.argmin(distances)]

    def generate_bidirectional_path(self, start_node, goal_node):
        path_start = self.generate_path(start_node)
        path_goal = self.generate_path(goal_node)
        return path_start + path_goal[::-1]

    def check_collision(self, node):
        if self.is_collision(node.x, node.y):
            return True
        if node.parent:
            for t in np.arange(0, 1, 0.1):
                x = node.parent.x + t * (node.x - node.parent.x)
                y = node.parent.y + t * (node.y - node.parent.y)
                if self.is_collision(x, y):
                    return True
        return False

    def generate_path(self, node):
        path = []
        current = node
        while current:
            path.append([current.x, current.y])
            current = current.parent
        return path[::-1]


class RRTStar(RRTBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_radius = 50

    def plan(self):
        self.nodes = [self.start]
        for _ in range(self.max_iter):
            rnd = self.get_random_point()
            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)

            if not self.check_collision(new_node):
                near_nodes = self.get_near_nodes(new_node)
                self.choose_parent(new_node, near_nodes)
                if new_node.parent:
                    self.nodes.append(new_node)
                    self.rewire(new_node, near_nodes)

                if self.dist(new_node, self.goal) <= self.step_size:
                    final_node = self.steer(new_node, [self.goal.x, self.goal.y])
                    if not self.check_collision(final_node):
                        self.goal.parent = new_node
                        self.goal.cost = new_node.cost + self.dist(new_node, self.goal)
                        return self.generate_path(self.goal)
        return None

    def get_near_nodes(self, node):
        nnode = len(self.nodes) + 1
        r = min(self.search_radius * np.sqrt((np.log(nnode) / nnode)), self.step_size * 5)
        return [n for n in self.nodes if self.dist(node, n) <= r]

    def choose_parent(self, new_node, near_nodes):
        costs = []
        for near_node in near_nodes:
            if not self.check_collision(self.steer(near_node, [new_node.x, new_node.y])):
                costs.append((near_node.cost + self.dist(near_node, new_node), near_node))
        if costs:
            min_cost, min_node = min(costs, key=lambda x: x[0])
            new_node.parent = min_node
            new_node.cost = min_cost

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            if near_node != new_node.parent:
                if not self.check_collision(self.steer(new_node, [near_node.x, near_node.y])):
                    new_cost = new_node.cost + self.dist(new_node, near_node)
                    if new_cost < near_node.cost:
                        near_node.parent = new_node
                        near_node.cost = new_cost

def draw_map(start, goal, obstacles, path=None, nodes=None):
    plt.clf()
    for obs in obstacles:
        if obs[0] == 'circle':
            circle = plt.Circle((obs[1], obs[2]), obs[3], color='black')
            plt.gca().add_artist(circle)
        elif obs[0] == 'rectangle':
            rectangle = plt.Rectangle((obs[1], obs[2]), obs[3], obs[4], color='black')
            plt.gca().add_artist(rectangle)
        elif obs[0] == 'triangle':
            triangle = plt.Polygon(obs[1:], color='black')
            plt.gca().add_artist(triangle)

    if nodes:
        node_x = [node.x for node in nodes]
        node_y = [node.y for node in nodes]
        plt.plot(node_x, node_y, '.b', markersize=1)

    plt.plot(start[0], start[1], 'bs', markersize=10)
    plt.plot(goal[0], goal[1], 'rs', markersize=10)

    if path:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('RRT Path Planning')
    plt.grid(True)


def run_trials(algorithm, n_trials=10):
    start = (10, 10)
    goal = (90, 90)
    map_size = (100, 100)
    obstacles = [
        ('rectangle', 20, 20, 20, 20),
        ('rectangle', 60, 60, 20, 20),
        ('rectangle', 60, 20, 10, 40),
        ('circle', 80, 40, 10),
        ('triangle', (30, 60), (45, 80), (60, 60))
    ]

    planning_times = []
    node_counts = []
    path_lengths = []

    for _ in range(n_trials):
        planner = algorithm(start, goal, obstacles, map_size, step_size=2, max_iter=20000)

        start_time = time.time()
        path = planner.plan()
        end_time = time.time()

        if path:
            planning_times.append(end_time - start_time)
            node_counts.append(len(planner.nodes))
            path_lengths.append(sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))))

    return planning_times, node_counts, path_lengths


# Run trials for each algorithm
algorithms = [RRT, BiRRT, RRTStar]
algorithm_names = ["RRT", "BiRRT", "RRT*"]

for alg, name in zip(algorithms, algorithm_names):
    print(f"\nResults for {name}:")
    planning_times, node_counts, path_lengths = run_trials(alg, 10)

    print(f"Planning time (MAX): {max(planning_times):.4f} seconds")
    print(f"Planning time (MIN): {min(planning_times):.4f} seconds")
    print(f"Average planning time: {np.mean(planning_times):.4f} seconds")
    print(f"Average number of nodes: {np.mean(node_counts):.2f}")
    print(f"Average path length: {np.mean(path_lengths):.2f}")

    # Generate and save a sample path for each algorithm
    start = (10, 10)
    goal = (90, 90)
    map_size = (100, 100)
    obstacles = [
        ('rectangle', 20, 20, 20, 20),
        ('rectangle', 60, 60, 20, 20),
        ('rectangle', 60, 20, 10, 40),
        ('circle', 80, 40, 10),
        ('triangle', (30, 60), (45, 80), (60, 60))
    ]

    planner = alg(start, goal, obstacles, map_size, step_size=2, max_iter=20000)
    path = planner.plan()

    # Replace these lines in your main code
    if path:
        draw_map(start, goal, obstacles, path, planner.nodes)
        plt.savefig(f'{name.replace("*", "_star")}_path_and_nodes.png')
        draw_map(start, goal, obstacles, path)
        plt.savefig(f'{name.replace("*", "_star")}_path.png')
    else:
        print(f"No path found for {name}")

    plt.close('all')  # Close all figures to free up memory

    print("\nAll trials completed. Path images have been saved.")


