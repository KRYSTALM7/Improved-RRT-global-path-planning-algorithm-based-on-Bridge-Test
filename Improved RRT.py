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


class ImprovedRRT:
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

    def plan(self):
        self.nodes = [self.start]
        narrow_passages = self.find_narrow_passages()
        for passage in narrow_passages:
            self.nodes.append(Node(passage[0], passage[1]))

        self.kdtree = KDTree([[node.x, node.y] for node in self.nodes])

        for _ in range(self.max_iter):
            if random.random() > self.goal_sample_rate:
                rnd = self.get_random_point()
            else:
                rnd = [self.goal.x, self.goal.y]

            nearest_node = self.get_nearest_node(rnd)
            new_node = self.steer(nearest_node, rnd)

            if not self.check_collision(new_node):
                self.nodes.append(new_node)
                self.kdtree = KDTree([[node.x, node.y] for node in self.nodes])  # Update KDTree

                if self.dist(new_node, self.goal) <= self.step_size:
                    final_node = self.steer(new_node, [self.goal.x, self.goal.y])
                    if not self.check_collision(final_node):
                        return self.generate_path(final_node)

        return None

    def find_narrow_passages(self):
        passages = []
        for _ in range(100):
            p = self.get_random_point()
            if self.cross_test(p):
                passages.append(p)
        return passages

    def cross_test(self, p):
        r = self.step_size * 2
        for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r)]:
            if self.is_collision(p[0] + dx, p[1] + dy) and self.is_collision(p[0] - dx, p[1] - dy):
                return True
        return False

    def get_random_point(self):
        return [random.uniform(0, self.map_size[0]), random.uniform(0, self.map_size[1])]

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

        if not self.check_collision(new_node):
            return new_node

        return None

    def check_collision(self, node):
        if node is None:
            return True

        if self.is_collision(node.x, node.y):
            return True

        if node.parent:
            dx = node.x - node.parent.x
            dy = node.y - node.parent.y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            step = min(self.step_size / 10, 0.5)  # Check collision more frequently
            steps = int(dist / step)
            for i in range(1, steps + 1):
                t = i / steps
                x = node.parent.x + t * dx
                y = node.parent.y + t * dy
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
            return ImprovedRRT.point_in_triangle(x, y, obstacle[1:])
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
    plt.title('Improved RRT Path Planning')
    plt.grid(True)


def run_trials(n_trials=10):
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
        rrt = ImprovedRRT(start, goal, obstacles, map_size, step_size=2, max_iter=20000)

        start_time = time.time()
        path = rrt.plan()
        end_time = time.time()

        if path:
            planning_times.append(end_time - start_time)
            node_counts.append(len(rrt.nodes))
            path_lengths.append(sum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))))

    return planning_times, node_counts, path_lengths


# Run trials and calculate statistics
planning_times, node_counts, path_lengths = run_trials(10)

print(f"Planning time (MAX): {max(planning_times):.4f} seconds")
print(f"Planning time (MIN): {min(planning_times):.4f} seconds")
print(f"Average planning time: {np.mean(planning_times):.4f} seconds")
print(f"Average number of nodes: {np.mean(node_counts):.2f}")
print(f"Average path length: {np.mean(path_lengths):.2f}")

# Draw map with path for the last trial
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

rrt = ImprovedRRT(start, goal, obstacles, map_size, step_size=1, max_iter=20000)
path = rrt.plan()

if path:
    print("Path found!")
    draw_map(start, goal, obstacles, path, rrt.nodes)
    plt.savefig('map_with_path_and_nodes.png')
    draw_map(start, goal, obstacles, path)
    plt.savefig('map_with_path.png')
else:
    print("No path found.")

plt.show()