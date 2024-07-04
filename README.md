# Improved-RRT-global-path-planning-algorithm-based-on-Bridge-Test

This repository contains implementations of various Rapidly-exploring Random Tree (RRT) algorithms and their improved versions, including:

RRT (Rapidly-exploring Random Tree)
BiRRT (Bidirectional RRT)
RRT (RRT Star)*
Improved RRT
These algorithms are commonly used for path planning in robotics and other applications requiring efficient navigation in complex environments.

Contents
RRT, BiRRT, RRT*
Improved RRT
Description
1. RRT, BiRRT, RRT*
The RRT,BiRRT,RRT_star.py file contains implementations of the basic RRT algorithm along with its bidirectional and optimal (RRT*) variants. These algorithms work as follows:

RRT: The standard RRT algorithm incrementally builds a tree by randomly sampling the space and extending the nearest node towards the sampled point.
BiRRT: This variant grows two trees simultaneously from the start and goal nodes and attempts to connect them, reducing the time required to find a path.
RRT*: RRT* improves upon RRT by continuously optimizing the path, ensuring asymptotic optimality.
2. Improved RRT
The Improved RRT.py file contains an enhanced version of the RRT algorithm, incorporating several improvements:

Narrow Passage Detection: Identifies and navigates through narrow passages in the environment.
KDTree for Nearest Neighbor Search: Utilizes KDTree for efficient nearest neighbor search.
Collision Detection Enhancements: Implements advanced collision detection to ensure path feasibility.
Usage
Dependencies
Ensure you have the following dependencies installed:

numpy
matplotlib
scipy
You can install them using pip:

sh
Copy code
pip install numpy matplotlib scipy
Running the Code
Each script contains a main function that demonstrates the use of the implemented algorithms. You can run the scripts as follows:

sh
Copy code
python RRT,BiRRT,RRT_star.py
python Improved\ RRT.py
Visualization
The algorithms include functions to visualize the path planning process using Matplotlib. The generated paths and nodes can be plotted to observe the behavior of the algorithms in different scenarios.

Examples
Improved RRT
The Improved RRT.py script includes a sample environment with various obstacles and demonstrates the path planning process. The script runs multiple trials and provides statistics on planning time, number of nodes, and path length.

Contributions
Contributions to this repository are welcome. Feel free to open issues or submit pull requests to improve the algorithms or add new features.

Acknowledgements
This repository is part of a literature survey on RRT algorithms and their variants. The implementations are based on various research papers and online resources.
