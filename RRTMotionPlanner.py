from matplotlib.pylab import rand
import numpy as np
from RRTTreeHW4 import RRTTree
import time


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob, start, goal, visualizer=None):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        
        # set visualizer for real-time visualization
        self.visualizer = visualizer

    def validate_checks(self, near_config, new_config):
        if new_config is None or (
                not self.bb.config_validity_checker(new_config)) or (
                not self.bb.edge_validity_checker(near_config, new_config)):
            return False
        return True

    def reconstruct_path(self, goal_id):
        path = []
        curr = goal_id
        while True:
            path.append(self.tree.vertices[curr].config)
            if curr == self.tree.get_root_id():
                break
            curr = self.tree.edges[curr]
        path.reverse()
        return path

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # HW3 2.2.3
        start_time = time.time()

        # Reset tree each run
        self.tree = RRTTree(self.bb, task="mp")

        # Add start as root
        root_id = self.tree.add_vertex(np.asarray(self.start, dtype=float))

        iteration = 0
        while True:
            iteration += 1
            #print(f"Tree size: {len(self.tree.vertices)}")
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)
            
            # Visualize the sampling in real-time (every 5 iterations to avoid slowdown)
            if self.visualizer is not None and iteration % 5 == 0:
                self.visualizer.visualize_sampling(rand_config, near_config, new_config, self.tree, self.start, self.goal)
            
            if not self.validate_checks(near_config, new_config):
                continue

            new_id = self.tree.add_vertex(new_config)
            edge_cost = self.bb.compute_distance(near_config, new_config)
            self.tree.add_edge(near_id, new_id, edge_cost=edge_cost)

            # Stop only when the goal was actually added to the tree
            if (new_config == np.asarray(self.goal, dtype=float)).all():
                goal_id = new_id
                break

        # Reconstruct path
        path = self.reconstruct_path(goal_id)

        self.plan_time = time.time() - start_time  
        return np.array(path, dtype=float)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        #  HW3 2.2.2
        total_cost = 0
        for i in range(len(plan) - 1):
            total_cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return total_cost

    def extend(self, near_config, rand_config, max_step_size = 0.7):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        # HW3 2.2.1
        near = np.asarray(near_config, dtype=float)
        rand = np.asarray(rand_config, dtype=float)

        diff = rand - near
        dist = float(np.linalg.norm(diff))
        if dist == 0.0:
            return None

        if self.ext_mode == "E1":
            return rand

        if self.ext_mode == "E2":
            eta = max_step_size  # Small step size in radians
            if dist <= eta:
                return rand 
            step = (eta / dist) * diff
            return near + step
        
