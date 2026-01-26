import numpy as np
from RRTTree import RRTTree
from RRTMotionPlanner import RRTMotionPlanner
import time
import math

class RRTStarPlanner(RRTMotionPlanner):

    def __init__(
        self,
        bb,
        max_step_size,
        ext_mode="E2",
        start=None,
        goal=None,
        max_itr=None,
        stop_on_goal=None,
        k=None,
        goal_prob=0.1,
        visualizer=None,
        max_plan_time=250.0,
        k_mult=10,
        goal_configs = []
    ):
        # Initialize base class (sets bb, tree, start, goal, ext_mode, goal_prob, visualizer)
        super().__init__(bb, ext_mode, goal_prob, start, goal, visualizer)

        # RRT* specific params
        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal
        self.k = k
        self.max_step_size = max_step_size
        self.max_plan_time = max_plan_time
        self.k_mult = k_mult
        self.goal_configs = goal_configs

    def dist(self, config1, config2):
        return self.bb.compute_distance(config1, config2)

    def add_node(self, near_id, near_config, new_config):
        """
        Add a node steered from the vertex with id near_id towards x_rand.
        after adding the node:
            check if the path to it can be improved by other ~log(n) neighbors in radius R
            check if it can improve the path to other ~log(n) neighbors in radius R.
        Returns new_id or None on failure/invalid.
        """
        if not RRTMotionPlanner.validate_checks(self, near_config=near_config, new_config=new_config):
            return None

        parent_id, parent_config = near_id, near_config
        new_config_cost = self.tree.get_cost(near_id) + self.dist(near_config, new_config)
        k = self.k_mult * int(np.ceil(np.log(max(len(self.tree.vertices), 2))))
        k = min(k, len(self.tree.vertices)-1)
        if k >= 2: # search for best parent
            knn_ids, knn_configs = self.tree.get_k_nearest_neighbors(new_config, k)
            for v_id, v_config in zip(knn_ids, knn_configs):
                better_cost = self.tree.get_cost(v_id) + self.dist(new_config, v_config)
                if better_cost < new_config_cost and self.bb.edge_validity_checker(v_config, new_config):
                    new_config_cost = better_cost
                    parent_id, parent_config = v_id, v_config

        # add node to tree with best parent
        new_id = self.tree.add_vertex(new_config)
        edge_cost = self.dist(parent_config, new_config)
        self.tree.add_edge(parent_id, new_id, edge_cost=edge_cost)

        if k >= 3: # make node parent of other nodes (RRT* improvement phase)
            for v_id, v_config in zip(knn_ids, knn_configs):
                edge_cost = self.dist(new_config, v_config)
                better_cost = new_config_cost + edge_cost
                if (better_cost < self.tree.get_cost(v_id)):
                    # print("better cost")
                    if self.bb.edge_validity_checker(new_config, v_config):
                        # improve tree by replacing the edge to v with a new edge.
                        old = self.tree.get_cost(v_id)
                        self.tree.update_subtree(v_id, new_id, edge_cost)
                        new = self.tree.get_cost(v_id)
                        print("rewire", v_id, "old", old, "new", new)
        return new_id

    def plan(self):
        """
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        """
        # HW3 3
        self.tree = RRTTree(self.bb)
        root_id = self.tree.add_vertex(np.asarray(self.start, dtype=float))
        avg_time_secs = 50.0 # TODO: set the time we got from earlier parts (50.0?)
        max_time_secs = 5.0 * avg_time_secs
        goal_ids = []
        iteration = 0
        start_time = time.time()
        goal_configs = list(self.goal_configs)

        # run until time budget expires
        print("begin planning")
        print("start conf = " + str(self.start), ", goal confs = " + str(goal_configs))
        while ((time.time() - start_time) < max_time_secs and (
                not (self.max_itr is not None and iteration > self.max_itr))
                ):
                # ) or (len(goal_ids) == 0):
            iteration += 1
            if iteration % 100 == 0:
                print("RRT* iteration:", iteration, ", time elapsed (s):", round(time.time() - start_time, 2))
                print("remaining goals: ", goal_configs)
            rand_configs = self.bb.sample_random_configs(self.goal_prob, goal_configs)
            if len(rand_configs) > 1:
                print("sampled goals. num gaols: ", len(rand_configs))
            for rand_config in rand_configs:
                # print("rand config: ", rand_config)
                near_id, near_config = self.tree.get_nearest_config(rand_config)
                new_config = self.extend(near_config, rand_config)
                new_id = self.add_node(near_id, near_config, new_config)
                if new_id is not None and len(rand_configs) > 1:
                    print("added new node id in direction of goal =", new_id, ", config =", new_config)
                if new_id is not None and any(np.array_equal(new_config, g) for g in goal_configs):
                    goal_ids.append(new_id)
                    # remove found goal to avoid duplicates
                    goal_configs = [g for g in goal_configs if not np.array_equal(new_config, g)]
                    print("goal ids = " + str(goal_ids))
                    if self.stop_on_goal:
                        break

                # if self.visualizer is not None and iteration % 5 == 0:
                #     self.visualizer.visualize_sampling(rand_config, near_config, new_config, self.tree, self.start, self.goal)

        print("finished planning. goal_ids =", goal_ids)
        if goal_ids is None:
            return [np.array([], dtype=float), np.inf]

        best_cost = np.inf
        best_id = None
        for goal_id in goal_ids:
            cost = self.tree.get_cost(goal_id)
            if cost < best_cost:
                best_id, best_cost = goal_id, cost

        path = self.reconstruct_path(best_id)

        return np.array(path, dtype=float), best_cost, self.tree.get_config_for_id(best_id)

    def compute_cost(self, plan):
        # HW3 3
        return RRTMotionPlanner.compute_cost(self, plan)

    def extend(self, x_near, x_rand):
        # HW3 3
        return RRTMotionPlanner.extend(self, x_near, x_rand, self.max_step_size)
    
    def plan_with_stats(self, log_every=50):
        """
        Same logic as plan(), but also logs:
        - best/ current goal cost vs iteration
        - success indicator vs iteration (0/1)

        Returns:
            path (np.array)
            iters (np.array)          logged iteration numbers
            costs (np.array)          cost at goal_id (np.inf if no solution yet)
            success (np.array)        1 if goal found by that iteration else 0
        """
        self.tree = RRTTree(self.bb)
        root_id = self.tree.add_vertex(np.asarray(self.start, dtype=float))

        max_time_secs = self.max_plan_time
        goal_id = None
        iteration = 0
        start_time = time.time()

        iters = []
        costs = []
        success = []

        # run until time budget expires (same as plan)
        while (iteration < 2000):

            iteration += 1
            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)
            near_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)
            new_id = self.add_node(near_id, near_config, new_config)

            # SAME goal condition as plan()
            if (new_config == np.asarray(self.goal, dtype=float)).all():
                goal_id = new_id
                if self.stop_on_goal:
                    break

            # log at constant iteration intervals
            if iteration == 1 or (iteration % log_every == 0):
                iters.append(iteration)
                if goal_id is None:
                    costs.append(np.inf)
                    success.append(0)
                else:
                    # IMPORTANT: use current tree cost to goal_id (may improve due to rewiring)
                    print(f"RRT* iteration {iteration}: best cost = {self.tree.get_cost(goal_id)}")
                    costs.append(self.tree.get_cost(goal_id))
                    success.append(1)

        # final log if last iteration wasn't logged
        if len(iters) == 0 or iters[-1] != iteration:
            iters.append(iteration)
            if goal_id is None:
                costs.append(np.inf)
                success.append(0)
            else:
                costs.append(self.tree.get_cost(goal_id))
                success.append(1)

        if goal_id is None:
            return np.array([], dtype=float), np.array(iters), np.array(costs), np.array(success)

        path = self.reconstruct_path(goal_id)
        return np.array(path, dtype=float), np.array(iters), np.array(costs), np.array(success)

    def find_path(self, start_conf, goal_confs, manipulator=None):
        # TODO: use manipulator somehow. idk why it is provided here.
        self.start = start_conf
        self.goal_configs = goal_confs
        return self.plan()