import numpy as np
import time

import environment
from RRTTree import RRTTree

class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)
        # testing variables
        self.t_curr = 0
        self.itr_no_goal_limit = 250
        self.sample_rotation = 0.1
        # self.TWO_PI = 2 * math.pi
        self.last_cost = -1
        self.last_ratio = 1

    def find_path(self, start_conf, goal_conf):
        # TODO: HW3 3
        pass

    def extend(self, x_near, x_random):
        '''
        Implement the Extend method
        @param x_near - Nearest Neighbor
        @param x_random - random sampled configuration
        return the extended configuration
        '''
        # TODO: HW3 3
        pass