import operator
import numpy as np

class RRTTree(object):

    def __init__(self, bb, task="mp"):

        self.bb = bb
        self.task = task
        self.vertices = {}
        self.edges = {}
        self.reverse_edges = {}

        # inspecion planning properties
        if self.task == "ip":
            self.max_coverage = 0
            self.max_coverage_id = 0

    def get_root_id(self):
        '''
        Returns the ID of the root in the tree.
        '''
        return 0

    def add_vertex(self, config, inspected_points=None):
        '''
        Add a state to the tree.
        @param config Configuration to add to the tree.
        '''
        vid = len(self.vertices)
        self.vertices[vid] = RRTVertex(config=config, inspected_points=inspected_points)

        # check if vertex has the highest coverage so far, and replace if so
        if self.task == "ip":
            v_coverage = self.bb.compute_coverage(inspected_points=inspected_points)
            if v_coverage > self.max_coverage:
                self.max_coverage = v_coverage
                self.max_coverage_id = vid

        return vid

    def add_edge(self, sid, eid, edge_cost=0):
        '''
        Adds an edge in the tree.
        @param sid stvart state ID
        @param eid end state ID
        '''
        self.edges[eid] = sid
        if sid not in self.reverse_edges:
            self.reverse_edges[sid] = []
        self.reverse_edges[sid].append(eid)
        self.vertices[eid].set_cost(cost=self.vertices[sid].cost + edge_cost)

    def remove_edge(self, sid, eid):
        self.edges.pop(eid, None)
        self.reverse_edges[sid].remove(eid)

    def is_goal_exists(self, config):
        '''
        Check if goal exists.
        @param config Configuration to check if exists.
        '''
        goal_idx = self.get_idx_for_config(config=config)
        if goal_idx is not None:
            return True
        return False

    def get_vertex_for_config(self, config):
        '''
        Search for the vertex with the given config and return it if exists
        @param config Configuration to check if exists.
        '''
        v_idx = self.get_idx_for_config(config=config)
        if v_idx is not None:
            return self.vertices[v_idx]
        return None

    def get_idx_for_config(self, config):
        '''
        Search for the vertex with the given config and return the index if exists
        @param config Configuration to check if exists.
        '''
        valid_idxs = [v_idx for v_idx, v in self.vertices.items() if (v.config == config).all()]
        if len(valid_idxs) > 0:
            return valid_idxs[0]
        return None

    def get_nearest_config(self, config):
        '''
        Find the nearest vertex for the given config and returns its state index and configuration
        @param config Sampled configuration.
        '''
        # compute distances from all vertices
        # print("get nearest config called")
        # print("config:", config)
        # print("vertices:", self.vertices)
        # print("vertices configs:", [v.config for v in self.vertices.values()])
        # def fmt_conf(x, precision=3):
        #     x = np.asarray(x, dtype=float)
        #     return np.array2string(x, precision=precision, suppress_small=True,
        #                            separator=", ", floatmode="fixed", max_line_width=10_000)
        #
        # print(f"config: {fmt_conf(config)}")
        # print("vertices configs:", [fmt_conf(v.config) for v in self.vertices.values()])
        dists = []
        for _, vertex in self.vertices.items():
            dists.append(self.bb.compute_distance(config, vertex.config))

        # retrieve the id of the nearest vertex
        vid, _ = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid].config

    def get_edges_as_states(self):
        '''
        Return the edges in the tree as a list of pairs of states (positions)
        '''

        return [[self.vertices[val].config,self.vertices[key].config] for (key, val) in self.edges.items()]

    def get_k_nearest_neighbors(self, config, k):
        '''
        Return k-nearest neighbors
        @param state Sampled state.
        @param k Number of nearest neighbors to retrieve.
        '''
        dists = []
        for _, vertex in self.vertices.items():
            dists.append(self.bb.compute_distance(config, vertex.config))

        dists = np.array(dists)
        knn_ids = np.argpartition(dists, k)[:k]
        #knn_dists = [dists[i] for i in knn_ids]
        knn_ids = np.argpartition(dists, k)[:k]
        return knn_ids.tolist(), [self.vertices[vid].config for vid in knn_ids]

    # New helpers to access stored costs
    def get_cost(self, vid):
        '''
        Return the accumulated cost stored at vertex id vid.
        '''
        return self.vertices[vid].cost

    def get_cost_for_config(self, config):
        '''
        Return cost for a vertex with given config (if exists), else None.
        '''
        v_idx = self.get_idx_for_config(config=config)
        if v_idx is not None:
            return self.vertices[v_idx].cost
        return None

    def update_subtree_cost_recursive(self, parent_id):
        """
        Update costs of all descendants of parent_id (parent cost assumed already correct).
        Works with vertex ids.
        """
        for child_id in self.reverse_edges.get(parent_id, []):
            edge_cost = self.bb.compute_distance(self.vertices[parent_id].config,
                                                 self.vertices[child_id].config)
            self.vertices[child_id].set_cost(self.vertices[parent_id].cost + edge_cost)
            self.update_subtree_cost_recursive(child_id)

    def update_subtree(self, v_id, new_parent_id, edge_cost):
        """
        Reparent vertex v_id to new_parent_id and update costs for v_id subtree.
        """
        old_parent = self.edges[v_id]
        self.remove_edge(old_parent, v_id)
        self.add_edge(new_parent_id, v_id, edge_cost=edge_cost)
        self.update_subtree_cost_recursive(v_id)

    def get_config_for_id(self, id):
        return self.vertices[id].config

class RRTVertex(object):

    def __init__(self, config, cost=0, inspected_points=None):
        self.config = config
        self.cost = cost
        self.inspected_points = inspected_points

    def set_cost(self, cost):
        '''
        Set the cost of the vertex.
        '''
        self.cost = cost