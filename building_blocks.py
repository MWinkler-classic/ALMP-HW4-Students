import numpy as np
import inverse_kinematics
from kinematics import Transform, UR5e_PARAMS, UR5e_without_camera_PARAMS

class BuildingBlocks3D(object):
    """
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    """

    def __init__(self, transform:Transform, ur_params:UR5e_PARAMS, env, p_bias=0.05, resolution=0.1):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [
            ["shoulder_link", "forearm_link"],
            ["shoulder_link", "wrist_1_link"],
            ["shoulder_link", "wrist_2_link"],
            ["shoulder_link", "wrist_3_link"],
            ["upper_arm_link", "wrist_1_link"],
            ["upper_arm_link", "wrist_2_link"],
            ["upper_arm_link", "wrist_3_link"],
            ["forearm_link", "wrist_2_link"],
            ["forearm_link", "wrist_3_link"],
        ]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # HW2 5.2.1
        uni_sample = np.random.uniform(low=0.0, high=1.0, size=None)
        if uni_sample <= goal_prob:
            # return goal_conf
            # print("returning goal conf")
            # print("goal conf: ", goal_conf)
            return goal_conf
        else:
            rand = np.random.uniform(low=-self.single_mechanical_limit, high=self.single_mechanical_limit, size=6)
            return rand

    def _convert_to_3d_spheres(self, link_spheres):
        new_spheres = []
        for sphere in link_spheres:
            sphere = np.array(sphere, dtype=np.float64).reshape(-1)
            new_spheres.append(sphere[:3])
        return np.vstack(new_spheres)

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
            # TODO: check for robot-robot collisiosn
                # idea: check robot-robot collisions by creating a single sphere for each link, and check for sphere-sphere collisions.

        # old code:
        sphere_coords = self.transform.conf2sphere_coords(conf)
        # print("sphere_coords: ", sphere_coords)
        radii = self.ur_params.sphere_radius


        # ------------------------------------------------------------
        # 2) Self-collision: link-link
        # ------------------------------------------------------------
        for link1, link2 in self.possible_link_collisions:
            # s1 = np.asarray(sphere_coords[link1])[:, :3]  # (n1, 3)
            # s2 = np.asarray(sphere_coords[link2])[:, :3]  # (n2, 3)
            s1 = self._convert_to_3d_spheres(sphere_coords[link1])  # (n1, 3)
            s2 = self._convert_to_3d_spheres(sphere_coords[link2])  # (n2, 3)
            r_sum = radii[link1] + radii[link2]

            # pairwise distances: broadcasting
            # result shape: (n1, n2)
            # print("s1:", s1)
            dists = np.linalg.norm(s1[:, None, :] - s2[None, :, :], axis=2)

            if np.any(dists < r_sum):
                # print("self-collision detected between " + link1 + " and " + link2)
                return False

        # ------------------------------------------------------------
        # 3) Floor + obstacle collisions
        # ------------------------------------------------------------
        obstacles = self.env.obstacles
        obs_r = self.env.radius

        for link in self.ur_params.ur_links:
            spheres = self._convert_to_3d_spheres(sphere_coords[link])
            r_link = radii[link]

            # floor collision (except shoulder)
            if link != "shoulder_link":
                if np.any(spheres[:, 2] < r_link):
                    # print("floor collision detected on link " + link)
                    return False

            # obstacle collision
            if obstacles is not None and len(obstacles) > 0:
                dists = np.linalg.norm(
                    spheres[:, None, :] - obstacles[None, :, :],
                    axis=2
                )
                if np.any(dists < (r_link + obs_r)):
                    # print("obstacle collision detected on link " + link)
                    return False

        return True


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # HW2 5.2.4
        res = min(0.5, self.resolution)
        progress = 0
        # iters = 0
        while True:
            # iters+=1
            conf = prev_conf * (1.0 - progress) + current_conf * progress
            if not self.config_validity_checker(conf):
                # print("iters = " +str(iters))
                return False
            if progress == 1.0:
                # print("iters = " + str(iters))
                return True
            progress += res
            progress = min(progress, 1.0) # do one last iteration, for the final config.



    def compute_distance(self, conf1, conf2):
        """
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        """
        # print("conf1: ", conf1)
        # print("conf2: ", conf2)
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5

    def validate_IK_solutions(self,configurations : np.array, original_transformation):
        
        legal_conf = []
        limits = list(self.ur_params.mechamical_limits.values())
        for conf in configurations:
            # check for angles limits
            valid_angles = True
            for i, angle in enumerate(conf):
                if not (limits[i][0] <= angle <= limits[i][1]):
                    valid_angles = False
                    break
            if not valid_angles:
                print("no valid angles")
                continue
            # check for collision 
            if not self.config_validity_checker(conf):
                print("collision detected")
                continue
            # verify solution: make the difference between the solution and the original matrix and calculate the norm
            transform_base_to_end = inverse_kinematics.forward_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, conf)
            diff = np.linalg.norm(np.array(original_transformation)-transform_base_to_end)
            if diff < 0.05:
                legal_conf.append(conf)
            else:
                print("solution verification failed, diff: ", diff)
        if len(legal_conf) == 0:
            raise ValueError("No legal configurations found")       
        return legal_conf
