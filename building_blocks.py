import numpy as np
import inverse_kinematics
from kinematics import Transform, UR5e_PARAMS, UR5e_without_camera_PARAMS

class BuildingBlocks3D(object):
    """
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    """

    def __init__(self, transform:Transform, ur_params:UR5e_PARAMS, env, resolution=0.1):
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

    def sample_random_config(self, goal_prob, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # TODO: HW2 5.2.1
        pass


    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return False if in collision
        @param conf - some configuration
        """
        # TODO: HW2 5.2.2- Pay attention that function is a little different than in HW2
        pass


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        """check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        """
        # TODO: HW2 5.2.4
        pass



    def compute_distance(self, conf1, conf2):
        """
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        """
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
                continue
            # check for collision 
            if not self.config_validity_checker(conf):
                continue
            # verify solution: make the difference between the solution and the original matrix and calculate the norm
            transform_base_to_end = inverse_kinematics.forward_kinematic_solution(inverse_kinematics.DH_matrix_UR5e, conf)
            diff = np.linalg.norm(np.array(original_transformation)-transform_base_to_end)
            if diff < 0.05:
                legal_conf.append(conf)
        if len(legal_conf) == 0:
            raise ValueError("No legal configurations found")       
        return legal_conf
