import json
import time
from enum import Enum
import numpy as np
from matplotlib.sankey import RIGHT

from environment import Environment

from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import BuildingBlocks3D
from visualizer import Visualize_UR

import inverse_kinematics

from environment import LocationType


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    with open(dir_path + 'output.txt', 'a') as file:
        file.write(f"{written_log}\n")


class Gripper(str, Enum):
    OPEN = "OPEN",
    CLOSE = "CLOSE"
    STAY = "STAY"




def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


class Experiment:
    def __init__(self, cubes=None):
        # environment params
        self.cubes = cubes
        self.right_arm_base_delta = 0  # deviation from the first link 0 angle
        self.left_arm_base_delta = 0  # deviation from the first link 0 angle
        self.right_arm_meeting_safety = None
        self.left_arm_meeting_safety = None

        # tunable params
        self.max_itr = 1000
        self.max_step_size = 0.5
        self.goal_bias = 0.05
        self.resolution = 0.1
        # start confs
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        # result dict
        self.experiment_result = []

    def push_step_info_into_single_cube_passing_data(self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def plan_single_arm(self, planner, start_conf, goal_conf, description, active_id, command, static_arm_conf, cubes_real,
                            gripper_pre, gripper_post):
        path, cost = planner.find_path(start_conf=start_conf,
                                       goal_conf=goal_conf,
                                       manipulator=active_id)
        # create the arm plan
        self.push_step_info_into_single_cube_passing_data(description,
                                                          active_id,
                                                          command,
                                                          static_arm_conf.tolist(),
                                                          [path_element.tolist() for path_element in path],
                                                          [list(cube) for cube in cubes_real],
                                                          gripper_pre,
                                                          gripper_post)

    def plan_single_cube_passing(self, cube_i, cubes,
                                 left_arm_start, right_arm_start,
                                 env, bb, planner, left_arm_transform, right_arm_transform,):
        # add a new step entry
        single_cube_passing_info = {
            "description": [],  # text to be displayed on the animation
            "active_id": [],  # active arm id
            "command": [],
            "static": [],  # static arm conf
            "path": [],  # active arm path
            "cubes": [],  # coordinates of cubes on the board at the given timestep
            "gripper_pre": [],  # active arm pre path gripper action (OPEN/CLOSE/STAY)
            "gripper_post": []  # active arm pre path gripper action (OPEN/CLOSE/STAY)
        }
        self.experiment_result.append(single_cube_passing_info)
        ###############################################################################
        # set variables
        description = "right_arm => [start -> cube pickup], left_arm static"
        active_arm = LocationType.RIGHT
        # start planning
        log(msg=description)
    
      
        #################################################################################
        #                                                                               #
        #   #######  #######      ######   #######      #######                         #
        #      #     #     #      #     #  #     #     #       #                        #
        #      #     #     #      #     #  #     #             #                        #
        #      #     #     #      #     #  #     #       ######                         #
        #      #     #     #      #     #  #     #      #                               #
        #      #     #     #      #     #  #     #      #                               #
        #      #     #######      ######   #######      ########                        #
        #                                                                               #
        #################################################################################
        cube_approach = None #TODO 2: find a conf for the arm to get the correct cube

        # plan the path
        self.plan_single_arm(planner, right_arm_start, cube_approach, description, active_arm, "move",
                                 left_arm_start, cubes, Gripper.OPEN, Gripper.STAY)
        ###############################################################################

        self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [0, 0, -0.14],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.CLOSE)
        #################################################################################
        #                                                                               #
        #   #######  #######      ######   #######      #######                         #
        #      #     #     #      #     #  #     #     #       #                        #
        #      #     #     #      #     #  #     #             #                        #
        #      #     #     #      #     #  #     #       ######                         #
        #      #     #     #      #     #  #     #             #                        #
        #      #     #     #      #     #  #     #     #       #                        #
        #      #     #######      ######   #######      #######                         #
        #                                                                               #
        #################################################################################
        return None, None #TODO 3: return left and right end position, so it can be the start position for the next interation.


    def plan_experiment(self):
        start_time = time.time()

        exp_id = 2
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT])
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT])

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        bb = BuildingBlocks3D(env=env,
                             resolution=self.resolution,
                             p_bias=self.goal_bias, )

        rrt_star_planner = RRT_STAR(max_step_size=self.max_step_size,
                                    max_itr=self.max_itr,
                                    bb=bb)
        visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                  transform_left_arm=transform_left_arm)
        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id, env)

        log(msg="calculate meeting point for the test.")
        ################################################################################
        #                                                                               #
        #   #######  #######      ######   #######        #                             #
        #      #     #     #      #     #  #     #       ##                             #
        #      #     #     #      #     #  #     #      # #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #     #      #     #  #     #        #                             #
        #      #     #######      ######   #######      #####                           #
        #                                                                               #
        #################################################################################
        self.right_arm_meeting_cond = None # TODO 1
        self.left_arm_meeting_conf = None # TODO 1

        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(i, self.cubes, left_arm_start, right_arm_start,env, bb, rrt_star_planner, left_arm_start, right_arm_start)


        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")
        # save the experiment to data:
        # Serializing json
        json_object = json.dumps(self.experiment_result, indent=4)
        # Writing to sample.json
        dir_path = r"./outputs/"
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)
        # show the experiment then export it to a GIF
        visualizer.show_all_experiment(dir_path + "plan.json")
        visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id, env):
        """
        Generates a list of initial cube positions for a specific experiment scenario.

        This method defines a 0.4m x 0.4m workspace grid and places cubes at specific
        coordinates based on the provided experiment ID. The coordinates are in world frame.

        Args:
            experiment_id (int): The identifier for the experiment scenario.
                - 1: A single cube scenario.
                - 2: A two-cube scenario.
            env (Environment): The environment object containing base offsets.

        Returns:
            list: A list of lists, where each inner list contains the [x, y, z] 
                  coordinates of a cube. The z-coordinate is set to half the 
                  cube's side length (0.02m) to place it on the surface.
        """
        cube_side = 0.04
        cubes = []
        offset = env.cube_area_corner[LocationType.RIGHT]
        if experiment_id == 1:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            pos = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos + offset).tolist())
        elif experiment_id == 2:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            pos1 = np.array([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            cubes.append((pos1 + offset).tolist())
            # row 1: cube 2
            pos2 = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos2 + offset).tolist())
        return cubes
