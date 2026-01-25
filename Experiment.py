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

    def select_best_valid_ik_solution(self, ik_solutions, bb, ur_params, start_conf=None):
        """
        Select the best valid IK solution from multiple solutions.
        
        Args:
            ik_solutions: Array of IK solutions (8x6)
            bb: BuildingBlocks3D object for collision checking
            ur_params: Robot parameters for joint limits
            start_conf: Optional start configuration to prefer solutions closer to it
            
        Returns:
            Best valid configuration, or None if no valid solution exists
        """

        # TODO: idea: alternative approach. take all valid solutions, and run the planner while sampling all of them.
        #  take the best (or first) solution reached this way.
        #  maybe once reaching a solution after X iterations, continue another O(X) iterations to try to find better solution.

        limits = list(ur_params.mechamical_limits.values())
        valid_solutions = []
        
        print(f"  Checking {len(ik_solutions)} IK solutions...")
        
        for i, conf in enumerate(ik_solutions):
            print(f"    Solution {i}: conf = {np.rad2deg(conf).round(2)}")
            
            # Check joint limits
            within_limits = all(limits[j][0] <= angle <= limits[j][1] 
                               for j, angle in enumerate(conf))
            if not within_limits:
                print(f"      REJECTED: Joint limits violated")
                violated_joints = [j for j, angle in enumerate(conf) 
                                  if not (limits[j][0] <= angle <= limits[j][1])]
                print(f"      Violated joints: {violated_joints}")
                continue
            
            # Check collision
            is_collision_free = bb.config_validity_checker(conf)
            if not is_collision_free:
                print(f"      REJECTED: Configuration collision")
                continue
            
            # Check edge validity from start if provided
            # TODO: this checks only from start to conf, may get false collisions while a path is still possible
            if start_conf is not None:
                edge_valid = bb.edge_validity_checker(start_conf, conf)
                if not edge_valid:
                    print(f"      REJECTED: Edge collision from start to goal")
                    continue
            
            # Calculate cost (distance from start if provided, otherwise just joint sum)
            if start_conf is not None:
                cost = bb.compute_distance(start_conf, conf)
            else:
                cost = np.sum(np.abs(conf))  # Prefer solutions with smaller joint angles
            
            print(f"      VALID: cost = {cost:.4f}")
            valid_solutions.append((i, conf, cost))
        
        if not valid_solutions:
            print(f"  NO valid solutions found out of {len(ik_solutions)} candidates")
            return None
        
        # Sort by cost and return the best one
        valid_solutions.sort(key=lambda x: x[2])
        best_idx, best_conf, best_cost = valid_solutions[0]
        
        print(f"  Selected solution {best_idx} out of {len(ik_solutions)} (cost: {best_cost:.4f})")
        print(f"  Found {len(valid_solutions)} valid solutions total")
        
        return best_conf

    def get_end_effector_position(self, conf, transform):
        """
        Get the end effector position for a given configuration.
        
        Args:
            conf: Joint configuration (6 values)
            transform: Transform object (left or right arm)
            
        Returns:
            [x, y, z] position of end effector in world coordinates
        """
        # Get transformation matrices for all links
        trans_matrix = transform.get_trans_matrix(conf)
        
        # The end effector is at the last link (wrist_3_link)
        # Get position in base frame
        end_effector_base = trans_matrix['wrist_3_link'][:3, 3]
        
        # Transform to world coordinates
        end_effector_homogeneous = np.array([end_effector_base[0], end_effector_base[1], end_effector_base[2], 1.0])
        end_effector_world = np.matmul(transform.base_transform, end_effector_homogeneous)
        
        return end_effector_world[:3].tolist()
    
    def update_cube_position(self, cubes, cube_i, new_position):
        """
        Update the position of a specific cube.
        
        Args:
            cubes: List of cube positions
            cube_i: Index of cube to update
            new_position: New [x, y, z] position
            
        Returns:
            Updated cubes list
        """
        updated_cubes = [list(cube) for cube in cubes]  # Deep copy
        updated_cubes[cube_i] = new_position
        return updated_cubes

    def plan_single_arm(self, planner, start_conf, goal_conf, description, active_id, command, static_arm_conf, cubes_real,
                            gripper_pre, gripper_post, env, ur_params, transform):
        log(msg=description)
        update_environment(env, active_id, static_arm_conf, cubes_real)
        # Update bb with the correct transform and ur_params for the active arm
        planner.bb.transform = transform
        planner.bb.ur_params = ur_params
        path, cost = planner.find_path(start_conf=start_conf,
                                       goal_conf=goal_conf
                                       )
                                       # ,manipulator=active_id)
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
                                 env, bb, planner, left_arm_transform, right_arm_transform, ur_params_left, ur_params_right):
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
        # TODO 2: find a conf for the arm to get the correct cube
        cube_coords = cubes[cube_i]

        right_arm = env.arm_base_location[LocationType.RIGHT]
        tool_len = inverse_kinematics.tool_length

        LIFT_HEIGHT = 0.2  # TODO: find correct Z value using simulations
        pickup_coords = (np.array(cube_coords) + np.array([0, 0, LIFT_HEIGHT])).tolist()

        pickup_rpy = [0, -np.pi/2, 0] # TODO: find correct orientation using simulations

        transformation_matrix_base_to_tool = right_arm_transform.get_base_to_tool_transform(
            position=pickup_coords,
            rpy=pickup_rpy)

        possible_cube_approach = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e,
                                                                                    transformation_matrix_base_to_tool)

        # Select best valid cube approach configuration
        print("Selecting best cube approach configuration...")
        cube_approach = self.select_best_valid_ik_solution(
            possible_cube_approach, bb, env.ur_params, right_arm_start)
        
        if cube_approach is None:
            raise ValueError(f"No valid configuration found for cube {cube_i} at position {cube_coords}")
        
        # plan the path
        print("cube approach conf: ", cube_approach)
        self.plan_single_arm(planner, right_arm_start, cube_approach, description, active_arm, "move",
                                 left_arm_start, cubes, Gripper.OPEN, Gripper.STAY, env, ur_params_right, right_arm_transform)  # gripper_pre: open before path, gripper_post: stay open after path
        ###############################################################################

        # After moving to cube_approach, the gripper goes down and closes
        # Calculate where the cube will be after picking up (at gripper position after going down)
        # The movel command moves down 0.14, so cube will be at that new position
        cube_after_pickup_pos = self.get_end_effector_position(cube_approach, right_arm_transform)
        cube_after_pickup_pos[2] -= 0.14  # Account for the down movement
        cubes_after_pickup = self.update_cube_position(cubes, cube_i, cube_after_pickup_pos)
        
        self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [0, 0, -0.14],
                                                          cubes_after_pickup,  # Cube is now at gripper position
                                                          Gripper.STAY,  # gripper_pre: already open, stay open
                                                          Gripper.CLOSE)  # gripper_post: close after reaching cube
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
        # TODO 3
        print("finished part 2, starting part 3")

        # TODO: gripper states
        # move right arm to meeting point (cube is now attached to right gripper)
        description = "right_arm => [cube pickup -> meeting point], left_arm static"
        
        # IMPORTANT: Cube is still at the pickup position during planning!
        # It moves WITH the robot, so we plan with cube at current (pickup) position
        self.plan_single_arm(planner, cube_approach, self.right_arm_meeting_conf, description, active_arm, "move",
                             left_arm_start, cubes_after_pickup, Gripper.STAY, Gripper.STAY, env, ur_params_right, right_arm_transform)  # gripper_pre: stay closed, gripper_post: stay closed (holding cube)

        # AFTER planning, update cube position to meeting point for next step
        cube_at_meeting_pos = self.get_end_effector_position(self.right_arm_meeting_conf, right_arm_transform)
        cubes_at_meeting = self.update_cube_position(cubes, cube_i, cube_at_meeting_pos)

        # move left arm to meeting point
        description = "left_arm => [home -> meeting point], right_arm static"
        active_arm = LocationType.LEFT
        self.plan_single_arm(planner, left_arm_start, self.left_arm_meeting_conf, description, active_arm, "move",
                             self.right_arm_meeting_conf, cubes_at_meeting, Gripper.STAY, Gripper.STAY, env, ur_params_left, left_arm_transform)  # gripper_pre: stay open, gripper_post: stay open

        # Transfer cube from right arm to left arm (cube stays at same position during transfer)
        self.push_step_info_into_single_cube_passing_data("transferring cube: left closes to grab",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(self.right_arm_meeting_conf),
                                                          [0, 0, 0],  # No movement, just gripper action
                                                          cubes_at_meeting,  # Cube still at meeting point
                                                          Gripper.STAY,  # gripper_pre: stay open
                                                          Gripper.CLOSE)  # gripper_post: close to grab cube

        # move left arm to B (cube now attached to left gripper)
        description = "left_arm => [meeting point -> place down], right_arm static"
        left_arm_end_conf = self.right_arm_home # TODO: find conf for placing the cube at B
        
        # IMPORTANT: Cube is still at meeting point during planning!
        # Plan with cube at current (meeting) position, not final position
        self.plan_single_arm(planner, self.left_arm_meeting_conf, left_arm_end_conf, description, active_arm, "move",
                             self.right_arm_meeting_conf, cubes_at_meeting, Gripper.STAY, Gripper.STAY, env, ur_params_left, left_arm_transform)  # gripper_pre: stay closed, gripper_post: stay closed (holding cube)

        # AFTER planning, update cube position to placement position for the movel step
        cube_at_placement_pos = self.get_end_effector_position(left_arm_end_conf, left_arm_transform)
        cubes_at_placement = self.update_cube_position(cubes, cube_i, cube_at_placement_pos)

        # Place down cube at B (cube goes down with gripper then stays there)
        cube_final_pos = list(cube_at_placement_pos)
        cube_final_pos[2] -= 0.14  # Cube goes down with gripper
        cubes_final = self.update_cube_position(cubes, cube_i, cube_final_pos)
        
        self.push_step_info_into_single_cube_passing_data("placing down cube: go down and open gripper",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(self.right_arm_meeting_conf),
                                                          [0, 0, -0.14],
                                                          cubes_final,  # Cube at final position
                                                          Gripper.STAY,  # gripper_pre: stay closed
                                                          Gripper.OPEN)   # gripper_post: open to release cube

        return left_arm_end_conf, self.right_arm_meeting_conf # return left and right end position, so it can be the start position for the next interation.


    def plan_experiment(self):
        start_time = time.time()

        exp_id = 1
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT])
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT])

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm


        bb = BuildingBlocks3D(env=env,
                             resolution=self.resolution,
                             p_bias=self.goal_bias,
                              # TODO: check if ur params and transform are right choice
                              ur_params=ur_params_right, transform=transform_right_arm)

        rrt_star_planner = RRT_STAR(max_step_size=self.max_step_size,
                                    max_itr=self.max_itr,
                                    bb=bb)
        visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                  transform_left_arm=transform_left_arm)
        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id, env)
            print("cubes for the experiment: ", self.cubes)

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
        # TODO 1
        ###############################START#############################################

        left_arm = env.arm_base_location[LocationType.LEFT]
        right_arm = env.arm_base_location[LocationType.RIGHT]
        tool_len = inverse_kinematics.tool_length
        right_x_bias = 0.5
        right_y_bias = 0.5

        base_meeting_coords = [((1-right_x_bias)*left_arm[0] + right_x_bias*right_arm[0]),
                               ((1-right_y_bias)*left_arm[1] + right_y_bias*right_arm[1]),
                               0.5]  # Increased Z from 0.35 to 0.5 to avoid collisions
        left_meeting_coords = (np.array(base_meeting_coords) + np.array([tool_len/2, -tool_len/2, 0])).tolist()
        right_meeting_coords = (np.array(base_meeting_coords) - np.array([tool_len/2, -tool_len/2, 0])).tolist()

        print(f"DEBUG: Meeting point coords - base: {base_meeting_coords}, left: {left_meeting_coords}, right: {right_meeting_coords}")

        left_meeting_rpy = [0, -np.pi/2, np.pi*3/4]  # temporary
        right_meeting_rpy = [np.pi, -np.pi/2, np.pi*3/4]  # temporary
        print(f"DEBUG: Meeting point RPY - left: {np.rad2deg(left_meeting_rpy)}, right: {np.rad2deg(right_meeting_rpy)}")

        transformation_matrix_base_to_tool_l = transform_left_arm.get_base_to_tool_transform(position=left_meeting_coords,
                                                                                            rpy=left_meeting_rpy)
        transformation_matrix_base_to_tool_r = transform_right_arm.get_base_to_tool_transform(position=right_meeting_coords,
                                                                                            rpy=right_meeting_rpy)
        left_arm_meeting_confs= inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e,transformation_matrix_base_to_tool_l)
        right_arm_meeting_confs= inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e,transformation_matrix_base_to_tool_r)

        # Select best valid IK solutions
        print("Selecting best left arm meeting point configuration.")
        bb_left = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                   ur_params=ur_params_left, transform=transform_left_arm)
        update_environment(env, LocationType.LEFT, self.right_arm_home, [])  # Set environment for left arm
        self.left_arm_meeting_conf = self.select_best_valid_ik_solution(
            left_arm_meeting_confs, bb_left, ur_params_left, None)  # No start_conf = no edge checking
        
        print("Selecting best right arm meeting point configuration (no collision check with other arm)...")
        bb_right = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                    ur_params=ur_params_right, transform=transform_right_arm)
        update_environment(env, LocationType.RIGHT, self.left_arm_home, [])  # Set environment for right arm
        self.right_arm_meeting_conf = self.select_best_valid_ik_solution(
            right_arm_meeting_confs, bb_right, ur_params_right, None)  # No start_conf = no edge checking
        
        if self.left_arm_meeting_conf is None:
            raise ValueError("No valid left arm meeting point configuration found!")
        if self.right_arm_meeting_conf is None:
            raise ValueError("No valid right arm meeting point configuration found!")
        
        print("left conf for meeting point: ", self.left_arm_meeting_conf)
        print("right conf for meeting point: ", self.right_arm_meeting_conf)

        #################################END#############################################


        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(i, self.cubes, left_arm_start, right_arm_start,env, bb, rrt_star_planner, transform_left_arm, transform_right_arm, ur_params_left, ur_params_right)


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
