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
        self.right_arm_meeting_confs = []
        self.left_arm_meeting_confs = []

    def push_step_info_into_single_cube_passing_data(self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

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

    def plan_single_arm(self, planner, start_conf, goal_confs, description, active_id, command, static_arm_conf, cubes_real,
                            gripper_pre, gripper_post, env, ur_params, transform):
        log(msg=description)
        update_environment(env, active_id, static_arm_conf, cubes_real)
        # Update bb with the correct transform and ur_params for the active arm
        planner.bb.transform = transform
        planner.bb.ur_params = ur_params
        path, cost, goal_conf = planner.find_path(start_conf=start_conf,
                                       goal_confs=goal_confs
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

        return goal_conf

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

        # TODO 2: find a conf for the arm to get the correct cube
        cube_coords = cubes[cube_i]

        right_arm = env.arm_base_location[LocationType.RIGHT]
        tool_len = inverse_kinematics.tool_length

        LIFT_HEIGHT = 0.4  # TODO: find correct Z value using simulations
        pickup_coords = (np.array(cube_coords) + np.array([0, 0, LIFT_HEIGHT])).tolist()

        # RPY for pickup: Account for right arm base rotation of -90° around Z
        # [roll, pitch, yaw] where yaw compensates for base rotation
        # TODO: return to [0, -np.pi/2, -np.pi/2] and debug why couldnt find confs for that
        pickup_rpy = [0, np.pi, np.pi] # [0, -np.pi/2, -np.pi/2]  # Adjusted for base rotation # TODO: find correct orientation using simulations

        transformation_matrix_base_to_tool = right_arm_transform.get_base_to_tool_transform(
            position=pickup_coords,
            rpy=pickup_rpy)

        # TODO: passing correct ur params????
        cube_approaches = bb.validate_IK_solutions(inverse_kinematics.inverse_kinematic_solution(
            inverse_kinematics.DH_matrix_UR5e, transformation_matrix_base_to_tool),transformation_matrix_base_to_tool)

        # plan the path
        print("cube approach confs: ", cube_approaches)
        cube_conf = self.plan_single_arm(planner, right_arm_start, cube_approaches, description, active_arm, "move",
                                 left_arm_start, cubes, Gripper.OPEN, Gripper.STAY, env, ur_params_right, right_arm_transform)  # gripper_pre: open before path, gripper_post: stay open after path
        ###############################################################################

        # After moving to cube_approach, the gripper goes down and closes
        # Calculate where the cube will be after picking up (at gripper position after going down)
        # The movel command moves down 0.14, so cube will be at that new position
        cube_after_pickup_pos = self.get_end_effector_position(cube_conf, right_arm_transform)
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


        # TODO 3
        print("finished part 2, starting part 3")

        # move right arm to meeting point (cube is now attached to right gripper)
        cubes_for_collision = [c for j, c in enumerate(cubes_after_pickup) if j != cube_i]
        description = "right_arm => [cube pickup -> meeting point], left_arm static"
        # TODO: start conf correct? cube conf or maybe it is updated after the movel they provided?
        right_meeting_point_conf = self.plan_single_arm(planner, cube_conf, self.right_arm_meeting_confs, description,
                                                        active_arm, "move", left_arm_start, cubes_for_collision,
                                                        Gripper.STAY, Gripper.STAY,
                                                        env, ur_params_right, right_arm_transform)  # gripper_pre: stay closed, gripper_post: stay closed (holding cube)

        # TODO: moved these lines from before to after the planning step. maybe wrong.
        # Cube moves with right gripper to meeting point
        cube_at_meeting_pos = self.get_end_effector_position(right_meeting_point_conf, right_arm_transform)
        cubes_at_meeting = self.update_cube_position(cubes, cube_i, cube_at_meeting_pos)

        # move left arm to meeting point
        description = "left_arm => [home -> meeting point], right_arm static"
        active_arm = LocationType.LEFT
        left_meeting_point_conf = self.plan_single_arm(planner, left_arm_start, self.left_arm_meeting_confs, description, active_arm, "move",
                             right_meeting_point_conf, cubes_at_meeting, Gripper.STAY, Gripper.STAY, env, ur_params_left, left_arm_transform)  # gripper_pre: stay open, gripper_post: stay open

        # Transfer cube from right arm to left arm (cube stays at same position during transfer)
        self.push_step_info_into_single_cube_passing_data("transferring cube: left closes to grab",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(right_meeting_point_conf),
                                                          [0, 0, 0],  # No movement, just gripper action
                                                          cubes_at_meeting,  # Cube still at meeting point
                                                          Gripper.STAY,  # gripper_pre: stay open
                                                          Gripper.CLOSE)  # gripper_post: close to grab cube

        # move left arm to B (cube now attached to left gripper)
        description = "left_arm => [meeting point -> place down], right_arm static"
        
        # Calculate Zone B position - try multiple candidate positions
        zone_b_offset = env.cube_area_corner[LocationType.LEFT]
        zone_b_size = 0.4  # 40cm x 40cm area
        cube_side = 0.04
        
        # Debug: Print Zone B boundaries
        print(f"\n=== ZONE B DEBUG INFO ===")
        print(f"Zone B corner (bottom-left): {zone_b_offset}")
        print(f"Zone B X range: [{zone_b_offset[0]:.3f}, {zone_b_offset[0] + zone_b_size:.3f}]")
        print(f"Zone B Y range: [{zone_b_offset[1]:.3f}, {zone_b_offset[1] + zone_b_size:.3f}]")
        print(f"=========================\n")
        
        # Try multiple candidate positions in Zone B
        # Left arm is at approximately [0.712, 1.516, 0] based on debug output
        # Zone B Y range is [1.588, 1.988], so we want Y closer to 1.588 (bottom of Zone B)
        candidate_positions = [
            # Bottom-right of Zone B (closest to left arm)
            [zone_b_offset[0] + zone_b_size * 0.75, zone_b_offset[1] + zone_b_size * 0.1, cube_side / 2.0],
            # Bottom-center
            [zone_b_offset[0] + zone_b_size * 0.5, zone_b_offset[1] + zone_b_size * 0.1, cube_side / 2.0],
            # Middle-right
            [zone_b_offset[0] + zone_b_size * 0.75, zone_b_offset[1] + zone_b_size * 0.3, cube_side / 2.0],
            # Middle-center
            [zone_b_offset[0] + zone_b_size * 0.5, zone_b_offset[1] + zone_b_size * 0.3, cube_side / 2.0],
            # Center
            [zone_b_offset[0] + zone_b_size * 0.5, zone_b_offset[1] + zone_b_size * 0.5, cube_side / 2.0],
        ]
        
        # Add lift height for approach (same as pickup)
        PLACEMENT_LIFT_HEIGHT = 0.2
        # RPY for placement: Account for left arm base rotation of +90° around Z
        placement_rpy = [0, -np.pi/2, np.pi/2]  # Adjusted for base rotation # TODO Find correct orientation using simulations
        
        bb_placement = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                       ur_params=ur_params_left, transform=left_arm_transform)
        update_environment(env, LocationType.LEFT, right_meeting_point_conf, cubes_at_meeting)
        
        valid_placement_confs = []
        left_arm_end_confs = []
        
        # Try each candidate position
        for i, candidate in enumerate(candidate_positions):
            zone_b_coords = candidate
            placement_coords = (np.array(zone_b_coords) + np.array([0, 0, PLACEMENT_LIFT_HEIGHT])).tolist()
            
            print(f"Trying Zone B position {i+1}/{len(candidate_positions)}: {placement_coords}")
            
            # Get transformation matrix for this candidate
            transformation_matrix_placement = left_arm_transform.get_base_to_tool_transform(
                position=placement_coords,
                rpy=placement_rpy)
            
            # Calculate IK solutions
            possible_placement_confs = inverse_kinematics.inverse_kinematic_solution(
                inverse_kinematics.DH_matrix_UR5e,
                transformation_matrix_placement)

            # for conf in possible_placement_confs:
            #     visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=right_arm_transform,
            #                               transform_left_arm=left_arm_transform)
            #     visualizer.draw_two_robots(conf_left=conf,
            #                                conf_right=right_meeting_point_conf)
            #
            # Try to find valid configurations (returns list of valid configs)
            try:
                valid_placement_confs =  bb_placement.validate_IK_solutions(
                    possible_placement_confs, transformation_matrix_placement)
            except ValueError as e:
                continue
            
            if len(valid_placement_confs) > 0:
                print(f"✓ Successfully found {len(valid_placement_confs)} valid placement(s) at position {i+1}")
                print(f"  Zone B placement coords: {zone_b_coords}")
                print(f"  Zone B approach coords (with lift): {placement_coords}")
                left_arm_end_confs = valid_placement_confs
                break
        
        if len(valid_placement_confs) == 0:
            raise ValueError(f"No valid configuration found for any Zone B placement position. Tried {len(candidate_positions)} positions.")
        
        # IMPORTANT: Cube is still at meeting point during planning!
        # Plan with cube at current (meeting) position, not final position
        left_arm_end_conf = self.plan_single_arm(planner, left_meeting_point_conf, left_arm_end_confs, description, active_arm, "move",
                             right_meeting_point_conf, cubes_at_meeting, Gripper.STAY, Gripper.STAY, env, ur_params_left, left_arm_transform)  # gripper_pre: stay closed, gripper_post: stay closed (holding cube)

        # AFTER planning, update cube position to placement position for the movel step
        cube_at_placement_pos = self.get_end_effector_position(left_arm_end_conf, left_arm_transform)
        cubes_at_placement = self.update_cube_position(cubes, cube_i, cube_at_placement_pos)

        # self.plan_single_arm(planner, self.left_arm_meeting_conf, left_arm_end_conf, description, active_arm, "move",
        #                      right_meeting_point_conf, cubes_at_placement, Gripper.STAY, Gripper.STAY, env)  # gripper_pre: stay closed, gripper_post: stay closed (holding cube)

        # Place down cube at B (cube goes down with gripper then stays there)
        cube_final_pos = list(cube_at_placement_pos)
        cube_final_pos[2] -= 0.14  # Cube goes down with gripper
        cubes_final = self.update_cube_position(cubes, cube_i, cube_final_pos)

        self.push_step_info_into_single_cube_passing_data("placing down cube: go down and open gripper",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(right_meeting_point_conf),
                                                          [0, 0, -0.14],
                                                          cubes_final,  # Cube at final position
                                                          Gripper.STAY,  # gripper_pre: stay closed
                                                          Gripper.OPEN)   # gripper_post: open to release cube

        return left_arm_end_conf, right_meeting_point_conf # return left and right end position, so it can be the start position for the next interation.


    def plan_experiment(self, DEMO=False):
        start_time = time.time()

        exp_id = 1
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)
        right_arm_rotation = [0, 0, -np.pi/2] 
        left_arm_rotation = [0, 0, np.pi/2]
        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],ur_rotation=right_arm_rotation)
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],ur_rotation=left_arm_rotation)

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        # TODO: buildings blocks left and right??
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
        # TODO 1
        ###############################START#############################################

        left_arm = env.arm_base_location[LocationType.LEFT]
        right_arm = env.arm_base_location[LocationType.RIGHT]
        tool_len = inverse_kinematics.tool_length
        right_x_bias = 0.5
        right_y_bias = 0.6

        print("left arm start = ", left_arm, " right arm start = ", right_arm)

        base_meeting_coords = [((1-right_x_bias)*left_arm[0] + right_x_bias*right_arm[0]),
                               ((1-right_y_bias)*left_arm[1] + right_y_bias*right_arm[1]),
                               0.5]  # Increased Z from 0.35 to 0.5 to avoid collisions
        left_meeting_coords = (np.array(base_meeting_coords) + np.array([-tool_len/2, tool_len/2, 0])).tolist() #check y should be 0 ?
        right_meeting_coords = (np.array(base_meeting_coords) + np.array([tool_len/2, -tool_len/2, 0])).tolist()

        print(f"DEBUG: Meeting point coords - base: {base_meeting_coords}, left: {left_meeting_coords}, right: {right_meeting_coords}")

        left_meeting_rpy = [np.pi/2, 0, np.pi*1/4]  # TODO Validate via simulation
        right_meeting_rpy = [np.pi/2, np.pi, -np.pi*3/4]

        transformation_matrix_base_to_tool_l = transform_left_arm.get_base_to_tool_transform(position=left_meeting_coords,
                                                                                            rpy=left_meeting_rpy)
        transformation_matrix_base_to_tool_r = transform_right_arm.get_base_to_tool_transform(position=right_meeting_coords,
                                                                                            rpy=right_meeting_rpy)

        # Select best valid IK solutions
        print("Selecting best left arm meeting point configuration.")
        print("transform to r: ", transformation_matrix_base_to_tool_r)
        ik_r = inverse_kinematics.inverse_kinematic_solution(
            inverse_kinematics.DH_matrix_UR5e,transformation_matrix_base_to_tool_r)
        print("IK r: ", ik_r)

        bb_left = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                   ur_params=ur_params_left, transform=transform_left_arm)
        update_environment(env, LocationType.LEFT, self.right_arm_home, [])  # Set environment for left arm
        self.left_arm_meeting_confs = bb_left.validate_IK_solutions(inverse_kinematics.inverse_kinematic_solution(
            inverse_kinematics.DH_matrix_UR5e,transformation_matrix_base_to_tool_l), transformation_matrix_base_to_tool_l)

        bb_right = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                    ur_params=ur_params_right, transform=transform_right_arm)
        update_environment(env, LocationType.RIGHT, self.left_arm_home, [])  # Set environment for right arm
        self.right_arm_meeting_confs = bb_right.validate_IK_solutions(inverse_kinematics.inverse_kinematic_solution(
            inverse_kinematics.DH_matrix_UR5e, transformation_matrix_base_to_tool_r), transformation_matrix_base_to_tool_r)  # No start_conf = no edge checking(?)

        print("right confs: ", self.right_arm_meeting_confs)


        if len(self.left_arm_meeting_confs) == 0:
            raise ValueError("No valid left arm meeting point configuration found!")
        if len(self.right_arm_meeting_confs) == 0:
            raise ValueError("No valid right arm meeting point configuration found!")

        print("left conf for meeting points: ", self.left_arm_meeting_confs)
        print("right conf for meeting points: ", self.right_arm_meeting_confs)

        if DEMO:
            for conf in self.right_arm_meeting_confs:
                visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                          transform_left_arm=transform_left_arm)
                visualizer.draw_two_robots(conf_left=self.left_arm_meeting_confs[0],
                                           conf_right=conf)
            return [self.left_arm_meeting_confs, self.right_arm_meeting_confs]

        #################################END#############################################


        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = self.plan_single_cube_passing(i, self.cubes, left_arm_start, right_arm_start, env, bb_right, rrt_star_planner, transform_left_arm, transform_right_arm, ur_params_left, ur_params_right)


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
