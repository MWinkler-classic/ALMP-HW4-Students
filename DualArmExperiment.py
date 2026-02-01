"""
Dual Arm Experiment - Parallel Execution using Threads

This module uses the existing Experiment.py planning logic but executes 
both arms simultaneously using threading for improved execution time.

Key approach:
1. Use existing RRT_STAR planner to plan paths for each arm
2. Plan both arms in parallel using threads during Phase 1
3. Execute paths simultaneously when arms don't conflict
"""

import json
import time
import threading
import numpy as np

from environment import Environment, LocationType
from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from building_blocks import BuildingBlocks3D
from planners import RRT_STAR
from visualizer import Visualize_UR
import inverse_kinematics


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    with open(dir_path + 'output_dual.txt', 'a') as file:
        file.write(f"{written_log}\n")


class DualArmExperiment:
    """
    Dual-arm experiment that plans using existing methods and executes with parallel threads.
    """
    
    def __init__(self, cubes=None):
        self.cubes = cubes
        
        # Tunable params (same as Experiment.py)
        self.max_itr = 1000
        self.max_step_size = 0.5
        self.goal_bias = 0.05
        self.resolution = 0.1
        
        # Start configurations
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        
        # Meeting point configurations (will be computed)
        self.right_arm_meeting_confs = []
        self.left_arm_meeting_confs = []
        
        # Result data - compatible with visualizer
        self.experiment_result = []
        
    def get_end_effector_position(self, conf, transform):
        """Get the end effector position for a given configuration."""
        trans_matrix = transform.get_trans_matrix(conf)
        end_effector_base = trans_matrix['wrist_3_link'][:3, 3]
        end_effector_homogeneous = np.array([end_effector_base[0], end_effector_base[1], end_effector_base[2], 1.0])
        end_effector_world = np.matmul(transform.base_transform, end_effector_homogeneous)
        return end_effector_world[:3].tolist()
    
    def update_cube_position(self, cubes, cube_i, new_position):
        """Update the position of a specific cube."""
        updated_cubes = [list(cube) for cube in cubes]
        updated_cubes[cube_i] = new_position
        return updated_cubes

    def push_step_info(self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post):
        """Add step info (compatible with existing visualizer)."""
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def plan_parallel_phase(self, env, bb_right, bb_left,
                            right_start, right_goals, left_start, left_goals,
                            cubes, ur_params_right, ur_params_left, 
                            transform_right, transform_left, description):
        """
        Plan both arms sequentially but execute them simultaneously.
        Planning is sequential to avoid race conditions, but the paths
        will be executed in parallel (synchronized).
        
        IMPORTANT: After planning, we validate that the parallel paths don't collide!
        """
        log(msg=description)
        
        t_start = time.time()
        max_retries = 3
        
        for retry in range(max_retries):
            if retry > 0:
                print(f"  Retry {retry}/{max_retries} - replanning to avoid collision...")
            
            # Plan RIGHT arm first (with left at its start position)
            print("  Planning right arm...")
            env.set_active_arm(LocationType.RIGHT)
            env.update_obstacles(cubes, left_start)
            bb_right.transform = transform_right
            bb_right.ur_params = ur_params_right
            
            planner_right = RRT_STAR(max_step_size=self.max_step_size, max_itr=self.max_itr, bb=bb_right, stop_on_goal=True)
            right_path, _, right_goal = planner_right.find_path(start_conf=right_start, goal_confs=right_goals)
            
            if len(right_path) == 0:
                raise ValueError(f"Failed to find right arm path: {description}")
            print(f"  Right arm path: {len(right_path)} waypoints")
            
            # Plan LEFT arm (with right at its start position - they move simultaneously)
            print("  Planning left arm...")
            env.set_active_arm(LocationType.LEFT)
            env.update_obstacles(cubes, right_start)
            bb_left.transform = transform_left
            bb_left.ur_params = ur_params_left
            
            planner_left = RRT_STAR(max_step_size=self.max_step_size, max_itr=self.max_itr, bb=bb_left, stop_on_goal=True)
            left_path, _, left_goal = planner_left.find_path(start_conf=left_start, goal_confs=left_goals)
            
            if len(left_path) == 0:
                raise ValueError(f"Failed to find left arm path: {description}")
            print(f"  Left arm path: {len(left_path)} waypoints")
            
            # VALIDATE: Check for arm-to-arm collisions during parallel execution
            is_safe, collision_idx = self.validate_parallel_paths(
                right_path, left_path,
                transform_right, transform_left,
                ur_params_right, ur_params_left
            )
            
            if is_safe:
                break
            elif retry == max_retries - 1:
                print(f"  WARNING: Could not find collision-free parallel paths after {max_retries} retries!")
                print(f"  Continuing with potentially unsafe paths - manual review recommended.")
        
        t_end = time.time()
        print(f"  Both arms planned in {t_end - t_start:.2f}s")
        
        return right_path, right_goal, left_path, left_goal

    def synchronize_paths(self, right_path, left_path):
        """Synchronize two paths to have the same length."""
        right_path = [np.array(p) for p in right_path]
        left_path = [np.array(p) for p in left_path]
        
        n_right = len(right_path)
        n_left = len(left_path)
        
        if n_right == n_left:
            return right_path, left_path
        
        if n_right > n_left:
            left_path.extend([left_path[-1]] * (n_right - n_left))
        else:
            right_path.extend([right_path[-1]] * (n_left - n_right))
        
        return right_path, left_path

    def check_arm_to_arm_collision(self, right_conf, left_conf, transform_right, transform_left,
                                    ur_params_right, ur_params_left):
        """
        Check if two arm configurations collide with each other.
        Returns True if collision detected, False if safe.
        """
        # Get sphere coordinates for both arms
        right_spheres = transform_right.conf2sphere_coords(right_conf)
        left_spheres = transform_left.conf2sphere_coords(left_conf)
        
        right_radii = ur_params_right.sphere_radius
        left_radii = ur_params_left.sphere_radius
        
        # Check collision between all links of both arms
        for right_link in ur_params_right.ur_links:
            right_link_spheres = self._convert_to_3d_spheres(right_spheres[right_link])
            r_right = right_radii[right_link]
            
            for left_link in ur_params_left.ur_links:
                left_link_spheres = self._convert_to_3d_spheres(left_spheres[left_link])
                r_left = left_radii[left_link]
                
                # Calculate pairwise distances
                dists = np.linalg.norm(
                    right_link_spheres[:, None, :] - left_link_spheres[None, :, :],
                    axis=2
                )
                
                # Check if any spheres are closer than their combined radii
                if np.any(dists < (r_right + r_left)):
                    return True  # Collision detected
        
        return False  # No collision

    def _convert_to_3d_spheres(self, link_spheres):
        """Convert sphere coords to 3D numpy array."""
        new_spheres = []
        for sphere in link_spheres:
            sphere = np.array(sphere, dtype=np.float64).reshape(-1)
            new_spheres.append(sphere[:3])
        return np.vstack(new_spheres)

    def validate_parallel_paths(self, right_path, left_path, transform_right, transform_left,
                                 ur_params_right, ur_params_left, resolution=0.1):
        """
        Validate that two synchronized paths don't collide when executed simultaneously.
        Returns True if paths are safe, False if collision detected.
        """
        right_sync, left_sync = self.synchronize_paths(right_path, left_path)
        
        print(f"  Validating parallel paths ({len(right_sync)} waypoints)...")
        
        for i in range(len(right_sync)):
            # Check at each waypoint
            if self.check_arm_to_arm_collision(
                right_sync[i], left_sync[i],
                transform_right, transform_left,
                ur_params_right, ur_params_left
            ):
                print(f"  WARNING: Arm-to-arm collision detected at waypoint {i}!")
                return False, i
            
            # Also check intermediate points between waypoints
            if i < len(right_sync) - 1:
                progress = resolution
                while progress < 1.0:
                    right_interp = right_sync[i] * (1.0 - progress) + right_sync[i+1] * progress
                    left_interp = left_sync[i] * (1.0 - progress) + left_sync[i+1] * progress
                    
                    if self.check_arm_to_arm_collision(
                        right_interp, left_interp,
                        transform_right, transform_left,
                        ur_params_right, ur_params_left
                    ):
                        print(f"  WARNING: Arm-to-arm collision detected between waypoints {i} and {i+1}!")
                        return False, i
                    
                    progress += resolution
        
        print(f"  Parallel paths validated - NO collisions detected!")
        return True, -1

    def store_parallel_motion(self, description, right_path, left_path, cubes,
                              right_gripper_pre="STAY", right_gripper_post="STAY",
                              left_gripper_pre="STAY", left_gripper_post="STAY"):
        """Store parallel motion for visualization."""
        right_sync, left_sync = self.synchronize_paths(right_path, left_path)
        
        self.push_step_info(
            description=f"[PARALLEL-R] {description}",
            active_id=LocationType.RIGHT,
            command="move",
            static_conf=list(left_sync[0]),
            path=[p.tolist() for p in right_sync],
            cubes=[list(c) for c in cubes],
            gripper_pre=right_gripper_pre,
            gripper_post=right_gripper_post
        )
        
        self.push_step_info(
            description=f"[PARALLEL-L] {description}",
            active_id=LocationType.LEFT,
            command="move",
            static_conf=list(right_sync[-1]),
            path=[p.tolist() for p in left_sync],
            cubes=[list(c) for c in cubes],
            gripper_pre=left_gripper_pre,
            gripper_post=left_gripper_post
        )

    def plan_single_cube_passing(self, cube_i, cubes, left_arm_start, right_arm_start,
                                  env, bb_right, bb_left,
                                  transform_left, transform_right, 
                                  ur_params_left, ur_params_right):
        """Plan cube passing with parallel motion where possible."""
        
        single_cube_passing_info = {
            "description": [], "active_id": [], "command": [],
            "static": [], "path": [], "cubes": [],
            "gripper_pre": [], "gripper_post": []
        }
        self.experiment_result.append(single_cube_passing_info)
        
        cube_coords = cubes[cube_i]
        LIFT_HEIGHT = 0.1
        
        # =====================================================
        # PHASE 1: PARALLEL - Right to cube, Left to meeting
        # =====================================================
        description = "PARALLEL: Right->cube, Left->meeting"
        
        pickup_coords = (np.array(cube_coords) + np.array([0, 0, LIFT_HEIGHT])).tolist()
        pickup_rpy = [0, np.pi, np.pi]
        
        transformation_pickup = transform_right.get_base_to_tool_transform(
            position=pickup_coords, rpy=pickup_rpy)
        
        cubes_for_ik = [c for j, c in enumerate(cubes) if j != cube_i]
        env.set_active_arm(LocationType.RIGHT)
        env.update_obstacles(cubes_for_ik, left_arm_start)
        
        right_cube_approach_confs = bb_right.validate_IK_solutions(
            inverse_kinematics.inverse_kinematic_solution(
                inverse_kinematics.DH_matrix_UR5e, transformation_pickup),
            transformation_pickup)
        
        right_path, right_goal, left_path, left_goal = self.plan_parallel_phase(
            env, bb_right, bb_left,
            right_arm_start, right_cube_approach_confs,
            left_arm_start, self.left_arm_meeting_confs,
            cubes_for_ik, ur_params_right, ur_params_left,
            transform_right, transform_left, description
        )
        
        self.store_parallel_motion(description, right_path, left_path, cubes,
                                   right_gripper_pre="OPEN", left_gripper_pre="OPEN")
        
        cube_conf = right_goal
        left_at_meeting = left_goal
        
        # =====================================================
        # PHASE 2: Pickup cube (right arm only)
        # =====================================================
        CUBE_PICKUP_CONST = -0.10
        
        self.push_step_info("Right: pickup cube (go down)", LocationType.RIGHT, "movel",
                           list(left_at_meeting), [0, 0, CUBE_PICKUP_CONST], [], "STAY", "CLOSE")
        
        cube_after_pickup_pos = self.get_end_effector_position(cube_conf, transform_right)
        cube_after_pickup_pos[2] += CUBE_PICKUP_CONST
        cubes_after_pickup = self.update_cube_position(cubes, cube_i, cube_after_pickup_pos)
        
        self.push_step_info("Right: lift cube", LocationType.RIGHT, "movel",
                           list(left_at_meeting), [0, 0, -CUBE_PICKUP_CONST], cubes_after_pickup, "STAY", "STAY")
        
        # =====================================================
        # PHASE 3: Right arm to meeting point
        # =====================================================
        description = "Right->meeting (left already there)"
        log(msg=description)
        
        cubes_for_collision = [c for j, c in enumerate(cubes_after_pickup) if j != cube_i]
        env.set_active_arm(LocationType.RIGHT)
        env.update_obstacles(cubes_for_collision, left_at_meeting)
        bb_right.transform = transform_right
        bb_right.ur_params = ur_params_right
        
        planner_right = RRT_STAR(max_step_size=self.max_step_size, max_itr=self.max_itr, bb=bb_right, stop_on_goal=True)
        path, _, right_at_meeting = planner_right.find_path(start_conf=cube_conf, goal_confs=self.right_arm_meeting_confs)
        
        self.push_step_info(description, LocationType.RIGHT, "move", list(left_at_meeting),
                           [p.tolist() for p in path], cubes_for_collision, "STAY", "STAY")
        
        cube_at_meeting_pos = self.get_end_effector_position(right_at_meeting, transform_right)
        cubes_at_meeting = self.update_cube_position(cubes, cube_i, cube_at_meeting_pos)
        
        # =====================================================
        # PHASE 4: Cube transfer at meeting point
        # =====================================================
        moving_back = 0.11
        
        self.push_step_info("Transfer: left grabs cube", LocationType.LEFT, "movel",
                           list(right_at_meeting), [0, -moving_back, 0], cubes_at_meeting, "STAY", "CLOSE")
        
        self.push_step_info("Transfer: right releases", LocationType.RIGHT, "movel",
                           list(left_at_meeting), [0, 0, 0], cubes_at_meeting, "STAY", "OPEN")
        
        self.push_step_info("Transfer: left back up", LocationType.LEFT, "movel",
                           list(right_at_meeting), [0, moving_back, 0], cubes_at_meeting, "STAY", "STAY")
        
        # =====================================================
        # PHASE 5: Left arm to Zone B
        # =====================================================
        description = "Left->ZoneB (right stays at meeting)"
        log(msg=description)
        
        cubes_for_phase5 = [c for i, c in enumerate(cubes) if i != cube_i]
        
        zone_b_offset = env.cube_area_corner[LocationType.LEFT]
        zone_b_size = 0.4
        cube_side = 0.04
        PLACEMENT_LIFT_HEIGHT = 0.2
        placement_rpy = [0, np.pi, 0]
        
        candidate_positions = [
            [zone_b_offset[0] + zone_b_size * 0.5, zone_b_offset[1] + zone_b_size * 0.5, cube_side / 2.0],
            [zone_b_offset[0] + zone_b_size * 0.75, zone_b_offset[1] + zone_b_size * 0.1, cube_side / 2.0],
            [zone_b_offset[0] + zone_b_size * 0.5, zone_b_offset[1] + zone_b_size * 0.1, cube_side / 2.0],
        ]
        
        valid_placement_confs = []
        for candidate in candidate_positions:
            placement_coords = (np.array(candidate) + np.array([0, 0, PLACEMENT_LIFT_HEIGHT])).tolist()
            transformation_placement = transform_left.get_base_to_tool_transform(
                position=placement_coords, rpy=placement_rpy)
            
            possible_confs = inverse_kinematics.inverse_kinematic_solution(
                inverse_kinematics.DH_matrix_UR5e, transformation_placement)
            
            env.set_active_arm(LocationType.LEFT)
            env.update_obstacles(cubes_for_phase5, right_at_meeting)
            
            try:
                valid_confs = bb_left.validate_IK_solutions(possible_confs, transformation_placement)
                if len(valid_confs) > 0:
                    valid_placement_confs = valid_confs
                    break
            except ValueError:
                continue
        
        if not valid_placement_confs:
            raise ValueError("No valid Zone B placement found!")
        
        env.set_active_arm(LocationType.LEFT)
        env.update_obstacles(cubes_for_phase5, right_at_meeting)
        bb_left.transform = transform_left
        bb_left.ur_params = ur_params_left
        
        planner_left = RRT_STAR(max_step_size=self.max_step_size, max_itr=self.max_itr, bb=bb_left, stop_on_goal=True)
        left_path, _, left_at_placement = planner_left.find_path(start_conf=left_at_meeting, goal_confs=valid_placement_confs)
        
        self.push_step_info(description, LocationType.LEFT, "move", list(right_at_meeting),
                           [p.tolist() for p in left_path], cubes_for_phase5, "STAY", "STAY")
        
        # =====================================================
        # PHASE 6: Place cube down
        # =====================================================
        cube_at_placement_pos = self.get_end_effector_position(left_at_placement, transform_left)
        cube_final_pos = list(cube_at_placement_pos)
        cube_final_pos[2] = 0.02
        cubes_final = self.update_cube_position(cubes, cube_i, cube_final_pos)
        
        self.push_step_info("Left: place cube down", LocationType.LEFT, "movel",
                           list(right_at_meeting), [0, 0, 0], cubes_final, "STAY", "OPEN")
        
        self.cubes[cube_i] = cube_final_pos
        
        return left_at_placement, right_at_meeting

    def plan_experiment(self, DEMO=False):
        """Plan and execute the dual-arm experiment."""
        start_time = time.time()
        
        exp_id = 2
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)
        
        env = Environment(ur_params=ur_params_right)
        right_arm_rotation = [0, 0, -np.pi/2]
        left_arm_rotation = [0, 0, np.pi/2]
        
        transform_right = Transform(ur_params=ur_params_right,
                                    ur_location=env.arm_base_location[LocationType.RIGHT],
                                    ur_rotation=right_arm_rotation)
        transform_left = Transform(ur_params=ur_params_left,
                                   ur_location=env.arm_base_location[LocationType.LEFT],
                                   ur_rotation=left_arm_rotation)
        
        env.arm_transforms[LocationType.RIGHT] = transform_right
        env.arm_transforms[LocationType.LEFT] = transform_left
        
        bb_right = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                    ur_params=ur_params_right, transform=transform_right)
        bb_left = BuildingBlocks3D(env=env, resolution=self.resolution, p_bias=self.goal_bias,
                                   ur_params=ur_params_left, transform=transform_left)
        
        visualizer = Visualize_UR(ur_params_right, env=env,
                                  transform_right_arm=transform_right,
                                  transform_left_arm=transform_left)
        
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id, env)
            print(f"Cubes for experiment: {self.cubes}")
        
        log(msg="Calculate meeting points")
        
        left_arm = env.arm_base_location[LocationType.LEFT]
        right_arm = env.arm_base_location[LocationType.RIGHT]
        tool_len = inverse_kinematics.tool_length
        
        right_x_bias = 0.5
        right_y_bias = 0.6
        
        left_meeting_rpy = [np.pi / 2, np.pi / 2, np.pi * 1 / 2]  # TODO Validate via simulation
        right_meeting_rpy = [np.pi / 2, 0, -np.pi / 2]
        
        base_meeting_coords = [
            ((1 - right_x_bias) * left_arm[0] + right_x_bias * right_arm[0]),
            ((1 - right_y_bias) * left_arm[1] + right_y_bias * right_arm[1]),
            0.5
        ]
        left_meeting_coords = (np.array(base_meeting_coords) + np.array([-tool_len/2, 0, 0])).tolist()
        right_meeting_coords = (np.array(base_meeting_coords) + np.array([tool_len/2, 0, 0])).tolist()
        
        transformation_left = transform_left.get_base_to_tool_transform(
            position=left_meeting_coords, rpy=left_meeting_rpy)
        transformation_right = transform_right.get_base_to_tool_transform(
            position=right_meeting_coords, rpy=right_meeting_rpy)
        
        env.set_active_arm(LocationType.LEFT)
        env.update_obstacles([], self.right_arm_home)
        self.left_arm_meeting_confs = bb_left.validate_IK_solutions(
            inverse_kinematics.inverse_kinematic_solution(
                inverse_kinematics.DH_matrix_UR5e, transformation_left),
            transformation_left)
        
        env.set_active_arm(LocationType.RIGHT)
        env.update_obstacles([], self.left_arm_home)
        self.right_arm_meeting_confs = bb_right.validate_IK_solutions(
            inverse_kinematics.inverse_kinematic_solution(
                inverse_kinematics.DH_matrix_UR5e, transformation_right),
            transformation_right)
        
        print(f"Left meeting confs: {len(self.left_arm_meeting_confs)}")
        print(f"Right meeting confs: {len(self.right_arm_meeting_confs)}")
        
        if DEMO:
            for conf in self.right_arm_meeting_confs:
                visualizer.draw_two_robots(conf_left=self.left_arm_meeting_confs[0], conf_right=conf)
            return
        
        log(msg="Start dual-arm planning with parallel execution")
        
        left_arm_current = self.left_arm_home
        right_arm_current = self.right_arm_home
        
        for cube_i in range(len(self.cubes)):
            print(f"\n{'='*50}")
            print(f"Processing cube {cube_i + 1}/{len(self.cubes)}")
            print(f"{'='*50}")
            
            left_arm_current, right_arm_current = self.plan_single_cube_passing(
                cube_i=cube_i, cubes=self.cubes,
                left_arm_start=left_arm_current, right_arm_start=right_arm_current,
                env=env, bb_right=bb_right, bb_left=bb_left,
                transform_left=transform_left, transform_right=transform_right,
                ur_params_left=ur_params_left, ur_params_right=ur_params_right
            )
        
        t_end = time.time()
        print(f"\n{'='*50}")
        print(f"Total planning time: {t_end - start_time:.2f} seconds")
        print(f"{'='*50}")
        
        json_object = json.dumps(self.experiment_result, indent=4)
        dir_path = r"./outputs/"
        with open(dir_path + "plan_dual.json", "w") as outfile:
            outfile.write(json_object)
        
        print(f"Results saved to {dir_path}plan_dual.json")
        
        visualizer.show_all_experiment(dir_path + "plan_dual.json")
        visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id, env):
        """Generate cube positions (same as Experiment.py)."""
        cube_side = 0.04
        cubes = []
        offset = env.cube_area_corner[LocationType.RIGHT]
        
        if experiment_id == 1:
            x_min, x_max = 0.0, 0.4
            y_min, y_max = 0.0, 0.4
            x_slice = (x_max - x_min) / 4.0
            y_slice = (y_max - y_min) / 2.0
            pos = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos + offset).tolist())
        elif experiment_id == 2:
            x_min, x_max = 0.0, 0.4
            y_min, y_max = 0.0, 0.4
            x_slice = (x_max - x_min) / 4.0
            y_slice = (y_max - y_min) / 2.0
            pos1 = np.array([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            cubes.append((pos1 + offset).tolist())
            pos2 = np.array([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
            cubes.append((pos2 + offset).tolist())
        
        return cubes


def main():
    print("=" * 60)
    print("DUAL-ARM EXPERIMENT - PARALLEL PLANNING & EXECUTION")
    print("=" * 60)
    print("\nThis plans paths for both arms.")
    print("Use live_demo.py with run_json_parallel() to execute on real robots.")
    print("")
    
    experiment = DualArmExperiment()
    experiment.plan_experiment()


if __name__ == "__main__":
    main()
