import json
import numpy as np
import threading
from environment import Environment, LocationType
import inverse_kinematics
from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from building_blocks import BuildingBlocks3D
from visualizer import Visualize_UR
from Control_robot import BaseRobot
from numpy import pi
import time
from Experiment import Experiment

left_arm_ip = "192.168.0.11"  # TODO
right_arm_ip = "192.168.0.10"  # TODO

def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)

def draw_two_robots():
    ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
    ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)
    env = Environment(ur_params=ur_params_right)
    right_arm_rotation = [0, 0, -np.pi/2] 
    left_arm_rotation = [0, 0, np.pi/2]
    transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],ur_rotation=right_arm_rotation)
    transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],ur_rotation=left_arm_rotation)
    visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                transform_left_arm=transform_left_arm)
    right_meeting_conf = [
                    -2.9934641040971934,
                    -2.930924293268491,
                    0.8826038548093044,
                    2.048320438459186,
                    0.1481285494926033,
                    -3.1415926535897927
                ]
    right_meeting_conf[0] += np.pi/2
    left_meeting_conf = [
                    2.8495422749821753,
                    -1.0553575187730182,
                    0.8826038548093021,
                    0.17275366396371578,
                    2.849542274982176,
                    -2.0366847201866515e-16
                ]
    left_meeting_conf[0] -= np.pi/2
    visualizer.draw_two_robots(conf_left=left_meeting_conf,conf_right=right_meeting_conf)

def connect_move_robots():
    
    robot = BaseRobot(left_arm_ip, right_arm_ip)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
    robot.robot_left.move_home()
    robot.robot_right.move_home()
    time.sleep(2)
    p1 = np.deg2rad([9.51, -118.24, -97.15, -56.06, 90.18, 9.58])
    p2 = np.deg2rad([-106.02, -133.07, -94, 29.18, 16.48, -66.73])
    path = [home_config, p1, p2]
    print("Testing movement for right arm from", path[1], "to", path[2])
    speed = 0.5
    acceleration = 0.5
    blend_radius = 0.05
    path_with_params = [
        [*target_config, speed, acceleration, blend_radius] for target_config in path
    ]
    path_with_params[-1][-1] = 0
    robot.robot_right.moveJ(path_with_params, asynchronous=False) # Default is false, this means the function will block until the movement has completed.
    # https://sdurobotics.gitlab.io/ur_rtde/api/api.html#_CPPv4N7ur_rtde20RTDEControlInterface5moveJERKNSt6vectorINSt6vectorIdEEEEb
    robot.close_connection()

def run_json(json_path):
    # left_arm_ip = "192.168.0.17"  # TODO
    # right_arm_ip = "192.168.0.10"  # TODO
    robot = BaseRobot(left_arm_ip, right_arm_ip)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
    robot.robot_left.move_home()
    robot.robot_right.move_home()
    time.sleep(2)

    start_time = time.time()

    with open(json_path, 'r') as openfile:
            # Reading from json file
            steps = json.load(openfile)
            for step in steps:
                # iterate over the step elements
                for i in range(len(step["active_id"])):
                    print(step["description"][i])
                    # which arm are we moving?
                    arm_id = step["active_id"][i]
                    robot.set_active_robot(arm_id)
                    # first, check gripper pre status
                    robot.gripper_action(robot.active_robot, step["gripper_pre"][i])
                    # now move according to the path
                    curr_conf = step["path"][i][0]
                    if step["command"][i] == "move":
                        # robot.move([curr_conf, conf])
                        robot.move_path(step["path"][i],False) # Default is false, this means the function will block until the movement has completed.
                        # super(robot_interface.RobotInterfaceWithGripper, robot.active_robot).move_path([curr_conf, conf])
                        # super(robot_interface.RobotInterfaceWithGripper, robot.active_robot).getInverseKinematics()
                    elif step["command"][i] == "movel":
                        relative_pose = step["path"][i]
                        robot.moveL_relative(relative_pose)
                    # lastly, check gripper post status
                    robot.gripper_action(robot.active_robot, step["gripper_post"][i])

    end_time = time.time()
    print("run took ", end_time-start_time, " seconds")

    robot.close_connection()


def run_json_parallel(json_path):
    """
    Run experiment with PARALLEL execution for steps marked with [PARALLEL-R] and [PARALLEL-L].
    Both arms move SIMULTANEOUSLY using threads for those steps.
    """
    robot = BaseRobot(left_arm_ip, right_arm_ip)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]

    print("Moving both arms to home position...")
    robot.robot_left.move_home()
    robot.robot_right.move_home()
    time.sleep(2)

    def move_path_async(robot_arm, path):
        """Move robot along path asynchronously (non-blocking)."""
        speed = 0.5
        acceleration = 0.5
        blend_radius = 0.05
        path_with_params = [[*config, speed, acceleration, blend_radius] for config in path]
        path_with_params[-1][-1] = 0  # Last point should have 0 blend radius
        robot_arm.moveJ(path_with_params, asynchronous=True)

    def wait_for_robot(robot_arm, target_config, threshold=0.1):
        """Wait until robot reaches target configuration."""
        while True:
            current = robot_arm.getActualQ()
            dist = np.linalg.norm(np.array(current) - np.array(target_config))
            if dist < threshold:
                break
            time.sleep(0.01)

    def execute_parallel_motion(right_path, left_path):
        """Execute both robot paths SIMULTANEOUSLY using threads."""
        print(">>> PARALLEL EXECUTION: Both arms moving simultaneously!")

        right_target = right_path[-1]
        left_target = left_path[-1]

        t_start = time.time()

        # Start both robots at the same time using threads
        def start_right():
            move_path_async(robot.robot_right, right_path)

        def start_left():
            move_path_async(robot.robot_left, left_path)

        thread_right = threading.Thread(target=start_right)
        thread_left = threading.Thread(target=start_left)

        thread_right.start()
        thread_left.start()

        thread_right.join()
        thread_left.join()

        # Wait for both robots to complete their motion (in parallel)
        def wait_right():
            wait_for_robot(robot.robot_right, right_target)

        def wait_left():
            wait_for_robot(robot.robot_left, left_target)

        wait_thread_right = threading.Thread(target=wait_right)
        wait_thread_left = threading.Thread(target=wait_left)

        wait_thread_right.start()
        wait_thread_left.start()

        wait_thread_right.join()
        wait_thread_left.join()

        t_end = time.time()
        print(f"    Parallel execution completed in {t_end - t_start:.2f}s")

    with open(json_path, 'r') as openfile:
        steps = json.load(openfile)
        for step in steps:
            i = 0
            while i < len(step["active_id"]):
                description = step["description"][i]

                # Check for parallel motion marker
                if (description.startswith("[PARALLEL-R]") and
                    i + 1 < len(step["active_id"]) and
                    step["description"][i + 1].startswith("[PARALLEL-L]")):

                    print(f"\n>>> {description.replace('[PARALLEL-R] ', '')} (SIMULTANEOUS)")

                    # Get both paths
                    right_path = step["path"][i]
                    left_path = step["path"][i + 1]

                    # Handle grippers before motion
                    robot.gripper_action(robot.robot_right, step["gripper_pre"][i])
                    robot.gripper_action(robot.robot_left, step["gripper_pre"][i + 1])

                    # Execute PARALLEL motion using threads!
                    execute_parallel_motion(right_path, left_path)

                    # Handle grippers after motion
                    robot.gripper_action(robot.robot_right, step["gripper_post"][i])
                    robot.gripper_action(robot.robot_left, step["gripper_post"][i + 1])

                    i += 2  # Skip both parallel steps
                    continue

                # Regular single-arm motion (sequential)
                print(f"\n>>> {description}")

                arm_id = step["active_id"][i]
                robot.set_active_robot(arm_id)

                # Gripper pre
                robot.gripper_action(robot.active_robot, step["gripper_pre"][i])

                # Execute motion
                if step["command"][i] == "move":
                    robot.move_path(step["path"][i], False)
                elif step["command"][i] == "movel":
                    relative_pose = step["path"][i]
                    robot.moveL_relative(relative_pose)
                    time.sleep(1.0)

                # Gripper post
                robot.gripper_action(robot.active_robot, step["gripper_post"][i])

                i += 1

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE WITH PARALLEL EXECUTION")
    print("="*60)
    robot.close_connection()

def create_json():
    exp1 = Experiment()
    exp1.plan_experiment()

def animation(plan_file):
    ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
    ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)
    env = Environment(ur_params=ur_params_right)
    right_arm_rotation = [0, 0, -np.pi/2] 
    left_arm_rotation = [0, 0, np.pi/2]
    transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],ur_rotation=right_arm_rotation)
    transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],ur_rotation=left_arm_rotation)
    visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                transform_left_arm=transform_left_arm)
    dir_path = ""
    visualizer.show_all_experiment(dir_path + plan_file)
    visualizer.animate_by_pngs()

def IK():
    ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
    ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)
    env = Environment(ur_params=ur_params_right)
    right_arm_rotation = [0, 0, -np.pi/2] 
    left_arm_rotation = [0, 0, np.pi/2]
    transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],ur_rotation=right_arm_rotation)
    transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],ur_rotation=left_arm_rotation)
    visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                transform_left_arm=transform_left_arm)
    bb_right = BuildingBlocks3D(transform=transform_right_arm,ur_params=ur_params_right,env=env,
                            resolution=0.1)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
    env.arm_transforms[LocationType.RIGHT] = transform_right_arm
    env.arm_transforms[LocationType.LEFT] = transform_left_arm
    update_environment(env, LocationType.RIGHT, home_config, [])
    # point in world space for right robot [m]
    p = [1.93,0.45,0.1]
    # orintation of end-effector relative to world [rad]
    roll,pith,yaw = [np.pi,0,0]
    transformation_matrix_base_to_tool = transform_right_arm.get_base_to_tool_transform(position=p,rpy=[roll,pith,yaw])
    IK_configurations = inverse_kinematics.inverse_kinematic_solution(inverse_kinematics.DH_matrix_UR5e,transformation_matrix_base_to_tool)
    valid_conf = bb_right.validate_IK_solutions(IK_configurations,transformation_matrix_base_to_tool)
    visualizer.draw_two_robots(conf_left=home_config,conf_right=valid_conf[0])
    print("visualized IK solution:",valid_conf[0])
#     now do the same for the confs: [array([ 2.96593145e+00, -2.82992828e+00,  0.00000000e+00,  2.82992828e+00,
#         6.09736961e-01,  8.76591808e-17]), array([ 2.96593145e+00, -2.82992828e+00, -0.00000000e+00,  2.82992828e+00,
#         6.09736961e-01,  8.76591808e-17]), array([ 2.96593145e+00, -2.86626543e+00,  0.00000000e+00,  2.86626543e+00,
#        -6.09736961e-01,  8.76591808e-17]), array([ 2.96593145e+00, -2.86626543e+00, -0.00000000e+00,  2.86626543e+00,
#        -6.09736961e-01,  8.76591808e-17])]
#     confs = [np.array([ 2.96593145e+00, -2.82992828e+00,  0.00000000e+00,  2.82992828e+00,
#         6.09736961e-01,  8.76591808e-17]), np.array([ 2.96593145e+00, -2.82992828e+00, -0.00000000e+00,  2.82992828e+00,
#         6.09736961e-01,  8.76591808e-17]), np.array([ 2.96593145e+00, -2.86626543e+00,  0.00000000e+00,  2.86626543e+00,
#        -6.09736961e-01,  8.76591808e-17]), np.array([ 2.96593145e+00, -2.86626543e+00, -0.00000000e+00,  2.86626543e+00,
#        -6.09736961e-01,  8.76591808e-17])]
#
#     for conf in confs:
#         # valid_conf = bb_right.validate_IK_solutions([conf],transformation_matrix_base_to_tool)
#         # if valid_conf:
#         visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
#                                   transform_left_arm=transform_left_arm)
#         visualizer.draw_two_robots(conf_left=home_config,conf_right=conf)
#         print("visualized IK solution:",conf)
#         time.sleep(2)

def visualize_conf_right(conf):
    ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
    ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)
    env = Environment(ur_params=ur_params_right)
    right_arm_rotation = [0, 0, -np.pi / 2]
    left_arm_rotation = [0, 0, np.pi / 2]
    transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],
                                    ur_rotation=right_arm_rotation)
    transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],
                                   ur_rotation=left_arm_rotation)
    visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                              transform_left_arm=transform_left_arm)
    bb_right = BuildingBlocks3D(transform=transform_right_arm, ur_params=ur_params_right, env=env,
                                resolution=0.1)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
    env.arm_transforms[LocationType.RIGHT] = transform_right_arm
    env.arm_transforms[LocationType.LEFT] = transform_left_arm
    update_environment(env, LocationType.RIGHT, home_config, [])

    visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                              transform_left_arm=transform_left_arm)
    visualizer.draw_two_robots(conf_left=home_config, conf_right=conf)


def IK_experiment():
    exp1 = Experiment()
    left, right = exp1.plan_experiment(DEMO=True)
    # for conf in right:
    #     visualize_conf_right(conf)


if __name__ == '__main__':
    # draw_two_robots()
    # connect_move_robots()

    # Regular sequential execution (original):
    # run_json("outputs/plan.json")

    # PARALLEL execution (new - uses threads for simultaneous motion):
    run_json_parallel("outputs/plan_dual.json")

    #create_json()
    # animation("plan_fixed.json")
    # IK()
    # IK_experiment()
    
    