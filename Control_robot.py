import json
from time import sleep, time
import numpy as np

from Experiment import log, Gripper
from environment import LocationType
# grapper
from numpy import pi
import time
import robot_interface


class BaseRobot:
    ip_left: int
    ip_right: int
    home = [0, -pi / 2, 0, -pi / 2, 0, 0]
    left_arm_gripper: robot_interface.RobotInterfaceWithGripper
    right_arm_gripper: robot_interface.RobotInterfaceWithGripper
    active_robot: robot_interface.RobotInterfaceWithGripper

    def __init__(self, left_arm_ip, right_arm_ip):
        self.robot_left = None
        while self.robot_left is None:
            try:
                self.robot_left = robot_interface.RobotInterfaceWithGripper(left_arm_ip)
            except:
                print('Cannot connect to robot. Retrying...')
                sleep(5)
        self.robot_right = None
        while self.robot_right is None:
            try:
                self.robot_right = robot_interface.RobotInterfaceWithGripper(right_arm_ip)
            except:
                print('Cannot connect to robot. Retrying...')
                sleep(5)
        self.active_robot = None

    def set_active_robot(self, robot_type:LocationType):
        if robot_type == LocationType.LEFT:
            self.active_robot = self.robot_left
            self.arm_rotation_fix = self.left_arm_rotation_shift
        if robot_type == LocationType.RIGHT:
            self.active_robot = self.robot_right
            self.arm_rotation_fix = self.right_arm_rotation_shift

    def move(self, config):
        dist = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
        # print('Moving to:', config)
        try:
            # print(f'moving to {config}')
            self.active_robot.movej(config, acc=10, vel=0.5)
        except:
            pass

        while dist(self.get_config(), config) > 0.1:
            pass

    def movel(self, target_pose):
        dist = lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
        # print('Moving to:', config)
        try:
            # print(f'moving to {config}')
            self.active_robot.movel(target_pose, acc=10, vel=0.5)
        except:
            pass

        while dist(self.get_pose()[0:3], target_pose) > 0.1:
            pass

    def move_all(self, config_list):
        # movexs(self, command, pose_list, acc=0.01, vel=0.01, radius=0.01, wait=True, threshold=None):
        self.active_robot.movexs('movej', config_list, acc=50, vel=0.5, radius=0.1)

    def move_home(self):
        self.move(self.home)

    def get_config(self):
        return self.active_robot.getj()

    def get_pose(self):
        return self.active_robot.getl()

    def gripper_action(self, active_arm:robot_interface.RobotInterfaceWithGripper,
                       gripper_status:Gripper):
        # gripper_arm = self.robot_right
        # if arm_id == LocationType.LEFT:
        #     gripper_arm = self.robot_right
        if gripper_status == Gripper.OPEN:
            # open gripper
            active_arm.release_grasp()
        elif gripper_status == Gripper.CLOSE:
            # close gripper
            active_arm.grasp()

    def execute_path(self, experiment_json, timing_profile=None):
        """
        Executes the path
        :param path: List of configs
        :param timing_profile: Transition times between each pair of configs - format: real times!
        """
        dir_path = r"./outputs/"
        start_time = time.time()
        with open(dir_path + experiment_json, 'r') as openfile:
            # Reading from json file
            steps = json.load(openfile)
            for step in steps:
                # iterate over the step elements
                for i in range(len(step["active_id"])):
                    log(step["description"][i])
                    # which arm are we moving?
                    arm_id = step["active_id"][i]
                    self.set_active_robot(arm_id)
                    # first, check gripper pre status
                    self.gripper_action(self.active_robot, step["gripper_pre"][i])
                    # now move according to the path
                    curr_conf = step["path"][i][0]
                    if step["command"][i] == "move":
                        # robot.move([curr_conf, conf])
                        self.move_path(step["path"][i],False) # Default is false, this means the function will block until the movement has completed.
                        # super(robot_interface.RobotInterfaceWithGripper, robot.active_robot).move_path([curr_conf, conf])
                        # super(robot_interface.RobotInterfaceWithGripper, robot.active_robot).getInverseKinematics()
                    elif step["command"][i] == "movel":
                        relative_pose = step["path"][i]
                        self.moveL_relative(relative_pose)
                    # lastly, check gripper post status
                    self.gripper_action(robot.active_robot, step["gripper_post"][i])
    def move_path(self, path, asynchronous = False):
        speed = 0.5
        acceleration = 0.5
        blend_radius = 0.05
        path_with_params = [
            [*target_config, speed, acceleration, blend_radius] for target_config in path
        ]
        path_with_params[-1][-1] = 0 # reach the end point
        self.active_robot.moveJ(path_with_params, asynchronous=asynchronous)

    def moveL_relative(self, relative_pose):
        self.active_robot.moveL_relative(relative_pose)

    def close_connection(self):
        """
        Stops the control script and disconnects from both robots.
        """
        if self.robot_left:
            print(f"Disconnecting from left robot ({self.robot_left.robot_ip})...")
            # self.robot_left.stopControl()
            self.robot_left.disconnect()
        if self.robot_right:
            print(f"Disconnecting from right robot ({self.robot_right.robot_ip})...")
            # self.robot_right.stopControl()
            self.robot_right.disconnect()


if __name__ == '__main__':
    left_arm_ip = "TODO"#TODO
    right_arm_ip = "TODO"#TODO
    experiment_json = "plan.json"
    robot = BaseRobot(left_arm_ip, right_arm_ip)
    robot.robot_left.move_home()
    robot.robot_right.move_home()
    robot.execute_path(experiment_json)
    robot.close_connection()
