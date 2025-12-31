import numpy as np
import math


class UR5e_PARAMS(object):
    '''
    UR5e_PARAMS determines physical costants for UR5e Robot manipulator
    @param inflation_factor - by what factor inflate the minimal sphere radious of each link
    @param tool_lenght - the lenght of the tool [meters]. for the gripper, set 0.135 meter 
    '''
    def __init__(self, inflation_factor=1.0, tool_lenght=0.145): #0.135 => 0.185
                    # alpha, a, d, theta_const
        self.ur_DH = [ [0, 0, 0.1625, 0], # shoulder - > base_link
                    [np.pi/2, 0,0, 0], #upper_arm_link - > shoulder
                    [0, -0.425, 0, 0], # forearm_link -> upper_arm_link
                    [0, -0.3922, 0.1333, 0], # wrist1 -> forearm_link
                    [np.pi/2, 0, 0.0997, 0], # wrist2 -> wrist1
                    [-np.pi/2, 0, 0.0996, 0]] # wrist3 -> wrist2

        self.ur_geometry = [['shoulder_link', np.array([0,0, -0.1625]), 'z'],
                    ['upper_arm_link',   np.array([-0.425,0,0.135]), 'x'],
                    ['upper_arm_link',   np.array([0,0,0.135]), 'z'],
                    ['forearm_link'  ,   np.array([0,0,0.135]),'z'],
                    ['forearm_link'  ,   np.array([-0.392,0,0.015]),'x'],
                    ['wrist_1_link',     np.array([0,0,-0.109]), 'z'],
                    ['wrist_1_link',     np.array([0,-0.11,0]), 'y'],
                    ['wrist_1_link',     np.array([0,0.05,0]), 'y'],
                    ['wrist_2_link',     np.array([0,0.0825,0]), 'y'],
                    # ['wrist_2_link',     np.array([0,-0.05,0]), 'y'],
                    ['wrist_3_link',     np.array([0.1,-0.11,0]), 'x'], # test
                    ['wrist_3_link',     np.array([-0.1,-0.11,0]), 'x'], # test
                    ['wrist_3_link',     np.array([0,-0.15,0]), 'y'], # test
                    ['wrist_3_link',     np.array([0,0, tool_lenght]), 'z']]

        self.ur_links = [ 'shoulder_link', 'upper_arm_link',
                        'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']

        self.ur_links_color = {'base_link':'gray', 'base_link_inertia':'gray',
                               'shoulder_link':'gray','upper_arm_link':'yellow',
                             'forearm_link':'green', 'wrist_1_link':'purple'
                            , 'wrist_2_link':'black', 'wrist_3_link':'blue'}

        self.mechamical_limits = {'shoulder_link':[-np.pi, np.pi], 'upper_arm_link':[-np.pi, np.pi],
                                   'forearm_link':[-np.pi, np.pi], 'wrist_1_link':[-np.pi, np.pi],
                                    'wrist_2_link':[-np.pi, np.pi], 'wrist_3_link':[-np.pi, np.pi]}

        self.min_sphere_radius = {'shoulder_link':0.06, 'upper_arm_link':0.05, 'forearm_link':0.05,
                        'wrist_1_link':0.04, 'wrist_2_link':0.04, 'wrist_3_link':0.04}

        self.sphere_radius = dict()
        for key, val in self.min_sphere_radius.items():
            self.sphere_radius[key] = val * inflation_factor


class UR5e_without_camera_PARAMS(object):
    '''
    UR5e_PARAMS determines physical costants for UR5e Robot manipulator
    @param inflation_factor - by what factor inflate the minimal sphere radious of each link
    @param tool_lenght - the lenght of the tool [meters]. for the gripper, set 0.135 meter
    '''

    def __init__(self, inflation_factor=1.0, tool_lenght=0.145):
        # alpha, a, d, theta_const
        self.ur_DH = [[0, 0, 0.1625, 0],  # shoulder - > base_link
                      [np.pi / 2, 0, 0, 0],  # upper_arm_link - > shoulder
                      [0, -0.425, 0, 0],  # forearm_link -> upper_arm_link
                      [0, -0.3922, 0.1333, 0],  # wrist1 -> forearm_link
                      [np.pi / 2, 0, 0.0997, 0],  # wrist2 -> wrist1
                      [-np.pi / 2, 0, 0.0996, 0]]  # wrist3 -> wrist2

        self.ur_geometry = [['shoulder_link', np.array([0, 0, -0.1625]), 'z'],
                            ['upper_arm_link', np.array([-0.425, 0, 0.135]), 'x'],
                            ['upper_arm_link', np.array([0, 0, 0.135]), 'z'],
                            ['forearm_link', np.array([0, 0, 0.135]), 'z'],
                            ['forearm_link', np.array([-0.392, 0, 0.015]), 'x'],
                            ['wrist_1_link', np.array([0, 0, -0.109]), 'z'],
                            ['wrist_1_link', np.array([0, -0.11, 0]), 'y'],
                            ['wrist_1_link', np.array([0, 0.05, 0]), 'y'],
                            ['wrist_2_link', np.array([0, 0.0825, 0]), 'y'],
                            ['wrist_2_link', np.array([0, -0.05, 0]), 'y'],
                            ['wrist_3_link', np.array([0, 0, tool_lenght]), 'z']]

        self.ur_links = ['shoulder_link', 'upper_arm_link',
                         'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']

        self.ur_links_color = {'base_link': 'gray', 'base_link_inertia': 'gray',
                               'shoulder_link': 'gray', 'upper_arm_link': 'yellow',
                               'forearm_link': 'green', 'wrist_1_link': 'purple'
            , 'wrist_2_link': 'black', 'wrist_3_link': 'blue'}

        self.mechamical_limits = {'shoulder_link': [-np.pi, np.pi], 'upper_arm_link': [-np.pi, np.pi],
                                  'forearm_link': [-np.pi, np.pi], 'wrist_1_link': [-np.pi, np.pi],
                                  'wrist_2_link': [-np.pi, np.pi], 'wrist_3_link': [-np.pi, np.pi]}

        self.min_sphere_radius = {'shoulder_link': 0.06, 'upper_arm_link': 0.05, 'forearm_link': 0.05,
                                  'wrist_1_link': 0.04, 'wrist_2_link': 0.04, 'wrist_3_link': 0.04}

        self.sphere_radius = dict()
        for key, val in self.min_sphere_radius.items():
            self.sphere_radius[key] = val * inflation_factor


class Transform(object):
    '''
    Trasform class implemets the Forward kinematics method
    to finds the coordinates of the spheres along the manipulator's links
    
    @param ur_params: An object containing the UR robot's physical parameters (DH table, link geometry, etc.).
    @param ur_location: A list or array [x, y, z] representing the base location of the UR robot in the world frame.
    @param ur_rotation: A list or array [roll, pitch, yaw] representing the rotation of the UR robot's base in the world frame (in radians). according to https://msl.cs.uiuc.edu/planning/node102.html
    '''
    def __init__(self, ur_params, ur_location, ur_rotation=[0, 0, 0]):
        
        self.ur = ur_params.ur_DH
        self.frame_list = ur_params.ur_links
        self.sphere_radius = ur_params.sphere_radius
        self.local_sphere_coords = []
        self.robot_geometry = ur_params.ur_geometry
        self.local_sphere_coords = dict()
        self.location = np.array([ur_location[0], ur_location[1], ur_location[2], 1.0])

        # Create the base to world transformation matrix
        roll, pitch, yaw = ur_rotation
        rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = np.dot(rz, np.dot(ry, rx))

        # trasformation matrix - base to world:
        self.base_transform = np.eye(4)
        self.base_transform[0:3, 0:3] = rotation_matrix
        self.base_transform[0:3, 3] = ur_location

        # self.location is kept for compatibility if other parts of the code use it for translation only.
        self.location = np.array([ur_location[0], ur_location[1], ur_location[2], 1.0])


        # create a simplified geometric model of a robot arm for collision detection purposes
        for frame in self.frame_list:
            self.local_sphere_coords[frame] = []
        axis_dict = {'x': 0, 'y': 1, 'z': 2}
        for frame, offset, axis in self.robot_geometry:
            spheres_amount =  max(int(math.ceil(abs(offset[axis_dict[axis]] / (self.sphere_radius[frame])))+1), 3)
            local_sphere_offset = np.linspace(0, offset[axis_dict[axis]], num=spheres_amount, endpoint=True)
            for sphere_offset in local_sphere_offset:
                if axis == 'x':
                    self.local_sphere_coords[frame].append(np.array([sphere_offset, offset[1],offset[2],1], dtype= float))
                elif axis =='y':
                    self.local_sphere_coords[frame].append(np.array([offset[0],sphere_offset,offset[2],1], dtype=float))
                else:
                    self.local_sphere_coords[frame].append(np.array([offset[0],offset[1],sphere_offset ,1], dtype=float))
        
    def get_trans_matrix(self, conf):
        '''
        Returns the transformation matrix for given configuration 
        '''
        trans = [[] for _ in range(len(self.ur))]
        for i in range(len(self.ur)):
            alpha, a, d, theta_const =  self.ur[i]
            theta = conf[i] + theta_const
            # Trasformation from frame {i} to frame {i-1}
            trans[i] = np.asarray([[np.cos(theta), -np.sin(theta), 0, a],
                                [np.sin(theta) * np.cos(alpha) , np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
                                [np.sin(theta) * np.sin(alpha) , np.cos(theta) * np.sin(alpha), np.cos(alpha),  d * np.cos(alpha)],
                                [0,0,0,1]], dtype="object") #dtype=float
        trans_matrix = dict()
        trans_matrix['shoulder_link'] = trans[0]
        for i in range(1, len(trans)):
            trans_matrix[self.frame_list[i]] = np.matmul(trans_matrix[self.frame_list[i-1]] , trans[i])
        return trans_matrix

    def get_global_sphere_coords(self, trans_matrix):
        '''
        Returns the coordinates of the spheres along the manipulator for a given transformation matrix
        '''
        global_sphere_coords = dict()
        for frame in self.local_sphere_coords.keys():
            global_sphere_coords[frame] = []
            for i in range(len(self.local_sphere_coords[frame])):
                coords_in_base = np.matmul(trans_matrix[frame], self.local_sphere_coords[frame][i].T)
                current_global_coords = np.matmul(self.base_transform, coords_in_base)
                global_sphere_coords[frame].append(current_global_coords)
        return global_sphere_coords
    
    def conf2sphere_coords(self, conf):
        '''
        Returns the coordinates of the spheres along the manipulator's links for a given configuration,
        in the base_link frame
        @param conf - some configuration
        '''
        trans_matrix = self.get_trans_matrix(conf)
        return self.get_global_sphere_coords(trans_matrix)
    
    def world_to_base_frame(self, world_coords):
        """
        Transforms world coordinates to the robot's base frame.
        @param world_coords: A numpy array or list [x, y, z] in world coordinates.
        @return: A numpy array [x, y, z] in the robot's base frame.
        """
        world_coords_homogeneous = np.array([world_coords[0], world_coords[1], world_coords[2], 1.0])
        # @ - This is Python's dedicated operator for matrix multiplication, introduced in version 3.5.
        base_coords_homogeneous = np.linalg.inv(self.base_transform) @ world_coords_homogeneous
        return base_coords_homogeneous[:3]
    
    def rpy_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Converts roll, pitch, yaw angles to a 3x3 rotation matrix.
        @param roll: Rotation around the X-axis (in radians).
        @param pitch: Rotation around the Y-axis (in radians).
        @param yaw: Rotation around the Z-axis (in radians).
        @return: A 3x3 NumPy array representing the rotation matrix.
        """
        Rx = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])

        Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                       [0, 1, 0],
                       [-math.sin(pitch), 0, math.cos(pitch)]])

        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                       [math.sin(yaw), math.cos(yaw), 0],
                       [0, 0, 1]])

        # Combined rotation matrix (R = Rz * Ry * Rx)
        R = np.dot(Rz, np.dot(Ry, Rx))
        return R
    def get_base_to_tool_transform(self, position, rpy):
        """
        Calculates the transformation matrix from the world frame to the tool frame.
        @param position: A list or array [x, y, z] representing the tool's position in world coordinates.
        @param rpy: A list or array [roll, pitch, yaw] representing the tool's orientation in world coordinates (in radians).
        @return: A 4x4 NumPy array representing the transformation matrix from base to tool frame.
        """
        roll, pitch, yaw = rpy
        rotation_matrix_in_world = self.rpy_to_rotation_matrix(roll, pitch, yaw)
        transform_matrix_world_tool = np.eye(4)
        transform_matrix_world_tool[0:3, 0:3] = rotation_matrix_in_world
        transform_matrix_world_tool[0:3, 3] = position
        transform_matrix_base_tool = np.linalg.inv(self.base_transform) @ transform_matrix_world_tool
        return transform_matrix_base_tool
