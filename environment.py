
import numpy as np
from enum import Enum


class LocationType(int, Enum):
    LEFT = 1
    RIGHT = 2


class Environment(object):
    '''
    Environment class implements the physical robot's environment 
    '''
    def __init__(self, ur_params):
        self.radius = 0.05
        self.ur_params = ur_params
        self.obstacles_non_np = []
        self.obstacles = None
        # env measurements (CM)
        area_size = 40
        manipulator_size = 20
        # Left arm table
        left_table_left = 50
        left_table_right = left_table_left + 184.4
        left_table_bottom = 136.8
        left_table_top = left_table_bottom + 84
        # Left arm
        left_arm_left = left_table_left + 11.2
        left_arm_right = left_arm_left + manipulator_size
        left_arm_bottom = left_table_bottom + 28.1
        left_arm_top = left_arm_bottom + manipulator_size
        # Right table
        right_table_left = 151
        right_table_right = right_table_left + 84
        right_table_bottom = 0
        right_table_top = right_table_bottom + 127.7
        # Right arm
        right_arm_left = right_table_left + 32
        right_arm_right = right_arm_left + manipulator_size
        right_arm_bottom = right_table_bottom + 87.4
        right_arm_top = right_arm_bottom + manipulator_size
        ## CUBE AREAS
        # left area
        left_area_left = 0
        left_area_right = left_area_left + area_size
        left_area_bottom = left_table_bottom + 22
        left_area_top = left_area_bottom + area_size
        # right area
        right_area_left = right_table_left + 22
        right_area_right = right_area_left + area_size
        right_area_bottom = right_table_bottom + 25
        right_area_top = right_area_bottom + area_size
        ## SET THE VARS
        self.active_arm = None

        self.arm_base_location = {
            LocationType.LEFT:  [(left_arm_left + left_arm_right) / (2.0 * 100),
                             (left_arm_bottom + left_arm_top) / (2.0 * 100), 0],
            LocationType.RIGHT:  [(right_arm_left + right_arm_right) / (2.0 * 100),
                              (right_arm_bottom + right_arm_top) / (2.0 * 100), 0]
        }

        self.arm_transforms = {
            LocationType.LEFT: None,
            LocationType.RIGHT: None
        }

        self.cube_areas = {
            LocationType.LEFT: [[left_area_left / 100.0, left_area_top / 100.0],
                                     [left_area_right / 100.0, left_area_bottom / 100.0]],
            LocationType.RIGHT: [[right_area_left / 100.0, right_area_top / 100.0],
                                      [right_area_right / 100.0, right_area_bottom / 100.0]]
        }
        self.cube_area_corner = {
            LocationType.LEFT: np.array([left_area_left / 100.0, left_area_bottom / 100.0, 0]),
            LocationType.RIGHT: np.array([right_area_left / 100.0, right_area_bottom / 100.0, 0])
        }
        self.tables = {
            LocationType.LEFT: [[left_table_left / 100.0, left_table_top / 100.0],
                                     [left_table_right / 100.0, left_table_bottom / 100.0]],
            LocationType.RIGHT: [[right_table_left / 100.0, right_table_top / 100.0],
                                      [right_table_right / 100.0, right_table_bottom / 100.0]]
        }
        self.bbox = [[0, left_table_top / 100.0], [right_table_right / 100.0, 0]]

        wall_height = 0.3
        self.walls = [
            [(self.arm_base_location[LocationType.LEFT][0] + self.arm_base_location[LocationType.RIGHT][0]) / 2.0, 0, left_table_top / 100.0, 0, wall_height, 0]
        ]

        for wall in self.walls:
            if wall[5] == 0:
                x_static_from_left_table_side = round((wall[0]) * 100, 1) - left_table_left
                log_to_write = (f"Wall: X static on x={x_static_from_left_table_side}CM from the left side of the table,"
                                f" rest of the wall boundaries: Y: "
                                f"{round(wall[1] * 100, 1)}CM to {round(wall[2] * 100, 1)}CM, "
                                f"Z: {round(wall[3] * 100, 1)} to {round(wall[4] * 100, 1)}CM.")
                # Experiment.log(Experiment.LogType.INFO, log_to_write)
                self.wall_x_const(0, left_table_top / 100.0, 0, wall_height,
                                  (self.arm_base_location[LocationType.LEFT][0] + self.arm_base_location[LocationType.RIGHT][0]) / 2.0, self.obstacles_non_np)

    def set_active_arm(self, active_arm):
        self.active_arm = active_arm

    def get_other_arm(self):
        return LocationType.LEFT if self.active_arm is LocationType.RIGHT else LocationType.RIGHT

    def update_obstacles(self, cubes, static_arm_conf):
        self.radius = 0.025
        # cubes
        all_obstacles = [*cubes, *self.get_static_arm_spheres(self.arm_transforms[self.get_other_arm()], static_arm_conf), *self.obstacles_non_np]
        self.obstacles = np.array(all_obstacles)

    def get_static_arm_spheres(self, static_arm_transform, static_arm_conf):
        global_sphere_coords = static_arm_transform.conf2sphere_coords(static_arm_conf)
        global_sphere_coords_list = global_sphere_coords.keys()
        spheres = []
        for link in global_sphere_coords_list:
            link_spheres = global_sphere_coords[link]
            for sphere in link_spheres:
                spheres.append([sphere[0], sphere[1], sphere[2]])
        return spheres

    def add_box(self, x, y, z, dx, dy, dz, skip):
        self.obstacles_non_np = [] # temp
        self.box(x=x, y=y, z=z, dx=dx, dy=dy, dz=dz, obstacles=self.obstacles_non_np, skip=skip)

    def sphere_num(self, min_coord, max_cord):
        '''
        Return the number of spheres based on the distance
        '''
        return int(np.ceil(abs(max_cord-min_coord) / (self.radius*2))+2)
    
    def wall_y_const(self, x_min, x_max, z_min, z_max, y_const, obstacles):
        '''
        Constructs a wall with constant y coord value
        '''
        num_x = self.sphere_num(x_min, x_max)
        num_z = self.sphere_num(z_min, z_max)
        for x in list(np.linspace(x_min, x_max,  num= num_x, endpoint=True)):
                for z in list(np.linspace(z_min, z_max, num= num_z, endpoint=True)):
                    obstacles.append([x, y_const, z])
    
    def wall_x_const(self, y_min, y_max, z_min, z_max, x_const, obstacles):
        '''
        Constructs a wall with constant x coord value
        '''
        num_y = self.sphere_num(y_min, y_max)
        num_z = self.sphere_num(z_min, z_max)
        for y in list(np.linspace(y_min, y_max,  num= num_y, endpoint=True)):
                for z in list(np.linspace(z_min, z_max , num= num_z, endpoint=True)):
                    obstacles.append([x_const, y, z])
    
    def wall_z_const(self, x_min, x_max, y_min, y_max, z_const, obstacles):
        '''
        Constructs a wall with constant z coord value
        '''
        num_y = self.sphere_num(y_min, y_max)
        num_x = self.sphere_num(x_min, x_max)
        for y in list(np.linspace(y_min, y_max,  num= num_y, endpoint=True)):
                for x in list(np.linspace(x_min, x_max , num= num_x, endpoint=True)):
                    obstacles.append([x, y, z_const])
    
    def box(self, x, y, z, dx, dy, dz, obstacles, skip =[]):
        '''
        Constructs a Box
        '''
        if '-x' not in skip:
            self.wall_x_const(y-dy/2, y+dy/2, z-dz/2, z+dz/2, x-dx/2, obstacles)
        if 'x' not in skip:
            self.wall_x_const(y-dy/2, y+dy/2, z-dz/2, z+dz/2, x+dx/2, obstacles)
        if '-y' not in skip:
            self.wall_y_const(x-dx/2, x+dx/2, z-dz/2, z+dz/2, y-dy/2, obstacles)
        if 'y' not in skip:
            self.wall_y_const(x-dx/2, x+dx/2, z-dz/2, z+dz/2, y+dy/2, obstacles)
        if '-z' not in skip:
            self.wall_z_const(x-dx/2, x+dx/2, y-dy/2, y+dy/2, z-dz/2, obstacles)
        if 'z' not in skip:
            self.wall_z_const(x-dx/2, x+dx/2, y-dy/2, y+dy/2, z+dz/2, obstacles)