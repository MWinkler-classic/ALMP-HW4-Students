import os
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import glob
import re
import imageio.v3 as iio
import json
from environment import LocationType
from kinematics import Transform


class Visualize_UR(object):
    def __init__(self, ur_params, env, transform_right_arm:Transform, transform_left_arm:Transform):
        self.plt_i = 1
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.colors = ur_params.ur_links_color
        self.sphere_radius = ur_params.sphere_radius
        self.end_effector_pos = np.array([0, 0, 0, 1])
        self.env = env
        self.transform = transform_right_arm
        self.transform_right_arm = transform_right_arm
        self.transform_left_arm = transform_left_arm
        self.drawn_items = []
        # self.ax.view_init(elev=20, azim=220)  # Set elevation and azimuth angles
        # plt.ioff()
        plt.show(block=False)

    def plot_links(self, end_efctors):
        for link_edge in end_efctors:
            self.ax.scatter(link_edge[0], link_edge[1], link_edge[2])
        for i in range(len(end_efctors) - 1):
            self.ax.plot([end_efctors[i][0], end_efctors[i + 1][0]], [end_efctors[i][1], end_efctors[i + 1][1]],
                     [end_efctors[i][2], end_efctors[i + 1][2]])

    def show(self, end_effctors=None):
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        max_val = max(self.env.bbox[0][0], self.env.bbox[1][0], self.env.bbox[0][1], self.env.bbox[1][1])
        self.ax.set_xlim3d([0, max_val])
        self.ax.set_ylim3d([0, max_val])
        self.ax.set_zlim3d([0, max_val])
        self.ax.view_init(elev=15, azim=250)  # Set elevation and azimuth angles
        self.ax.plot([0, 0.5], [0, 0], [0, 0], c='red')
        self.ax.plot([0, 0], [0, 0.5], [0, 0], c='green')
        self.ax.plot([0, 0], [0, 0], [0, 0.5], c='blue')
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()


    def draw_cubes(self, cubes):
        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        for sphere in cubes:
            x = np.cos(u) * np.sin(v) * self.env.radius + sphere[0]
            y = np.sin(u) * np.sin(v) * self.env.radius + sphere[1]
            z = np.cos(v) * self.env.radius + sphere[2]
            self.ax.plot_surface(x, y, z, color='red', alpha=0.5)

    def draw_obstacles(self):
        '''
        Draws the spheres constructing the obstacles
        '''
        for wall in self.env.walls:
            static_axis = wall[5]
            if static_axis == 0:
                x = [wall[0], wall[0], wall[0], wall[0]]
                y = [wall[1], wall[2], wall[2], wall[1]]
                z = [wall[3], wall[3], wall[4], wall[4]]
                vertices = [list(zip(x, y, z))]
                poly = Poly3DCollection(vertices, facecolors='blue', alpha=0.5, edgecolors='black')
                self.ax.add_collection3d(poly)

    def draw_spheres(self, global_sphere_coords, track_end_effector=False):
        '''
        Draws the spheres constructing the manipulator
        '''
        u, v = np.mgrid[0:2 * np.pi:10j, 0:np.pi:10j]
        for frame in global_sphere_coords.keys():
            for sphere_coords in global_sphere_coords[frame]:
                x = np.cos(u) * np.sin(v) * self.sphere_radius[frame] + sphere_coords[0]
                y = np.sin(u) * np.sin(v) * self.sphere_radius[frame] + sphere_coords[1]
                z = np.cos(v) * self.sphere_radius[frame] + sphere_coords[2]
                self.ax.plot_surface(x, y, z, color=self.colors[frame], alpha=0.3)
        if track_end_effector:
            self.end_effector_pos = np.vstack((self.end_effector_pos, global_sphere_coords['wrist_3_link'][-1]))
            self.ax.scatter(self.end_effector_pos[:, 0], self.end_effector_pos[:, 1], self.end_effector_pos[:, 2])

    def show_path(self, path, sleep_time=0.02):
        '''
        Plots the path
        '''
        confs_num = len(path) - 1
        resolution = 5 * np.pi / 180
        j = 0
        for i in range(confs_num):
            max_diff = np.max(np.abs(path[i + 1] - path[i]))
            nums = max(int(max_diff / resolution), 2)
            if i + 1 == confs_num:
                confs_to_plot = np.linspace(start=path[i], stop=path[i + 1], num=nums, endpoint=True)
            else:
                confs_to_plot = np.linspace(start=path[i], stop=path[i + 1], num=nums, endpoint=False)
            for conf in confs_to_plot:
                j += 1
                global_sphere_coords = self.transform.conf2sphere_coords(conf)
                self.draw_spheres(global_sphere_coords, track_end_effector=True)
                self.draw_square()
                self.show()
                time.sleep(sleep_time)
                self.ax.axes.clear()

    def show_stage_moving_single_arm(self, arm_id, path, static_conf, cubes, msg="", passing_cube = False, sleep_time=0.02):
        '''
        Plots the path
        '''
        confs_num = len(path) - 1
        resolution = 5 * np.pi / 180
        j = -1
        for i in range(confs_num):
            max_diff = np.max(np.abs(path[i + 1] - path[i]))
            nums = max(int(max_diff / resolution), 2)
            if i + 1 == confs_num:
                confs_to_plot = np.linspace(start=path[i], stop=path[i + 1], num=nums, endpoint=True)
            else:
                confs_to_plot = np.linspace(start=path[i], stop=path[i + 1], num=nums, endpoint=False)
            for conf in confs_to_plot:
                j += 1
                if j == 32:
                    pass
                if arm_id == LocationType.RIGHT:
                    conf_right = conf
                    conf_left = static_conf
                else:
                    conf_right = static_conf
                    conf_left = conf
                global_sphere_coords_right = self.transform_right_arm.conf2sphere_coords(conf_right)
                global_sphere_coords_left = self.transform_left_arm.conf2sphere_coords(conf_left)
                if passing_cube:
                    self.ax.view_init(elev=15, azim=225)  # Set elevation and azimuth angles
                else:
                    self.ax.view_init(elev=15, azim=250)  # Set elevation and azimuth angles
                self.draw_spheres(global_sphere_coords_right)
                self.draw_spheres(global_sphere_coords_left)
                self.draw_square()
                self.ax.set_title(msg)
                self.ax.text(self.env.arm_base_location[LocationType.RIGHT][0]+0.1,
                             self.env.arm_base_location[LocationType.RIGHT][1]+0.1, 0,
                             "Right",
                             (0, 0, 0), color='blue', bbox=dict(facecolor='white', edgecolor='blue', pad=1.0), zorder=10)
                self.ax.text(self.env.arm_base_location[LocationType.LEFT][0]+0.1,
                             self.env.arm_base_location[LocationType.LEFT][1]+0.1, 0,
                             "Left",
                             (0, 0, 0), color='blue', bbox=dict(facecolor='white', edgecolor='blue', pad=1.0), zorder=10)
                self.draw_cubes(cubes)
                self.draw_obstacles()
                self.show()
                self.save_plot()
                time.sleep(sleep_time)
                self.ax.axes.clear()

    def draw_square(self):
        # tables
        for table in [self.env.tables[LocationType.LEFT], self.env.tables[LocationType.RIGHT]]:
            self.ax.plot([table[0][0], table[1][0], table[1][0], table[0][0], table[0][0]],
                         [table[0][1], table[0][1], table[1][1], table[1][1], table[0][1]], color='blue')
        # cube areas
        for area in [self.env.cube_areas[LocationType.LEFT], self.env.cube_areas[LocationType.RIGHT]]:
            self.ax.plot([area[0][0], area[1][0], area[1][0], area[0][0], area[0][0]],
                         [area[0][1], area[0][1], area[1][1], area[1][1], area[0][1]], color='red')
        # arms
        arm_base_radius = 0.1
        for arm_base_location in [self.env.arm_base_location[LocationType.RIGHT], self.env.arm_base_location[LocationType.LEFT]]:
            arm_x = arm_base_location[0]
            arm_y = arm_base_location[1]
            self.ax.plot([arm_x - arm_base_radius, arm_x + arm_base_radius,arm_x + arm_base_radius,
                          arm_x - arm_base_radius, arm_x - arm_base_radius],
                         [arm_y + arm_base_radius, arm_y + arm_base_radius, arm_y - arm_base_radius,
                          arm_y - arm_base_radius, arm_y + arm_base_radius], color='black')

    def show_conf(self, conf: np.array):
        '''
        Plots configuration
        '''
        global_sphere_coords = self.transform.conf2sphere_coords(conf)
        self.draw_spheres(global_sphere_coords)
        self.draw_square()
        plt.ioff()
        self.show()
        plt.show()
        time.sleep(0.1)


    def save_plot(self):
        dir_path = r"./outputs/"
        plt.savefig(dir_path + f'plot{self.plt_i}.png', dpi=300)
        self.plt_i += 1

    def animate_by_pngs(self):
        pattern = r"plot(\d+)\.png$"
        dir_path = r"./outputs/"
        files = glob.glob(os.path.join(dir_path, "*"))
        matched_files = [f for f in files if re.search(pattern, os.path.basename(f))]
        if not matched_files:
            print(f"No plot PNG files found in {dir_path}. Nothing to animate.")
            return
        sorted_files = sorted(matched_files, key=lambda x: self.extract_number(x))
        imgs = [iio.imread(f) for f in sorted_files]
        iio.imwrite(dir_path + "mygif.gif", imgs, extension='.gif', duration=0.15)
        for file in sorted_files:
            try:
                os.remove(file)
                print(f"Removed {file}")
            except OSError as e:
                print(f"Error: {file} - {e}")

    def extract_number(self, filename):
        match = re.search(r"plot(\d+)\.png", filename)
        if match:
            return int(match.group(1))
        else:
            return -1  # Return a default value or handle error cases

    def draw_two_robots(self, conf_right, conf_left, cubes=[], msg=""):
            """
            Plots two robot configurations
            """
            global_sphere_coords_right = self.transform_right_arm.conf2sphere_coords(
                conf_right
            )
            global_sphere_coords_left = self.transform_left_arm.conf2sphere_coords(
                conf_left
            )
            self.ax.view_init(elev=15, azim=250)  # Set elevation and azimuth angles
            self.draw_spheres(global_sphere_coords_right)
            self.draw_spheres(global_sphere_coords_left)
            self.draw_square()
            self.ax.set_title(msg)
            self.ax.text(
                self.env.arm_base_location[LocationType.RIGHT][0] + 0.1,
                self.env.arm_base_location[LocationType.RIGHT][1] + 0.1,
                0,
                "Right",
                (0, 0, 0),
                color="blue",
                bbox=dict(facecolor="white", edgecolor="blue", pad=1.0),
                zorder=10,
            )
            self.ax.text(
                self.env.arm_base_location[LocationType.LEFT][0] + 0.1,
                self.env.arm_base_location[LocationType.LEFT][1] + 0.1,
                0,
                "Left",
                (0, 0, 0),
                color="blue",
                bbox=dict(facecolor="white", edgecolor="blue", pad=1.0),
                zorder=10,
            )
            self.draw_cubes(cubes)
            # self.draw_obstacles()
            self.show()
            plt.show()
    
    def show_all_experiment(self, experiment_json):
        # Opening JSON file
        with open(experiment_json, 'r') as openfile:
            # Reading from json file
            steps = json.load(openfile)
            for step in steps:
                # iterate over the step dicts
                for i in range(len(step["active_id"])):
                    # display the movement
                    if step["command"][i] == "movel":
                        continue
                    path = [np.array(path_conf) for path_conf in step["path"][i]]
                    if step["description"][i] == "passing the cube.":
                        self.show_stage_moving_single_arm(step["active_id"][i], path, np.array(step["static"][i]), step["cubes"][i],
                                                            step["description"][i], True)
                    else:
                        self.show_stage_moving_single_arm(step["active_id"][i], path, np.array(step["static"][i]), step["cubes"][i],
                                                            step["description"][i])