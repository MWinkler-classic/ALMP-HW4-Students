from enum import Enum

import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from building_blocks import BuildingBlocks3D
from visualizer import Visualize_UR
import numpy as np
from numpy import linalg
from math import pi, cos, sin, atan2, acos, sqrt, asin

from environment import LocationType
# tool_length = 0.135 # [m]
tool_length = 0.145 # [m]

DH_matrix_UR3e = np.array([[0, pi / 2.0, 0.15185],
                            [-0.24355, 0, 0],
                            [-0.2132, 0, 0],
                            [0, pi / 2.0, 0.13105],
                            [0, -pi / 2.0, 0.08535],
                            [0, 0, 0.0921]])

DH_matrix_UR5e = np.array([[0, pi / 2.0, 0.1625],
                            [-0.425, 0, 0],
                            [-0.3922, 0, 0],
                            [0, pi / 2.0, 0.1333],
                            [0, -pi / 2.0, 0.0997],
                            [0, 0, 0.0996 + tool_length]])

DH_matrix_UR10e = np.array([[0, pi / 2.0, 0.1807],
                             [-0.6127, 0, 0],
                             [-0.57155, 0, 0],
                             [0, pi / 2.0, 0.17415],
                             [0, -pi / 2.0, 0.11985],
                             [0, 0, 0.11655]])

DH_matrix_UR16e = np.array([[0, pi / 2.0, 0.1807],
                             [-0.4784, 0, 0],
                             [-0.36, 0, 0],
                             [0, pi / 2.0, 0.17415],
                             [0, -pi / 2.0, 0.11985],
                             [0, 0, 0.11655]])

DH_matrix_UR3 = np.array([[0, pi / 2.0, 0.1519],
                           [-0.24365, 0, 0],
                           [-0.21325, 0, 0],
                           [0, pi / 2.0, 0.11235],
                           [0, -pi / 2.0, 0.08535],
                           [0, 0, 0.0819]])

DH_matrix_UR5 = np.array([[0, pi / 2.0, 0.089159],
                           [-0.425, 0, 0],
                           [-0.39225, 0, 0],
                           [0, pi / 2.0, 0.10915],
                           [0, -pi / 2.0, 0.09465],
                           [0, 0, 0.0823]])

DH_matrix_UR10 = np.array([[0, pi / 2.0, 0.1273],
                            [-0.612, 0, 0],
                            [-0.5723, 0, 0],
                            [0, pi / 2.0, 0.163941],
                            [0, -pi / 2.0, 0.1157],
                            [0, 0, 0.0922]])


def mat_transform_DH(DH_matrix, n, edges=np.zeros(6)):
    n = n - 1
    t_z_theta = np.array([[cos(edges[n]), -sin(edges[n]), 0, 0],
                           [sin(edges[n]), cos(edges[n]), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    t_zd = np.identity(4)
    t_zd[2, 3] = DH_matrix[n, 2]
    t_xa = np.identity(4)
    t_xa[0, 3] = DH_matrix[n, 0]
    t_x_alpha = np.array([[1, 0, 0, 0],
                           [0, cos(DH_matrix[n, 1]), -sin(DH_matrix[n, 1]), 0],
                           [0, sin(DH_matrix[n, 1]), cos(DH_matrix[n, 1]), 0],
                           [0, 0, 0, 1]])
    transform = t_z_theta @ t_zd @ t_xa @ t_x_alpha
    return transform


def forward_kinematic_solution(DH_matrix, edges=np.zeros(6)):
    """
    Calculates the forward kinematic solution for a given DH_matrix and joint angles.

    Parameters:
        DH_matrix (numpy.matrix): The Denavit-Hartenberg parameter matrix.
        edges (numpy.matrix): A 6x1 matrix of joint angles (theta_1 to theta_6).

    Returns:
        numpy.matrix: The 4x4 transformation matrix from the base frame to the end-effector frame.
    """

    t01 = mat_transform_DH(DH_matrix, 1, edges)
    t12 = mat_transform_DH(DH_matrix, 2, edges)
    t23 = mat_transform_DH(DH_matrix, 3, edges)
    t34 = mat_transform_DH(DH_matrix, 4, edges)
    t45 = mat_transform_DH(DH_matrix, 5, edges)
    t56 = mat_transform_DH(DH_matrix, 6, edges)
    answer = t01 @ t12 @ t23 @ t34 @ t45 @ t56
    return answer


def inverse_kinematic_solution(DH_matrix, transform_matrix):
    """
    Calculates the inverse kinematic solution for a given DH_matrix and end-effector transform.

    Parameters:
        DH_matrix (numpy.matrix): The Denavit-Hartenberg parameter matrix.
        transform_matrix (numpy.matrix): The 4x4 transformation matrix from the base frame to the end-effector frame.

    Returns:
        numpy.array: A 8x6 matrix containing up to 8 possible joint angle solutions (theta_1 to theta_6).
                      Each row represents a different solution.
    """

    theta = np.zeros((6, 8))
    # theta 1
    T06 = transform_matrix

    P05 = T06 @ np.array([0, 0, -DH_matrix[5, 2], 1])
    psi = atan2(P05[1], P05[0])
    phi_val = (DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2]) / sqrt(P05[0] ** 2 + P05[1] ** 2)
    phi = acos(np.clip(phi_val, -1, 1))
    theta[0, 0:4] = psi + phi + pi / 2
    theta[0, 4:8] = psi - phi + pi / 2

    # theta 5
    for i in {0, 4}:
            th5cos = (T06[0, 3] * sin(theta[0, i]) - T06[1, 3] * cos(theta[0, i]) - (
                    DH_matrix[1, 2] + DH_matrix[3, 2] + DH_matrix[2, 2])) / DH_matrix[5, 2]
            th5 = acos(np.clip(th5cos, -1, 1))
            theta[4, i:i + 2] = th5
            theta[4, i + 2:i + 4] = -th5
    # theta 6
    for i in {0, 2, 4, 6}:
        # if sin(theta[4, i]) == 0:
        #     theta[5, i:i + 1] = 0 # any angle
        #     break
        T60 = linalg.inv(T06)
        th = atan2((-T60[1, 0] * sin(theta[0, i]) + T60[1, 1] * cos(theta[0, i])),
                   (T60[0, 0] * sin(theta[0, i]) - T60[0, 1] * cos(theta[0, i])))
        theta[5, i:i + 2] = th

    # theta 3
    for i in {0, 2, 4, 6}:
        T01 = mat_transform_DH(DH_matrix, 1, theta[:, i])
        T45 = mat_transform_DH(DH_matrix, 5, theta[:, i])
        T56 = mat_transform_DH(DH_matrix, 6, theta[:, i])
        T14 = linalg.inv(T01) @ T06 @ linalg.inv(T45 @ T56)
        P13 = T14 @ np.array([0, -DH_matrix[3, 2], 0, 1])
        costh3 = ((P13[0] ** 2 + P13[1] ** 2 - DH_matrix[1, 0] ** 2 - DH_matrix[2, 0] ** 2) /
                  (2 * DH_matrix[1, 0] * DH_matrix[2, 0]))
        th3 = acos(np.clip(costh3, -1, 1))
        theta[2, i] = th3
        theta[2, i + 1] = -th3

    # theta 2,4
    for i in range(8):
        T01 = mat_transform_DH(DH_matrix, 1, theta[:, i])
        T45 = mat_transform_DH(DH_matrix, 5, theta[:, i])
        T56 = mat_transform_DH(DH_matrix, 6, theta[:, i])
        T14 = linalg.inv(T01) @ T06 @ linalg.inv(T45 @ T56)
        P13 = T14 @ np.array([0, -DH_matrix[3, 2], 0, 1])

        asin_val = -DH_matrix[2, 0] * sin(theta[2, i]) / sqrt(P13[0] ** 2 + P13[1] ** 2)
        theta[1, i] = atan2(-P13[1], -P13[0]) - asin(np.clip(asin_val, -1, 1))

        T32 = linalg.inv(mat_transform_DH(DH_matrix, 3, theta[:, i]))
        T21 = linalg.inv(mat_transform_DH(DH_matrix, 2, theta[:, i]))
        T34 = T32 @ T21 @ T14
        theta[3, i] = atan2(T34[1, 0], T34[0, 0])
    
    return theta.T
