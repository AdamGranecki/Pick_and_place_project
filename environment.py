import time
import math
import random
import cv2
import numpy as np
import pybullet as p
import pybullet_data

from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class Environment:

    SIMULATION_STEP_DELAY = 1 / 100000000000

    def __init__(self, robot, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)

        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setRealTimeSimulation(1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.planeID = p.loadURDF("plane.urdf")

        # load ycb object
        self.bottle = p.loadURDF("ycb/006_mustard_bottle.urdf", [0, 0, .3])

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # # Create red cylinder
        # cyl_radius = 0.03
        # cyl_height = 0.1
        # cyl_mass = 0.1
        # cyl_pos = [0.0, 0.0, cyl_height / 2]
        #
        # # Wizualna i kolizyjna reprezentacja
        # cyl_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=cyl_radius, length=cyl_height, rgbaColor=[1, 0, 0, 1])
        # cyl_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=cyl_radius, height=cyl_height)
        #
        # # Stwórz cylinder jako ciało sztywne
        # self.cylinder = p.createMultiBody(baseMass=cyl_mass,
        #                                baseCollisionShapeIndex=cyl_collision,
        #                                baseVisualShapeIndex=cyl_visual,
        #                                basePosition=cyl_pos)


    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def step(self, action, control_method='end'):
        """
        action: (x, y, z, roll, pitch, yaw, gripper_opening_length) for End Effector Position Control
                (a1, a2, a3, a4, a5, a6, a7, gripper_opening_length) for Joint Position Control
        control_method:  'end' for end effector position control
                         'joint' for joint position control
        """
        assert control_method in ('joint', 'end')
        self.robot.move_ee(action, control_method)
        for _ in range(12):  # Wait for a few steps
            self.step_simulation()

        return self.camera.shot_rgbd(self.robot), action


    def reset_object(self, obj):
        x = np.random.uniform(-0.15, 0.35)
        y = np.random.uniform(-0.15, 0.3)
        z = np.random.uniform(0, 0.4)
        pos = [x, y, z]
        # orn = p.getQuaternionFromEuler([-.71, 0, 0])
        orn = p.getQuaternionFromEuler([-.71, 0, np.random.uniform(-math.pi, math.pi)])
        p.resetBasePositionAndOrientation(obj, pos, orn)

    def get_object_position(self, obj):
        pos, rot = p.getBasePositionAndOrientation(obj)
        euler = p.getEulerFromQuaternion(rot)  # (roll, pitch, yaw)
        return list(pos) + [euler[2]]

    def reset(self):
        # self.robot.reset()
        self.reset_object(self.bottle)

    def close(self):
        p.disconnect(self.physicsClient)

class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up_vector, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up_vector)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def configure_view_from_robot(self, robot):
        ls = p.getLinkState(robot.id, robot.eef_id, computeForwardKinematics=1)
        cam_pos, cam_orn = ls[4], ls[5]

        offset_local = [.1, 0, 0]
        offset_world = p.rotateVector(cam_orn, offset_local)
        cam_pos = [cam_pos[i] + offset_world[i] for i in range(3)]

        tilt_angle = math.radians(5)
        tilt_quat = p.getQuaternionFromAxisAngle([1, 0, 0], tilt_angle)
        _, cam_orn = p.multiplyTransforms([0, 0, 0], tilt_quat, [0, 0, 0], cam_orn)

        rot = p.getMatrixFromQuaternion(cam_orn)
        forward = [rot[0], rot[3], rot[6]]
        up = [rot[2], rot[5], rot[8]]

        # draw the camera coordinate
        p.addUserDebugLine(cam_pos, [cam_pos[i] + 0.1 * rot[i] for i in range(3)], [1, 0, 0], 2)  # X - czerwony
        p.addUserDebugLine(cam_pos, [cam_pos[i] + 0.1 * rot[3 + i] for i in range(3)], [0, 1, 0], 2)  # Y - zielony
        p.addUserDebugLine(cam_pos, [cam_pos[i] + 0.1 * rot[6 + i] for i in range(3)], [0, 0, 1], 2)  # Z - niebieski

        self.view_matrix = p.computeViewMatrix(cameraEyePosition=cam_pos,
                                               cameraTargetPosition=[cam_pos[i] + forward[i] for i in range(3)],
                                               cameraUpVector=up)

        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov,
                                                              aspect=self.width / self.height,
                                                              nearVal=self.near,
                                                              farVal=self.far)

    def shot_rgbd(self, robot):
        self.configure_view_from_robot(robot)
        w, h = self.width, self.height
        img = p.getCameraImage(w, h, self.view_matrix, self.projection_matrix,
                               renderer=p.ER_TINY_RENDERER)
        rgb, depth, seg = img[2], img[3], img[4]

        rgb = rgb/255
        # depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        depth_expanded = depth[..., None]
        # depth_expanded = depth_expanded.astype(np.uint8)[..., None]

        rgbd = np.concatenate([rgb[:, :, :3], depth_expanded], axis=2)
        return rgbd