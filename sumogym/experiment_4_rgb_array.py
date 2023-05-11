from . import vehicle
from . import utils
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np
from simple_pid import PID
from typing import List

import cv2


p = bc.BulletClient(connection_mode=pybullet.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

# Simulation Parameters
simulation_frequency = 240  # Hz
timestep = 1/simulation_frequency


# Load robot and setup pybullet simulation
wheel_torque_limit = 2
robot = vehicle.Robot(p=p, wheel_torque_limit=wheel_torque_limit)


# Prepare PID controllers
wheel_controllers: List[PID] = [None]*0
for _ in range(len(vehicle.wheel_joints)):
    wheel_controllers.append(PID(10, 0.3, 0.0, setpoint=0, sample_time=timestep,
                             output_limits=[-wheel_torque_limit, wheel_torque_limit]))


# RGB Array Stuff
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
camTargetPos = [0, 0, 0]
cam_dist = 5.20
cam_pitch = -60
cam_yaw = 1.2
render_width = 1001
render_height = 1000
view_matrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPosition=camTargetPos,
    distance=cam_dist,
    yaw=cam_yaw,
    pitch=cam_pitch,
    roll=0,
    upAxisIndex=2)

proj_matrix = p.computeProjectionMatrixFOV(
    fov=60,
    aspect=float(render_width)/render_height,
    nearVal=0.1,
    farVal=100.0)


runSimulation = True
counter = 0
renderFPS = 30
while runSimulation:
    counter += 1

    robot_state = robot.getState()
    desired_wheel_velocities = np.array([2, 2])
    current_wheel_velocities = robot_state["wheel_velocities"]
    torques = np.zeros((2))

    for ii, (wheel_controller, current_wheel_velocity, desired_wheel_velocity) in \
            enumerate(zip(wheel_controllers, current_wheel_velocities, desired_wheel_velocities)):
        wheel_controller.setpoint = desired_wheel_velocity
        torques[ii] = wheel_controller(current_wheel_velocity)

    robot.setState(torques)

    p.stepSimulation()

    time.sleep(timestep)

    if counter % renderFPS == 0:
        (_, _, px, _, _) = p.getCameraImage(
            width=render_width,
            height=render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        counter = 0
        rgb_array = np.array(px).reshape((render_height, render_width, 4))
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", rgb_array)
        cv2.waitKey(1)
