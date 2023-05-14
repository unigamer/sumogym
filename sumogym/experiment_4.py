import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np
from simple_pid import PID
from typing import List


from . import utils

from . import vehicle

p = bc.BulletClient(connection_mode=pybullet.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
# dojo = 
p.setGravity(0, 0, -9.8)

# Simulation Parameters
simulation_frequency = 240  # Hz
timestep = 1/simulation_frequency

wheel_torque_limit = 2

# Load robot and setup pybullet simulation
robot = vehicle.Robot(p=p, wheel_torque_limit=wheel_torque_limit)


# Prepare PID controllers
wheel_controllers: List[PID] = [None]*0
for _ in range(len(vehicle.wheel_joints)):
    wheel_controllers.append(PID(10, 0.3, 0.0, setpoint=0, sample_time=timestep,
                             output_limits=[-wheel_torque_limit, wheel_torque_limit]))


runSimulation = True
while runSimulation:

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
