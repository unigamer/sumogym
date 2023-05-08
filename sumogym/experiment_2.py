import pathlib
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np

import utils

p = bc.BulletClient(connection_mode=pybullet.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

# Simulation Parameters
simulation_frequency = 240  # Hz
timestep = 1/simulation_frequency

# URDF location
robot_path = pathlib.Path(
    __file__).parents[0] / "robot_descriptions/simple_robot/simple_robot_2.urdf"

wheel_joints = ["left_wheel_joint", "right_wheel_joint"]
wheel_torque_limit = 10

# Load robot and setup pybullet simulation
robot = p.loadURDF(str(robot_path), basePosition=[0, 0, 0.3])
p.setGravity(0, 0, -9.8)

# Prepare joints
joint_dictionary = {} # Key: joint name, Value: joint index
for ii in range(p.getNumJoints(robot)):
    jointInfo = p.getJointInfo(robot, ii)
    joint_dictionary[jointInfo[1].decode("utf-8")] = jointInfo[0]
    print(utils.get_joint_info_dict(p,robot, ii))
print(joint_dictionary)
for joint in wheel_joints: # Disable built in velocity controller
    p.setJointMotorControl2(robot, joint_dictionary[joint], p.VELOCITY_CONTROL, force=0)



integral_error = 0.0
prev_error = 0.0

runSimulation = True
while runSimulation:

    desired_velocity = 0.1  # rad/s

    joint_index = 1
    # Calculate the torque required to achieve the desired velocity
    joint_state = p.getJointState(robot, joint_index)
    current_velocity = joint_state[1]

    # print(joint_state)

    error = desired_velocity - current_velocity

    kp = 0.5
    ki = 0.1
    proportional_error = kp * error
    integral_error += ki * error

    # Apply the proportional and integral components to the control signal
    control_signal = proportional_error  + integral_error

    torque = control_signal

    torque = np.clip(torque, -wheel_torque_limit, wheel_torque_limit)

    # print(torque)

    # Apply the desired torque to the joint
    p.setJointMotorControl2(robot, joint_index, p.TORQUE_CONTROL, force=torque)
    # p.setJointMotorControl2(robot, joint_index, p.VELOCITY_CONTROL, targetVelocity = 0.2)

    # Step the s

    # joint_state = p.getJointState(robot, 1)
    # # Calculate the torque based on the joint position and velocity
    # position = joint_state[0]
    # velocity = joint_state[1]
    # torque = -0.1 * position - 0.01 * velocity
    # p.setJointMotorControl2(robot, 1, p.TORQUE_CONTROL, force = torque)

    # p.setJointMotorControl2(robot, joint_dictionary["joint_right_wheel"], p.TORQUE_CONTROL, force = 40)
    # jointStates = p.getJointStates(robot, [joint_dictionary["joint_left_wheel"], joint_dictionary["joint_right_wheel"]])
    # print(jointStates)
    # print(jointStates[joint_dictionary["joint_left_wheel"]][1])
    p.stepSimulation()
    # print(p.getJointStates(robot, [1])[0][1])
    time.sleep(timestep)
