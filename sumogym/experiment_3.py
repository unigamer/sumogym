import pathlib
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np
from simple_pid import PID
from typing import List

from PIL import Image
from . import utils

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
wheel_torque_limit = 2

# Load robot and setup pybullet simulation
robot = p.loadURDF(str(robot_path), basePosition=[0, 0, 0.3])
p.setGravity(0, 0, -9.8)

# Prepare joints
joint_dictionary = {}  # Key: joint name, Value: joint index
for ii in range(p.getNumJoints(robot)):
    jointInfo = p.getJointInfo(robot, ii)
    joint_dictionary[jointInfo[1].decode("utf-8")] = jointInfo[0]
    print(utils.get_joint_info_dict(p, robot, ii))
print(joint_dictionary)
for joint in wheel_joints:  # Disable built in velocity controller
    p.setJointMotorControl2(
        robot, joint_dictionary[joint], p.VELOCITY_CONTROL, force=0)

# Prepare PID controllers
wheel_controllers: List[PID] = [None]*0
for _ in range(len(wheel_joints)):
    wheel_controllers.append(PID(1, 0.3, 0.0, setpoint=0, sample_time=timestep,
                             output_limits=[-wheel_torque_limit, wheel_torque_limit]))



# img = Image.open("/mnt/sde/src/rl/sumo/sumogym/sumogym/robot_descriptions/simple_robot/meow.png")
# img_data = img.tobytes()
# texture_id = pybullet.createTexture(img.width, img.height, img_data)
# material_id = pybullet.createMaterial()
x = p.loadTexture('/mnt/sde/src/rl/sumo/sumogym/sumogym/robot_descriptions/simple_robot/meow.png')
p.changeVisualShape(objectUniqueId=robot, linkIndex=0, textureUniqueId=x)
p.changeVisualShape(objectUniqueId=robot, linkIndex=1, textureUniqueId=x)



runSimulation = True
while runSimulation:

    for wheel_controller, joint in zip(wheel_controllers, wheel_joints):
        joint_index = joint_dictionary[joint]
        if joint == wheel_joints[0]:
            wheel_controller.setpoint = 5  # Get this input from somewhere
        else:
            wheel_controller.setpoint = 5
        current_velocity = p.getJointState(robot, joint_index)[1]
        torque = wheel_controller(current_velocity)
        torque = np.clip(torque, -wheel_torque_limit, wheel_torque_limit)
        print(torque)
        # torque = wheel_torque_limit

        p.setJointMotorControl2(
            robot, joint_index, p.TORQUE_CONTROL, force=torque)
        
        # if joint == wheel_joints[0]:
        #     print(current_velocity)

    p.stepSimulation()

    time.sleep(timestep)
