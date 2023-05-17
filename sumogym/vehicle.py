from typing import Any
import pybullet_utils.bullet_client as bc
import pathlib
from . import utils

import numpy as np

wheel_joints = ["left_wheel_joint", "right_wheel_joint"]

# URDF location
urdf_name = "simple_robot_2.urdf"
urdf_folder_path = pathlib.Path(
    __file__).parents[0] / "robot_descriptions/simple_robot"
robot_path = urdf_folder_path / urdf_name


class Robot():
    def __init__(self,
                 p: bc.BulletClient,
                 timestep: float = 240,
                 wheel_torque_limit: float = 2.0,
                 basePosition: np.ndarray = [0, 0, 0.3],
                 color: np.ndarray = [1,1,1,1]) -> None:

        self.p = p  # The pybullet simulation we're connected to
        self.timestep = timestep
        self.wheel_torque_limit = wheel_torque_limit

        self.robot = self.p.loadURDF(str(robot_path), basePosition=basePosition)

        # Prepare joints
        self.joint_dictionary = {}  # Key: joint name, Value: joint index
        for ii in range(self.p.getNumJoints(self.robot)):
            jointInfo = self.p.getJointInfo(self.robot, ii)
            self.joint_dictionary[jointInfo[1].decode("utf-8")] = jointInfo[0]
            print(utils.get_joint_info_dict(p, self.robot, ii))
        print(self.joint_dictionary)
        for joint in wheel_joints:  # Disable built in velocity controller
            p.setJointMotorControl2(
                self.robot, self.joint_dictionary[joint], p.VELOCITY_CONTROL, force=0)

        x = self.p.loadTexture(str(urdf_folder_path / "meow.png"))
        p.changeVisualShape(objectUniqueId=self.robot,
                            linkIndex=0, textureUniqueId=x)
        p.changeVisualShape(objectUniqueId=self.robot,
                            linkIndex=1, textureUniqueId=x)
        self.p.changeVisualShape(objectUniqueId=self.robot, linkIndex=-1, rgbaColor=color)

    def getState(self) -> dict:
        state = {}
        joint_states = self.p.getJointStates(
            self.robot, list(self.joint_dictionary.values()))
        state["wheel_velocities"] = np.array(
            [joint_states[0][1], joint_states[1][1]])
        return state

    def setState(self, input: np.ndarray):
        self.p.setJointMotorControlArray(
            self.robot, list(self.joint_dictionary.values()), self.p.TORQUE_CONTROL, forces=input)

    def __call__(self) -> int:
        return self.robot
