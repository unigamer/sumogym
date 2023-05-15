import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import copy
import pprint

import gymnasium as gym
from gymnasium import spaces

from .. import vehicle


class RobotSumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "robotA_angular_position": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                "robotA_angular_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "robotA_linear_position": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "robotA_linear_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "robotA_wheel_velocities": spaces.Box(low=-30.0, high=30.0, shape=(2,), dtype=np.float64),

                "robotB_angular_position": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                "robotB_angular_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "robotB_linear_position": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "robotB_linear_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "robotB_wheel_velocities": spaces.Box(low=-30.0, high=30.0, shape=(2,), dtype=np.float64),

                "robots_colliding": spaces.MultiBinary(1)
            }
        )

        # Action Space
        self.action_space = spaces.Dict(
            {
                "robotA": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64),
                "robotB": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Pybullet Simulation
        simulation_frequency = 240  # Hz
        self.timestep = 1/simulation_frequency
        if self.render_mode == "human":
            connection_mode = pybullet.GUI
        else:
            connection_mode = pybullet.DIRECT
        self.p = bc.BulletClient(connection_mode=connection_mode)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = self.p.loadURDF("plane.urdf")
        self.dojo = self.p.loadURDF(str(vehicle.urdf_folder_path/"dojo.urdf"),
                                    useFixedBase=True, basePosition=[0, 0, 0.25])
        # x = self.p.loadTexture(str(vehicle.urdf_folder_path / "meow.png"))
        # self.p.changeVisualShape(objectUniqueId=self.dojo,
        #                     linkIndex=-1, textureUniqueId=x)
        self.p.setGravity(0, 0, -9.8)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)  # Remove the GUI

        # Load robot and setup pybullet simulation
        wheel_torque_limit = 2
        self.robotA = vehicle.Robot(p=self.p,
                                    wheel_torque_limit=wheel_torque_limit)
        self.robotB = vehicle.Robot(p=self.p,
                                    wheel_torque_limit=wheel_torque_limit)
        self.robots = [self.robotA, self.robotB]
        self.robot_names = ["robotA", "robotB"]

    def _get_obs(self):

        robots_colliding = bool(self.p.getContactPoints(self.robotA(), self.robotB()))
        observation_dict = {}
        for robot, robot_name in zip(self.robots, self.robot_names):
            bodyID = robot()
            linear_position, angular_position = self.p.getBasePositionAndOrientation(bodyID)
            linear_velocity, angular_velocity = self.p.getBaseVelocity(bodyID)
            observation_dict[f"{robot_name}_linear_position"] = np.array(linear_position)
            observation_dict[f"{robot_name}_angular_position"] = np.array(angular_position)
            observation_dict[f"{robot_name}_linear_velocity"] = np.array(linear_velocity)
            observation_dict[f"{robot_name}_angular_velocity"] = np.array(angular_velocity)
            observation_dict[f"{robot_name}_wheel_velocities"] = robot.getState()[
                "wheel_velocities"]
        observation_dict["robots_colliding"] = np.array([robots_colliding], dtype=np.int8)
        # pprint.pprint(observation_dict)
        return observation_dict

    def _get_info(self):
        floor_collisions = [None]*0
        for robot in self.robots:
            bodyID = robot()
            floor_collision = bool(self.p.getContactPoints(bodyID, self.plane))
            floor_collisions.append(floor_collision)
        return {"robotA_floor": floor_collisions[0], "robotB_floor": floor_collisions[1]}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # We need the following line to seed self.np_random

        polarAngle = np.random.uniform(low=np.deg2rad(0), high=np.deg2rad(360), size=(1))[0]
        yawAngleOffset = np.random.uniform(low=np.deg2rad(0), high=np.deg2rad(90), size=(1))[0]
        radius = np.random.uniform(low=0.5, high=1.3, size=(1))[0]
        for ii, robot in enumerate(self.robots):
            polarAngleRobot = polarAngle + ii*np.deg2rad(180)
            x = radius*np.sin(polarAngleRobot)
            y = radius*np.cos(polarAngleRobot)
            quaternion = self.p.getQuaternionFromEuler(
                [0, 0, np.deg2rad(270)-polarAngleRobot+yawAngleOffset])
            position = [x, y, 0.8]
            self.p.resetBasePositionAndOrientation(robot(), position, quaternion)

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):

        # Apply action
        self.robotA.setState(action["robotA"])
        self.robotB.setState(action["robotB"])

        # Step simulation
        self.p.stepSimulation()

        # Get observation
        observation = self._get_obs()

        # Get info
        info = self._get_info()

        # Termination conditions
        if info["robotA_floor"] or info["robotB_floor"]:
            terminated = True
        else:
            terminated = False

        # Calculate reward
        reward = 0  # Construct your own reward for each agent. It's a zero sum game.

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        return None

    def _render(self):
        return None

    def close(self):
        return None
