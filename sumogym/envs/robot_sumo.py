import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .. import vehicle


class RobotSumoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "robotA": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float64),
                "robotB": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float64),
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
        self.dojo = self.p.loadURDF(str(vehicle.urdf_folder_path/"dojo.urdf"),useFixedBase=True,basePosition = [0,0,0.25])
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

    def _get_obs(self):
        return {"robotA": self.robotA.getState()["wheel_velocities"], "robotB": self.robotB.getState()["wheel_velocities"]}

    def _get_info(self):
        return {
            "key1": "value1"
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # We need the following line to seed self.np_random

        polarAngle = np.random.uniform(low=np.deg2rad(0), high=np.deg2rad(360), size=(1))[0]
        yawAngleOffset = np.random.uniform(low=np.deg2rad(0), high=np.deg2rad(90), size=(1))[0]
        radius = np.random.uniform(low=1, high=1.5, size=(1))[0]
        for ii, robot in enumerate(self.robots):
            polarAngleRobot = polarAngle+ ii*np.deg2rad(180)
            x = radius*np.sin(polarAngleRobot)
            y = radius*np.cos(polarAngleRobot)
            quaternion = self.p.getQuaternionFromEuler([0, 0, np.deg2rad(270)-polarAngleRobot+yawAngleOffset])
            position = [x, y, 0.8]
            self.p.resetBasePositionAndOrientation(robot(), position, quaternion)


        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):

        terminated = False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        self.robotA.setState(action["robotA"])
        self.robotB.setState(action["robotB"])
        self.p.stepSimulation()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        return None

    def _render(self):
        return None

    def close(self):
        return None
