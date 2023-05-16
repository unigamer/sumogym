import functools

import numpy as np
import gymnasium
from gymnasium import spaces

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from .. import vehicle
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import copy
import pprint

from gymnasium.utils import seeding


class RobotSumoParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "robotsumo_v0"}

    def __init__(self, render_mode=None):

        self.possible_agents = ["robotA", "robotB"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
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

        self._seed()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observation_space = spaces.Dict(
            {
                "self_angular_position": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                "self_angular_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "self_linear_position": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "self_linear_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "self_wheel_velocities": spaces.Box(low=-30.0, high=30.0, shape=(2,), dtype=np.float64),

                "robots_colliding": spaces.MultiBinary(1),

                "opponent_angular_position": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float64),
                "opponent_angular_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "opponent_linear_position": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "opponent_linear_velocity": spaces.Box(low=-10, high=10.0, shape=(3,), dtype=np.float64),
                "opponent_wheel_velocities": spaces.Box(low=-30.0, high=30.0, shape=(2,), dtype=np.float64),

                
            })

        return observation_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

    def render(self):
        pass

    def close(self):
        pass

    def _get_obs(self, agent):

        assert agent in self.possible_agents
        if agent == "robotA":
            robot_order = [self.robotA, self.robotB]
        else:
            robot_order = [self.robotB, self.robotA]


        robots_colliding = bool(self.p.getContactPoints(self.robotA(), self.robotB()))
        observation_dict = {}
        for robot, robot_name in zip(robot_order, ["self", "opponent"]):
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

    def _get_info(self, agent):
        assert agent in self.possible_agents
        if agent == "robotA":
            robot_order = [self.robotA, self.robotB]
        else:
            robot_order = [self.robotB, self.robotA]
        floor_collisions = [None]*0
        for robot in robot_order:
            bodyID = robot()
            floor_collision = bool(self.p.getContactPoints(bodyID, self.plane))
            floor_collisions.append(floor_collision)
        return {"self_floor_collision": floor_collisions[0], "opponent_floor_collision": floor_collisions[1]}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)


    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
       
        self.agents = self.possible_agents[:]

        polarAngle = self.np_random.uniform(low=np.deg2rad(0), high=np.deg2rad(360), size=(1))[0]
        yawAngleOffset = self.np_random.uniform(low=np.deg2rad(0), high=np.deg2rad(90), size=(1))[0]
        radius = self.np_random.uniform(low=0.5, high=1.3, size=(1))[0]
        for ii, robot in enumerate(self.robots):
            polarAngleRobot = polarAngle + ii*np.deg2rad(180)
            x = radius*np.sin(polarAngleRobot)
            y = radius*np.cos(polarAngleRobot)
            quaternion = self.p.getQuaternionFromEuler(
                [0, 0, np.deg2rad(270)-polarAngleRobot+yawAngleOffset])
            position = [x, y, 0.8]
            self.p.resetBasePositionAndOrientation(robot(), position, quaternion)

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        return observations, infos
    

    def step(self, actions):

        assert list(actions.keys()) == ["robotA", "robotB"]

        # Apply actions
        self.robotA.setState(actions["robotA"])
        self.robotB.setState(actions["robotB"])

        # Step simulation
        self.p.stepSimulation()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent) for agent in self.agents}

        time_penalty = -0.01
        floor_reward_penalty = 50
        contact_award = 0.1
        rewards = {}

        for agent in self.agents:
            reward = time_penalty - int(infos[agent]["self_floor_collision"])*floor_reward_penalty \
                        + int(infos[agent]["opponent_floor_collision"])*floor_reward_penalty \
                        + observations[agent]["robots_colliding"]*contact_award
            rewards[agent] =reward

        terminations = {"robotA": infos["robotA"]["self_floor_collision"], 
                        "robotB": infos["robotB"]["self_floor_collision"]}

        truncations = {"robotA": False, "robotB": False}

        
        return observations, rewards, terminations, truncations, infos