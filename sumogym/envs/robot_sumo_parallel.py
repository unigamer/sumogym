import functools

import numpy as np

from gymnasium import spaces
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc

from .. import vehicle


class RobotSumoParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "robotsumo_v0"}

    def __init__(self, render_mode=None, max_time=-1):
        self.max_time = max_time
        
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
        self.wheel_torque_limit = 2
        self.robotA = vehicle.Robot(p=self.p,
                                    wheel_torque_limit=self.wheel_torque_limit,
                                    color=[1, 0, 0, 1])
        self.robotB = vehicle.Robot(p=self.p,
                                    wheel_torque_limit=self.wheel_torque_limit,
                                    color=[0, 1, 0, 1])
        self.robots = [self.robotA, self.robotB]

        self._seed()

        if self.render_mode == "rgb_array":
            self._setup_render()


    def _setup_render(self):
        camTargetPos = [0, 0, 0]
        cam_dist = 5.20
        cam_pitch = -60
        cam_yaw = 1.2
        self.render_width = 400
        self.render_height = 400
        self.view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=camTargetPos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)

        self.proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.render_width)/self.render_height,
            nearVal=0.1,
            farVal=100.0)
        

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
        # torque = wheel_effort * self.wheel_torque_limit
        # [left_wheel_effort, right_wheel_effort]
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def render(self):
        assert self.render_mode == "rgb_array"
        (_, _, px, _, _) = self.p.getCameraImage(
            width=self.render_width,
            height=self.render_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px).reshape((self.render_height, self.render_width, 4))
        rgb_array = rgb_array[:, :, :3].astype(np.uint8)

        return rgb_array

    def close(self):
        pass

    def _get_robot_order(self,agent):
        assert agent in self.possible_agents
        if agent == "robotA":
            robot_order = [self.robotA, self.robotB]
        else:
            robot_order = [self.robotB, self.robotA]

        return robot_order
    
    def _get_all_obs(self):
        return {agent: self._get_obs(agent) for agent in self.agents}
    
    def _get_all_infos(self):
        return {agent: self._get_info(agent) for agent in self.agents}


    def _get_obs(self, agent):

        robot_order = self._get_robot_order(agent)

        robots_colliding = bool(self.p.getContactPoints(self.robotA(), self.robotB()))
        observation_dict = {}
        # The first robot is self, the second opponent
        for robot, robot_name in zip(robot_order, ["self", "opponent"]):
            bodyID = robot()
            linear_position, angular_position = self.p.getBasePositionAndOrientation(bodyID)
            linear_velocity, angular_velocity = self.p.getBaseVelocity(bodyID)
            wheel_velocities = robot.getState()["wheel_velocities"]
            observation_dict[f"{robot_name}_linear_position"] = np.array(linear_position)
            observation_dict[f"{robot_name}_angular_position"] = np.array(angular_position)
            observation_dict[f"{robot_name}_linear_velocity"] = np.array(linear_velocity)
            observation_dict[f"{robot_name}_angular_velocity"] = np.array(angular_velocity)
            observation_dict[f"{robot_name}_wheel_velocities"] = wheel_velocities
        observation_dict["robots_colliding"] = np.array([robots_colliding], dtype=np.int8)
        # pprint.pprint(observation_dict)
        return observation_dict

    def _get_info(self, agent):
        robot_order = self._get_robot_order(agent)
        
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

        self.current_time = 0
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

        observations = self._get_all_obs()
        infos = self._get_all_infos()

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

        time_penalty = 0
        floor_reward_penalty = 50
        contact_award = 0.5
        rewards = {}

        for agent in self.agents:
            reward = time_penalty - int(infos[agent]["self_floor_collision"])*floor_reward_penalty \
                + int(infos[agent]["opponent_floor_collision"])*floor_reward_penalty \
                + observations[agent]["robots_colliding"]*contact_award
            rewards[agent] = float(reward)

        terminations = {"robotA": infos["robotA"]["self_floor_collision"],
                        "robotB": infos["robotB"]["self_floor_collision"]}

        truncations = {"robotA": False, "robotB": False}

        self.current_time += self.timestep

        if self.max_time > 0: # If there is a time limit
            if self.current_time >= self.max_time:
                 truncations = {"robotA": True, "robotB": True}
                 terminations = {"robotA": True, "robotB": True}

        return observations, rewards, terminations, truncations, infos
