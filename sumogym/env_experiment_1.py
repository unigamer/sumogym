import gymnasium
# import sumogym
import time
import pprint
import numpy as np

from . import joystick
from . import utils
from simple_pid import PID
from typing import List

env = gymnasium.make('sumogym/RobotSumo-v0', render_mode="human")
observation, info = env.reset()

# assert False
# print("")
# print("")
# print("")
# print(observation)


# assert False

joy = joystick.Joystick()

action = {"robotA": np.array([0, 0]), "robotB": np.array([0, 0])}


# Constants for mapping joystick values to wheel speeds
MAX_JOYSTICK_VALUE = 1
MAX_WHEEL_SPEED = 10


# Prepare PID controllers
robots_wheel_controllers = [None]*0
for ii in range(2):  # For each robot
    wheel_controllers: List[PID] = [None]*0
    for _ in range(2):
        wheel_controllers.append(PID(10, 0.3, 0.0, setpoint=0, sample_time=env.timestep,
                                     output_limits=[-1, 1]))
    robots_wheel_controllers.append(wheel_controllers)

terminated = False
while not terminated:
    # Calculate the wheel speeds
    left_speed_A = (joy.left_axis_y + joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    right_speed_A = (joy.left_axis_y - joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    left_speed_B = (joy.right_axis_y + joy.right_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    right_speed_B = (joy.right_axis_y - joy.right_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    robot_wheel_speeds = [[left_speed_A, right_speed_A], [left_speed_B, right_speed_B]]

    robot_torques = [None]*0
    for wheel_speeds, robot_name, wheel_controllers \
        in zip(robot_wheel_speeds, ["robotA", "robotB"], robots_wheel_controllers): # For each robot

        robot_observation = observation[f"{robot_name}_wheel_velocities"]
        robot_torque_normalised = np.zeros(2)

        for ii, (wheel_controller, current_wheel_velocity, desired_wheel_velocity) in \
                enumerate(zip(wheel_controllers, robot_observation, wheel_speeds)): # For each wheel
            wheel_controller.setpoint = desired_wheel_velocity
            robot_torque_normalised[ii] = wheel_controller(current_wheel_velocity)

        robot_torques.append(robot_torque_normalised)

    action = {"robotA": np.array(robot_torques[0]), "robotB": robot_torques[1]}
    observation, reward, _, truncated, info = env.step(action)


    robotA_observation = utils.get_robot_observation("robotA", observation)
    robotB_observation = utils.get_robot_observation("robotB", observation)

    # pprint.pprint(robotA_observation)
    # pprint.pprint(robotB_observation)

    time.sleep(env.timestep)
