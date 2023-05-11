import gymnasium
# import sumogym
import time
import numpy as np

from . import joystick
from simple_pid import PID
from typing import List

env = gymnasium.make('sumogym/RobotSumo-v0', render_mode="human")
observation, info = env.reset()
# print("")
# print("")
# print("")
# print(observation)


# assert False

joy = joystick.Joystick()

action = {"robotA": np.array([0,0]), "robotB": np.array([0,0])}


# Constants for mapping joystick values to wheel speeds
MAX_JOYSTICK_VALUE = 1
MAX_WHEEL_SPEED = 40


# Prepare PID controllers
wheel_controllers: List[PID] = [None]*0
for _ in range(2):
    wheel_controllers.append(PID(10, 0.3, 0.0, setpoint=0, sample_time=env.timestep,
                             output_limits=[-1, 1]))



while True:
    # Calculate the wheel speeds
    left_speed = (joy.left_axis_y + joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    right_speed = (joy.left_axis_y - joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE

    robotA_observation = observation["robotA"]
    robotA_torque_normalised = np.zeros(2)

    for ii, (wheel_controller, current_wheel_velocity, desired_wheel_velocity) in \
        enumerate(zip(wheel_controllers, robotA_observation, [left_speed, right_speed])):
        wheel_controller.setpoint = desired_wheel_velocity
        robotA_torque_normalised[ii] = wheel_controller(current_wheel_velocity)

    action = {"robotA": np.array(robotA_torque_normalised), "robotB": [0,0]}
    observation, _, _, _, _ = env.step(action)
    time.sleep(env.timestep)




