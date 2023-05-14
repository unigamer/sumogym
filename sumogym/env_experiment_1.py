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

action = {"robotA": np.array([0, 0]), "robotB": np.array([0, 0])}


# Constants for mapping joystick values to wheel speeds
MAX_JOYSTICK_VALUE = 1
MAX_WHEEL_SPEED = 40


# Prepare PID controllers
robots_wheel_controllers = [None]*0
for ii in range(2):  # For each robot
    wheel_controllers: List[PID] = [None]*0
    for _ in range(2):
        wheel_controllers.append(PID(10, 0.3, 0.0, setpoint=0, sample_time=env.timestep,
                                     output_limits=[-1, 1]))
    robots_wheel_controllers.append(wheel_controllers)


while True:
    # Calculate the wheel speeds
    left_speed_A = (joy.left_axis_y + joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    right_speed_A = (joy.left_axis_y - joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE

    left_speed_B = (joy.right_axis_y + joy.right_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    right_speed_B = (joy.right_axis_y - joy.right_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE

    robot_wheel_speeds = [[left_speed_A, right_speed_A], [left_speed_B, right_speed_B]]

    robot_torques = [None]*0

    for wheel_speeds, robot_name, wheel_controllers in zip(robot_wheel_speeds, ["robotA", "robotB"], robots_wheel_controllers):
        
        robot_observation = observation[robot_name]
        robot_torque_normalised = np.zeros(2)

        for ii, (wheel_controller, current_wheel_velocity, desired_wheel_velocity) in \
                enumerate(zip(wheel_controllers, robot_observation, wheel_speeds)):
            wheel_controller.setpoint = desired_wheel_velocity
            robot_torque_normalised[ii] = wheel_controller(current_wheel_velocity)

        robot_torques.append(robot_torque_normalised)

    action = {"robotA": np.array(robot_torques[0]), "robotB": robot_torques[1]}
    observation, _, _, _, _ = env.step(action)
    time.sleep(env.timestep)
