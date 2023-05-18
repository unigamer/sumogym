import gymnasium
# import sumogym
import time
import pprint
import numpy as np

from . import joystick
from . import utils
from simple_pid import PID
from typing import List

from sumogym.envs.robot_sumo_single import RobotSumoSingleEnv
from simple_pid import PID
from . import joystick
from . import utils


env = RobotSumoSingleEnv(render_mode="human")
observation, info = env.reset()

pprint.pprint(observation)
pprint.pprint(info)


print("")
print("")
print("")
print("")

joy = joystick.Joystick()

# Constants for mapping joystick values to wheel speeds
MAX_JOYSTICK_VALUE = 1
MAX_WHEEL_SPEED = 10


# Prepare PID controllers

wheel_controllers: List[PID] = [None]*0
for _ in range(2):
    wheel_controllers.append(PID(10, 0.3, 0.0, setpoint=0, sample_time=env.timestep,
                                    output_limits=[-1, 1]))


terminated = False
while not terminated:

    # Calculate the wheel speeds
    left_speed_A = (joy.left_axis_y + joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE
    right_speed_A = (joy.left_axis_y - joy.left_axis_x) * MAX_WHEEL_SPEED / MAX_JOYSTICK_VALUE

    desired_wheel_velocities = [left_speed_A, right_speed_A]

    current_wheel_velocities = observation["self_wheel_velocities"]
    robot_torque_normalised = np.zeros(2)
    for ii, (wheel_controller, current_wheel_velocity, desired_wheel_velocity) in \
                enumerate(zip(wheel_controllers, current_wheel_velocities , desired_wheel_velocities)): # For each wheel
            wheel_controller.setpoint = desired_wheel_velocity
            robot_torque_normalised[ii] = wheel_controller(current_wheel_velocity)
    action = robot_torque_normalised
  

    observation, reward, terminated, truncatation, info = env.step(action)

    pprint.pprint(reward)



#     # pprint.pprint(rewards)
    time.sleep(1/240.0)