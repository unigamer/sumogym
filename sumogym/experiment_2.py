import pathlib
import pybullet_data
import pybullet
import pybullet_utils.bullet_client as bc
import time
import numpy as np

p = bc.BulletClient(connection_mode=pybullet.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")

simulation_frequency = 240 # Hz

timestep = 1/simulation_frequency

robot_path = pathlib.Path(
    __file__).parents[0] / "robot_descriptions/simple_robot/simple_robot_2.urdf"


# /mnt/sde/src/rl/sumo/sumogym/sumogym/robot_descriptions/simple_robot/simple_robot.urdf
# /mnt/sde/src/rl/sumo/sumogym/robot_descriptions/simple_robot/simple_robot.urdf

def get_joint_info_dict(bodyUniqueId, jointIndex, physicsClientId=0):
    joint_info = p.getJointInfo(bodyUniqueId, jointIndex, physicsClientId=physicsClientId)
    joint_dict = {
        "jointIndex": joint_info[0],
        "jointName": joint_info[1].decode("utf-8"),
        "jointType": joint_info[2],
        "qIndex": joint_info[3],
        "uIndex": joint_info[4],
        "flags": joint_info[5],
        "jointDamping": joint_info[6],
        "jointFriction": joint_info[7],
        "jointLowerLimit": joint_info[8],
        "jointUpperLimit": joint_info[9],
        "jointMaxForce": joint_info[10],
        "jointMaxVelocity": joint_info[11],
        "linkName": joint_info[12].decode("utf-8"),
        "jointAxis": joint_info[13],
        "parentFramePos": joint_info[14],
        "parentFrameOrn": joint_info[15],
        "parentIndex": joint_info[16]
    }
    return joint_dict


print(robot_path)


robot = p.loadURDF(str(robot_path), basePosition = [0,0, 0.3])
p.setGravity(0, 0, -9.8)

joint_dictionary = {}

for ii in range(p.getNumJoints(robot)):
    jointInfo = p.getJointInfo(robot,ii)
    joint_dictionary[jointInfo[1].decode("utf-8") ] = jointInfo[0]

    print(get_joint_info_dict(robot,ii))

print(joint_dictionary)

# p.setJointMotorControl2(robot, joint_dictionary["joint_left_wheel"], p.TORQUE_CONTROL)
# p.setJointMotorControl2(robot, joint_dictionary["joint_right_wheel"], p.TORQUE_CONTROL)

integral_error = 0.0
prev_error = 0.0

runSimulation = True
while runSimulation:
        

        

        desired_velocity = 1.0  # rad/s

        joint_index = 0
        # Calculate the torque required to achieve the desired velocity
        joint_state = p.getJointState(robot, joint_index)
        current_velocity = joint_state[1]

        error = desired_velocity - current_velocity

        kp = 0.1
        ki = 0.01
        proportional_error = kp * error
        integral_error += ki * error

        # Apply the proportional and integral components to the control signal
        control_signal = proportional_error + integral_error




        torque = control_signal

        torque = np.clip(torque, -40, 40 )

        print(torque)

        # Apply the desired torque to the joint
        p.setJointMotorControl2(robot, joint_index, p.TORQUE_CONTROL, force=torque)

    # Step the s

        # joint_state = p.getJointState(robot, 1)
        # # Calculate the torque based on the joint position and velocity
        # position = joint_state[0]
        # velocity = joint_state[1]
        # torque = -0.1 * position - 0.01 * velocity
        # p.setJointMotorControl2(robot, 1, p.TORQUE_CONTROL, force = torque)



        # p.setJointMotorControl2(robot, joint_dictionary["joint_right_wheel"], p.TORQUE_CONTROL, force = 40)
        # jointStates = p.getJointStates(robot, [joint_dictionary["joint_left_wheel"], joint_dictionary["joint_right_wheel"]])
        # print(jointStates)
        # print(jointStates[joint_dictionary["joint_left_wheel"]][1])
        p.stepSimulation()
        time.sleep(timestep)

        
