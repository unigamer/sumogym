
def get_joint_info_dict(p, bodyUniqueId, jointIndex, physicsClientId=0):
    joint_info = p.getJointInfo(
        bodyUniqueId, jointIndex, physicsClientId=physicsClientId)
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


def get_robot_observation(robot_name, observation):
    assert robot_name in ["robotA", "robotB"]

    if robot_name == "robotA":
        opponent_name = "robotB"
    else:
        opponent_name = "robotA"

    result = {}

    for key, value in observation.items():
        if robot_name in key:
            new_key = key.replace(f"{robot_name}", "my")
            result[new_key] = value

    for key, value in observation.items():
        if opponent_name in key:
            new_key = key.replace(f"{opponent_name}", "opponent")
            result[new_key] = value

    result["robots_colliding"] = observation["robots_colliding"]

    return result
