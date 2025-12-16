import mujoco
import mujoco.viewer
import numpy as np
import time
import copy
import os
import requests
import numpy as np


np.set_printoptions(precision=3)

url = "http://192.168.123.220:12000/unitree/z1"
database = {
    "func": "",
    "args": {},
}


def labelRun(label):
    assert len(label) < 10

    # copy data
    data = database.copy()
    data["func"] = "labelRun"
    data["args"] = {
        "label": label,
    }
    return requests.post(url, json=data)


def labelSave(label):
    assert len(label) < 10

    # copy data
    data = database.copy()
    data["func"] = "labelSave"
    data["args"] = {
        "label": label,
    }
    return requests.post(url, json=data)


def backToStart():
    data = database.copy()
    data["func"] = "backToStart"
    return requests.post(url, json=data)


def Passive():
    data = database.copy()
    data["func"] = "Passive"
    return requests.post(url, json=data)


def getQ():
    data = database.copy()
    data["func"] = "getQ"
    return requests.post(url, json=data)


def MoveJ(q: list, gripperPos=0, speed=0.5):
    assert len(q) == 6

    data = database.copy()
    data["func"] = "MoveJ"
    data["args"] = {
        "q": q,
        "gripperPos": gripperPos,
        "maxSpeed": speed,
    }
    return requests.post(url, json=data)

def setGripper(position, speed=128, force=128):
    position = max(0, min(255, position))
    speed = max(0, min(255, speed))
    force = max(0, min(255, force))

    data = database.copy()
    data["func"] = "setGripper"
    data["args"] = {
        "position": position,
        "speed": speed,
        "force": force,
    }
    return requests.post(url, json=data)

# -----------------------------------------------------------------------------
# 1. XML SETUP (Wrapper to combine Z1 and the Object)
# -----------------------------------------------------------------------------
SCENE_XML = """
<mujoco model="z1_scene">
    <include file="z1_description/z1.xml"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

        <!-- The Target Object -->
        <!-- body name="target_object" pos="0.35 0.0 0.021">
            <joint type="free"/>
            <geom name="target_geom" type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.1"/>
        </body> -->
    </worldbody>

    <!-- EXCLUDE COLLISIONS BETWEEN ADJACENT LINKS -->
    <!-- This prevents "self-collision" false positives at the joints -->

</mujoco>
"""


# -----------------------------------------------------------------------------
# 2. INVERSE KINEMATICS (IK)
# -----------------------------------------------------------------------------
def solve_ik(model, data, target_pos, target_quat, end_effector_id, max_steps=500, tol=1e-5):
    """
    Solves IK using Damped Least Squares (Levenberg-Marquardt).
    """
    q0 = data.qpos.copy()
    dof_indices = np.arange(6)

    success = False

    for i in range(max_steps):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        current_pos = data.geom_xpos[end_effector_id]
        current_mat = data.geom_xmat[end_effector_id].reshape(3, 3)

        err_pos = target_pos - current_pos

        target_mat_flat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat_flat, target_quat)
        target_mat = target_mat_flat.reshape(3, 3)

        err_rot = 0.5 * (np.cross(current_mat[:, 0], target_mat[:, 0]) +
                         np.cross(current_mat[:, 1], target_mat[:, 1]) +
                         np.cross(current_mat[:, 2], target_mat[:, 2]))

        error = np.concatenate([err_pos, err_rot])
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            success = True
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jacp, jacr, end_effector_id)

        jac = np.vstack([jacp, jacr])
        jac_arm = jac[:, dof_indices]

        lambda_val = 0.1
        diag = lambda_val * np.eye(6)

        dq_arm = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, error)

        data.qpos[dof_indices] += dq_arm

        # Approximate Joint Limits (Z1)
        limits_min = np.array([-2.6, 0.0, -2.8, -1.5, -1.3, -2.7])
        limits_max = np.array([2.6, 2.9, 0.0, 1.5, 1.3, 2.7])
        data.qpos[dof_indices] = np.clip(data.qpos[dof_indices], limits_min, limits_max)

    result_q = data.qpos.copy()
    data.qpos[:] = q0
    mujoco.mj_forward(model, data)

    if success:
        return result_q
    else:
        print(f"IK Failed to converge. Final error: {error_norm:.4f}")
        return None


# -----------------------------------------------------------------------------
# 3. COLLISION CHECKING
# -----------------------------------------------------------------------------
def is_collision(model, data, q_check, debug=False):
    """
    Checks if a configuration q_check is in collision.
    """
    q_save = data.qpos.copy()

    data.qpos[:6] = q_check[:6]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)

    in_collision = False

    for i in range(data.ncon):
        contact = data.contact[i]

        # If debugging, print names of colliding geoms
        if debug:
            name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            print(f"Collision Detected between: {name1} and {name2}")

        in_collision = True
        break

    data.qpos[:] = q_save
    return in_collision


# -----------------------------------------------------------------------------
# 4. RRT PATH PLANNING
# -----------------------------------------------------------------------------
class Node:
    def __init__(self, q, parent=None):
        self.q = q
        self.parent = parent


def rrt_plan(model, data, start_q, goal_q, max_iter=30000, step_size=0.5):
    print("Starting RRT Planner...")

    # 1. Validate Start State
    if is_collision(model, data, start_q, debug=True):
        print("CRITICAL: Start configuration is in collision! Cannot plan.")
        return None

    start_node = Node(start_q[:6])
    goal_node = Node(goal_q[:6])

    tree = [start_node]
    joint_limits_min = np.array([-2.6, 0.0, -2.8, -1.5, -1.3, -2.7])
    joint_limits_max = np.array([2.6, 2.9, 0.0, 1.5, 1.3, 2.7])

    for i in range(max_iter):
        if np.random.rand() < 0.2:
            sample_q = goal_node.q
        else:
            sample_q = np.random.uniform(joint_limits_min, joint_limits_max)

        distances = [np.linalg.norm(node.q - sample_q) for node in tree]
        nearest_idx = np.argmin(distances)
        nearest_node = tree[nearest_idx]

        direction = sample_q - nearest_node.q
        distance = np.linalg.norm(direction)

        direction = direction / distance
        move_dist = min(step_size, distance)
        new_q = nearest_node.q + direction * move_dist

        if not is_collision(model, data, new_q):
            new_node = Node(new_q, nearest_node)
            tree.append(new_node)

            if np.linalg.norm(new_q - goal_node.q) < 0.04:  # slightly larger acceptance radius
                if not is_collision(model, data, goal_node.q):
                    print(f"Path found in {i} iterations!")
                    path = [goal_node.q]
                    curr = new_node
                    while curr is not None:
                        path.append(curr.q)
                        curr = curr.parent
                    return path[::-1]

    print("RRT failed to find a path.")
    return None


# -----------------------------------------------------------------------------
# 5. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)

    try:
        ee_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "z1_GripperStator")
        if ee_geom_id == -1:
            ee_geom_id = model.ngeom - 1
    except:
        ee_geom_id = 0

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_object")

    has_display = os.environ.get('DISPLAY') is not None
    viewer = None

    if has_display:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception:
            has_display = False

    # ---------------------------------------
    # INITIALIZE ROBOT TO A SAFE "HOME" POSE
    # ---------------------------------------
    # Joint order: [base, shoulder, elbow, wrist1, wrist2, wrist3]
    # This pose bends the elbow so it's not sticking straight up/down
    home_pose = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.5])
    data.qpos[:7] = home_pose

    mujoco.mj_step(model, data)
    if viewer: viewer.sync()

    # 1. Get Target Position
    # target_pos = data.body(target_body_id).xpos.copy()
    # target_pos[2] += 0.05
    # target_pos[0] -= 0.05
    target_pos = [0.30, 0, 0.13]
    target_quat = np.array([0, -0.969, 0, 0.25])

    print(f"Target Position: {target_pos}")

    # 2. Solve IK
    print("Solving IK...")
    q_goal = solve_ik(model, data, target_pos, target_quat, ee_geom_id)

    if q_goal is None:
        print("Could not find IK solution. Exiting.")
        return

    # 3. Plan Path
    # Use the current safe home pose as start
    q_start = data.qpos[:6].copy()
    path = rrt_plan(model, data, q_start, q_goal)

    if path is None:
        return

    # 4. Execute Path
    print("Closing Gripper...")
    for _ in range(100):
        data.ctrl[6] = -1.5
        mujoco.mj_step(model, data)
        if viewer and viewer.is_running():
            viewer.sync()
            time.sleep(0.005)
    print(f"Executing Path with {len(path)} waypoints...")
    labelRun("forward")
    for i, waypoint in enumerate(path):
        MoveJ(waypoint.tolist(), gripperPos=-1.5, speed=0.7)
        setGripper(0, 255, 255)
        if i == len(path) - 1:
            MoveJ(waypoint.tolist(), gripperPos=0, speed=0.7)
            setGripper(255, 255, 255)
        # for _ in range(50):
        #     data.ctrl[:6] = waypoint
        #     mujoco.mj_step(model, data)
        #     if viewer and viewer.is_running():
        #         viewer.sync()
        #         time.sleep(0.01)
        #
        # if not has_display:
        #     print(f"Reached Waypoint {i + 1}/{len(path)}")
    backToStart()
    input()
    labelRun("forward")
    MoveJ(waypoint.tolist(), gripperPos=-1.5, speed=0.7)
    setGripper(255, 255, 255)
    # 5. Close Gripper
    print("Closing Gripper...")
    for _ in range(100):
        data.ctrl[6] = 0.0
        mujoco.mj_step(model, data)
        if viewer and viewer.is_running():
            viewer.sync()
            time.sleep(0.005)

    print("Path execution complete.")

    if viewer:
        print("Keeping window open.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()