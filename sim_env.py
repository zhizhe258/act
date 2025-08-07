import numpy as np
import os
import collections
import matplotlib.pyplot as plt
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

from constants import DT, XML_DIR, START_ARM_POSE
from constants import BIMANUAL_ALOHA_START_ARM_POSE, BIMANUAL_ALOHA_START_ARM_CONTROL, BIMANUAL_ALOHA_JOINT_NAMES
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def make_sim_env(task_name, enable_distractors=False):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'bimanual_aloha_peg_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, 'task_peg_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = BimanualAlohaPegInsertionTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'bimanual_aloha_slot_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, 'task_slot_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = BimanualAlohaSlotInsertionTask(random=enable_distractors)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'bimanual_aloha_hook_package' in task_name:
        xml_path = os.path.join(XML_DIR, 'task_hook_package.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = BimanualAlohaHookPackageTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'bimanual_aloha_pour_test_tube' in task_name or 'converted_bimanual_aloha_cube_transfer' in task_name:
        xml_path = os.path.join(XML_DIR, 'task_pour_test_tube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = BimanualAlohaPourTestTubeTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'bimanual_aloha_thread_needle' in task_name:
        xml_path = os.path.join(XML_DIR, 'task_thread_needle.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = BimanualAlohaThreadNeedleTask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7:7+6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7+6]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, left_gripper_action]
        full_right_gripper_action = [right_gripper_action, right_gripper_action]

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7*2:] = BOX_POSE[0] # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


class BimanualAlohaTask(base.Task):
    """Base class for bimanual ALOHA tasks"""
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        """
        Action space for bimanual ALOHA setup (14D):
        - left_arm_action (6): left arm joint positions
        - left_gripper_action (1): normalized left gripper position  
        - right_arm_action (6): right arm joint positions
        - right_gripper_action (1): normalized right gripper position
        """
        left_arm_action = action[:6]
        left_gripper_normalized = action[6]
        right_arm_action = action[7:13]
        right_gripper_normalized = action[13]

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(left_gripper_normalized)
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(right_gripper_normalized)

        # For bimanual ALOHA, each gripper is controlled by single actuator
        env_action = np.concatenate([
            left_arm_action, [left_gripper_action],  # left arm + gripper
            right_arm_action, [right_gripper_action]  # right arm + gripper
        ])
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # Extract positions for two arms: left (8), right (8)
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        
        return np.concatenate([
            left_arm_qpos, left_gripper_qpos, 
            right_arm_qpos, right_gripper_qpos
        ])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        # Extract velocities for two arms: left (8), right (8)
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        
        return np.concatenate([
            left_arm_qvel, left_gripper_qvel,
            right_arm_qvel, right_gripper_qvel
        ])

    @staticmethod
    def get_env_state(physics):
        # Get state of objects in environment (after robot joints)
        env_state = physics.data.qpos.copy()[16:]  # Skip 16 robot joints (2 arms * 8 joints each)
        return env_state

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        
        # Render from available cameras
        obs['images']['overhead_cam'] = physics.render(height=480, width=640, camera_id='overhead_cam')
        obs['images']['worms_eye_cam'] = physics.render(height=480, width=640, camera_id='worms_eye_cam')
        obs['images']['wrist_cam_left'] = physics.render(height=480, width=640, camera_id='wrist_cam_left')
        obs['images']['wrist_cam_right'] = physics.render(height=480, width=640, camera_id='wrist_cam_right')
        # Note: zed_cam_left and zed_cam_right were removed with middle arm
        
        return obs

    def get_reward(self, physics):
        # Default reward implementation - to be overridden by specific tasks
        return 0


class BimanualAlohaPegInsertionTask(BimanualAlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = BIMANUAL_ALOHA_START_ARM_POSE
            np.copyto(physics.data.ctrl, BIMANUAL_ALOHA_START_ARM_CONTROL)
        super().initialize_episode(physics)

    def get_reward(self, physics):
        # Peg Insertion Reward Logic
        touch_left_gripper = False
        touch_right_gripper = False
        peg_touch_table = False
        hole_touch_table = False
        peg_touch_hole = False
        pin_touched = False

        # Check contact pairs
        contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            geom1 = physics.model.id2name(id_geom_1, 'geom')
            geom2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "peg" and geom2.startswith("vx300s_7_gripper"):
                touch_right_gripper = True
            if geom1.startswith("hole-") and geom2.startswith("vx300s_7_gripper"):
                touch_left_gripper = True
            if geom1 == "table" and geom2 == "peg":
                peg_touch_table = True
            if geom1 == "table" and geom2.startswith("hole-"):
                hole_touch_table = True
            if geom1 == "peg" and geom2.startswith("hole-"):
                peg_touch_hole = True
            if geom1 == "peg" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not hole_touch_table): # grasp both
            reward = 2
        if peg_touch_hole and (not peg_touch_table) and (not hole_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward


class BimanualAlohaSlotInsertionTask(BimanualAlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = BIMANUAL_ALOHA_START_ARM_POSE
            np.copyto(physics.data.ctrl, BIMANUAL_ALOHA_START_ARM_CONTROL)
            
            # Randomize distractor positions if random is True
            if self.random:
                from utils import sample_distractor_poses
                distractor_poses = sample_distractor_poses()
                
                # Set distractor positions
                distractor_names = ['distractor1', 'distractor2', 'distractor3']
                for i, distractor_name in enumerate(distractor_names):
                    try:
                        joint_id = physics.model.name2id(f'{distractor_name}_joint', 'joint')
                        pose_start_idx = 16 + (joint_id - 16) * 7  # First 16 is robot qpos, 7 is pose dim
                        np.copyto(physics.data.qpos[pose_start_idx:pose_start_idx + 7], distractor_poses[i])
                    except:
                        print(f"Warning: Could not find {distractor_name}_joint")
        
        super().initialize_episode(physics)

    def get_reward(self, physics):
        # Slot Insertion Reward Logic
        touch_left_gripper = False
        touch_right_gripper = False
        stick_touch_table = False
        stick_touch_slot = False
        pins_touch = False

        # Check contact pairs
        contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            geom1 = physics.model.id2name(id_geom_1, 'geom')
            geom2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "stick" and geom2.startswith("right"):
                touch_right_gripper = True
            if geom1 == "stick" and geom2.startswith("left"):
                touch_left_gripper = True
            if geom1 == "table" and geom2 == "stick":
                stick_touch_table = True
            if geom1 == "stick" and geom2.startswith("slot-"):
                stick_touch_slot = True
            if geom1 == "pin-stick" and geom2 == "pin-slot":
                pins_touch = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not stick_touch_table):  # grasp stick
            reward = 2
        if stick_touch_slot and (not stick_touch_table):  # peg and socket touching
            reward = 3
        if pins_touch:  # successful insertion
            reward = 4
        return reward


class BimanualAlohaHookPackageTask(BimanualAlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = BIMANUAL_ALOHA_START_ARM_POSE
            np.copyto(physics.data.ctrl, BIMANUAL_ALOHA_START_ARM_CONTROL)
        super().initialize_episode(physics)

    def get_reward(self, physics):
        # Hook Package Reward Logic
        touch_left_gripper = False
        touch_right_gripper = False
        package_touch_table = False
        package_touch_hook = False
        pin_touched = False

        # Check contact pairs
        contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            geom1 = physics.model.id2name(id_geom_1, 'geom')
            geom2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("package-") and geom2.startswith("right"):
                touch_right_gripper = True
            if geom1.startswith("package-") and geom2.startswith("left"):
                touch_left_gripper = True
            if geom1 == "table" and geom2.startswith("package-"):
                package_touch_table = True
            if geom1 == "hook" and geom2.startswith("package-"):
                package_touch_hook = True
            if geom1 == "pin-package" and geom2 == "pin-hook":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not package_touch_table):  # grasp both
            reward = 2
        if package_touch_hook and (not package_touch_table):
            reward = 3
        if pin_touched:
            reward = 4
        return reward


class BimanualAlohaPourTestTubeTask(BimanualAlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 3  # Pour test tube has 3 levels

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = BIMANUAL_ALOHA_START_ARM_POSE
            np.copyto(physics.data.ctrl, BIMANUAL_ALOHA_START_ARM_CONTROL)
        super().initialize_episode(physics)

    def get_reward(self, physics):
        # Pour Test Tube Reward Logic (from gym_av_aloha)
        touch_left_gripper = False
        touch_right_gripper = False
        tube1_touch_table = False
        tube2_touch_table = False
        pin_touched = False

        # Check contact pairs
        contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            geom1 = physics.model.id2name(id_geom_1, 'geom')
            geom2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("tube1-") and geom2.startswith("vx300s_7_gripper"):
                touch_right_gripper = True
            if geom1.startswith("tube2-") and geom2.startswith("vx300s_7_gripper"):
                touch_left_gripper = True
            if geom1 == "table" and geom2.startswith("tube1-"):
                tube1_touch_table = True
            if geom1 == "table" and geom2.startswith("tube2-"):
                tube2_touch_table = True
            if geom1 == "ball" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not tube1_touch_table) and (not tube2_touch_table):  # grasp both
            reward = 2
        if pin_touched:
            reward = 3
        return reward


class BimanualAlohaThreadNeedleTask(BimanualAlohaTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 5  # Thread needle has 5 levels
        self.threaded_needle = False # New attribute to track if needle is threaded

    def initialize_episode(self, physics):
        with physics.reset_context():
            physics.named.data.qpos[:16] = BIMANUAL_ALOHA_START_ARM_POSE
            np.copyto(physics.data.ctrl, BIMANUAL_ALOHA_START_ARM_CONTROL)
        super().initialize_episode(physics)

    def get_reward(self, physics):
        # Thread Needle Reward Logic
        touch_left_gripper = False
        touch_right_gripper = False
        needle_touch_table = False
        needle_touch_wall = False
        needle_touch_pin = False

        # Check contact pairs
        contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            geom1 = physics.model.id2name(id_geom_1, 'geom')
            geom2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "needle" and geom2.startswith("right"):
                touch_right_gripper = True
            if geom1 == "needle" and geom2.startswith("left"):
                touch_left_gripper = True
            if geom1 == "table" and geom2 == "needle":
                needle_touch_table = True
            if geom1 == "needle" and geom2.startswith("wall-"):
                needle_touch_wall = True
            if geom1 == "pin-needle" and geom2 == "pin-wall":
                self.threaded_needle = True
            if geom1 == "needle" and geom2 == "pin-wall":
                needle_touch_pin = True

        reward = 0
        if touch_right_gripper:  # touch needle
            reward = 1
        if touch_right_gripper and (not needle_touch_table):  # grasp needle
            reward = 2
        if needle_touch_wall and (not needle_touch_table):  # needle touching wall
            reward = 3
        if self.threaded_needle:  # needle threaded
            reward = 4
        # grasped needle on other side
        if touch_left_gripper and (not touch_right_gripper) and (not needle_touch_table) and (not needle_touch_pin) and self.threaded_needle:
            reward = 5
        return reward


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)
    # arm action
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # gripper action
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)
    action[6] = normalized_left_pos
    action[7+6] = normalized_right_pos
    return action

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_right', init_node=False)

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()

    for t in range(1000):
        action = get_action(master_bot_left, master_bot_right)
        ts = env.step(action)
        episode.append(ts)

        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)


if __name__ == '__main__':
    test_sim_teleop()

