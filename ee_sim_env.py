import numpy as np
import collections
import os

from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose, sample_insertion_pose,sample_box_cupboard_pose, sample_stack_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
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
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    # Cupboard
    elif 'sim_cupboard_scripted' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_cupboard.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = CupboardEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                n_sub_steps=None, flat_observation=False)
        
    elif 'sim_stack_scripted' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_stack.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = StackEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

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
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

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


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

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


# Cupboard
class CupboardEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)

        box_pose, target_box_pose, drawer_initial_pose = sample_box_cupboard_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7  # first 16 is robot qpos, 7 is pose dim # hacky

        # randomize box position
        box_start_id = physics.model.name2id('green_box_joint', 'joint')
        box_start_idx = id2index(box_start_id)
        np.copyto(physics.data.qpos[box_start_idx: box_start_idx + 7], box_pose)

        target_box_id = physics.model.name2id('target_box_joint', 'joint')
        target_box_idx = id2index(target_box_id)
        np.copyto(physics.data.qpos[target_box_idx: target_box_idx + 7], target_box_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        """
        双臂协作抽屉任务奖励函数
        左臂：拉开抽屉（抓取handle_box）
        右臂：抓取绿色盒子并放入抽屉
        """
        # 获取所有接触对
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # 检测接触状态
        touch_left_gripper_handle = ("handle_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper_box = ("green_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        box_touch_table = ("green_box", "table") in all_contact_pairs
        box_touch_target = ("green_box", "target_box") in all_contact_pairs
        
        # 获取抽屉位置 - 抽屉滑出的距离
        drawer_position = physics.named.data.qpos['drawer_joint']  # 0表示关闭，正值表示打开
        drawer_opened = drawer_position > 0.05  # 抽屉打开超过5cm
        drawer_fully_opened = drawer_position > 0.10  # 抽屉充分打开
        
        # 获取绿色盒子位置
        box_pos = physics.named.data.qpos['green_box_joint'][:3]  # 盒子的xyz坐标
        
        # 定义抽屉内部区域（基于XML中抽屉的位置和尺寸）
        # cupboard位置是(-0.1, 0.6, 0.0)，旋转了180度
        # 抽屉内部空间大致为：
        drawer_interior_x_range = [-0.28, -0.12]  # 抽屉内部x范围
        drawer_interior_y_range = [0.45, 0.75]    # 抽屉内部y范围  
        drawer_interior_z_range = [0.04, 0.12]    # 抽屉内部z范围
        
        box_in_drawer = (drawer_interior_x_range[0] <= box_pos[0] <= drawer_interior_x_range[1] and
                         drawer_interior_y_range[0] <= box_pos[1] <= drawer_interior_y_range[1] and
                         drawer_interior_z_range[0] <= box_pos[2] <= drawer_interior_z_range[1])

        # 分阶段奖励设计
        reward = 0
        
        # 阶段1：左臂接触抽屉把手
        if touch_left_gripper_handle:
            reward = 1
            
        # 阶段2：抽屉开始打开
        if drawer_opened:
            reward = 2
            
        # 阶段3：右臂抓取物体
        if touch_right_gripper_box:
            reward = max(reward, 2)  # 至少2分
            
        # 阶段4：物体被提起（脱离桌面）
        if touch_right_gripper_box and not box_touch_table:
            reward = max(reward, 3)
            
        # 阶段5：双臂协作完成 - 抽屉开着且物体被抓取
        if drawer_opened and touch_right_gripper_box and not box_touch_table:
            reward = 4
            
        # 阶段6：物体成功放入抽屉内部 或者 与target_box接触
        if box_in_drawer or (box_touch_target and not box_touch_table):
            reward = 6
            
        return reward

class StackEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        
        # randomize block positions
        green_pose, red_pose, blue_pose= sample_stack_pose()

        
        id2index = lambda j_id: 16 + (j_id - 16) * 7  # first 16 is robot qpos, 7 is pose dim # hacky

        # Set green block (base) position
        green_start_id = physics.model.name2id('green_block_joint', 'joint')
        green_start_idx = id2index(green_start_id)
        np.copyto(physics.data.qpos[green_start_idx : green_start_idx + 7], green_pose)
        # print(f"randomized green block position to {green_pose}")

        # Set red block position  
        red_start_id = physics.model.name2id('red_block_joint', 'joint')
        red_start_idx = id2index(red_start_id)
        np.copyto(physics.data.qpos[red_start_idx : red_start_idx + 7], red_pose)
        # print(f"randomized red block position to {red_pose}")

        # Set blue block position
        blue_start_id = physics.model.name2id('blue_block_joint', 'joint')
        blue_start_idx = id2index(blue_start_id)
        np.copyto(physics.data.qpos[blue_start_idx : blue_start_idx + 7], blue_pose)
        # print(f"randomized blue block position to {blue_pose}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state



    def get_reward(self, physics):

        
        # Check current contacts
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # Check gripper contacts
        touch_red_left = ("red_block", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_blue_right = ("blue_block", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        
        # Check table contacts
        red_touch_table = ("red_block", "table") in all_contact_pairs
        blue_touch_table = ("blue_block", "table") in all_contact_pairs
        green_touch_table = ("green_block", "table") in all_contact_pairs

        # Check block-to-block contacts for stacking
        red_on_green = ("green_block", "red_block") in all_contact_pairs
        blue_on_red = ("red_block", "blue_block") in all_contact_pairs

        # 分阶段奖励设计
        reward = 0
        
        # 阶段1：左臂抓取红色方块
        if touch_red_left:
            reward = 1
            
        # 阶段2：红色方块成功堆叠在绿色上
        if red_on_green and not red_touch_table:
            reward = 2
            
        # 阶段3：右臂抓取蓝色方块（可以与阶段2并行）
        if touch_blue_right:
            reward = max(reward, 2)  # 至少2分
            
        # 阶段4：蓝色方块被提起（脱离桌面）
        if touch_blue_right and not blue_touch_table:
            reward = max(reward, 3)
            
            
        # 阶段6：完美三层塔 - 蓝色成功放在红色上
        if blue_on_red and red_on_green and not blue_touch_table and not red_touch_table:
            reward = 4

    
        return reward