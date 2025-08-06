import pathlib

### Task parameters
DATA_DIR = '/home/zzt/actnew/data'

# GYM-ALOHA task mapping (standardized naming)
GYM_ALOHA_TASK_MAPPING = {
    'peg-insertion-v1': 'bimanual_aloha_peg_insertion',
    'cube-transfer-v1': 'bimanual_aloha_cube_transfer', 
    'color-cubes-v1': 'bimanual_aloha_color_cubes',
    'thread-needle-v1': 'bimanual_aloha_thread_needle',
    'hook-package-v1': 'bimanual_aloha_hook_package',
    'pour-test-tube-v1': 'bimanual_aloha_pour_test_tube',
    'slot-insertion-v1': 'bimanual_aloha_slot_insertion',
}

SIM_TASK_CONFIGS = {
    # Original ACT tasks
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },

    # Bimanual ALOHA tasks (GYM-ALOHA compatible)
    'bimanual_aloha_cube_transfer': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_cube_transfer',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/cube-transfer-v1'
    },

    'bimanual_aloha_peg_insertion': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_peg_insertion',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/peg-insertion-v1'
    },

    'bimanual_aloha_color_cubes': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_color_cubes',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/color-cubes-v1'
    },

    'bimanual_aloha_slot_insertion': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_slot_insertion',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/slot-insertion-v1'
    },

    # Converted gym-aloha dataset (with real velocities - RECOMMENDED)
    'converted_bimanual_aloha_slot_insertion_with_vel': {
        'dataset_dir': DATA_DIR + '/converted_bimanual_aloha_slot_insertion_with_vel',
        'num_episodes': 50,  # 使用所有50个episodes
        'episode_len': 301,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/slot-insertion-v1',
        'converted_from': 'gv_sim_slot_insertion_2arms'
    },

    'bimanual_aloha_hook_package': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_hook_package',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/hook-package-v1'
    },

    'bimanual_aloha_pour_test_tube': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_pour_test_tube',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/pour-test-tube-v1'
    },

    'bimanual_aloha_thread_needle': {
        'dataset_dir': DATA_DIR + '/bimanual_aloha_thread_needle',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right'],
        'gym_aloha_env': 'gym_av_aloha/thread-needle-v1'
    },


}

### Simulation envs fixed constants
# Updated to match gym_aloha timing for proper trajectory playback
DT = 0.04  # Changed from 0.02 to match gym_aloha SIM_DT
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Bimanual ALOHA configuration
BIMANUAL_ALOHA_JOINT_NAMES = {
    'left': ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"],
    'right': ["right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate"]
}

# Start pose for bimanual ALOHA setup (left + right arms with grippers)
# Updated to match training data from converted_bimanual_aloha_slot_insertion
BIMANUAL_ALOHA_START_ARM_POSE = [
    # Left arm (6 joints + 1 gripper with 2 fingers)
    0, -0.082, 1.06, 0, -0.953, 0, 0.037, 0.037,  # gripper: both fingers open (0.037m = PUPPET_GRIPPER_POSITION_OPEN)
    # Right arm (6 joints + 1 gripper with 2 fingers) 
    0, -0.082, 1.06, 0, -0.953, 0, 0.037, 0.037   # gripper: both fingers open (0.037m = PUPPET_GRIPPER_POSITION_OPEN)
]

# Control values for bimanual ALOHA (14D: actuators only, no finger positions)
# Updated to match training data from converted_bimanual_aloha_slot_insertion
BIMANUAL_ALOHA_START_ARM_CONTROL = [
    # Left arm (6 joints + 1 gripper control) - gripper normalized to [0,1] where 1=open
    0, -0.082, 1.06, 0, -0.953, 0, 1.0,
    # Right arm (6 joints + 1 gripper control) - gripper normalized to [0,1] where 1=open
    0, -0.082, 1.06, 0, -0.953, 0, 1.0
]



XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.037   # Match XML ctrlrange upper bound
PUPPET_GRIPPER_POSITION_CLOSE = 0.002  # Match XML ctrlrange lower bound

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

# Gym-ALOHA to ACT gripper conversion functions
# gym_aloha uses normalized gripper values (0=closed, 1=open)
# actnew expects actual finger positions in meters (slide joints)
GYM_ALOHA_GRIPPER_TO_ACT_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(x)
ACT_TO_GYM_ALOHA_GRIPPER_FN = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x)

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
