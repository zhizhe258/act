#!/usr/bin/env python3

import h5py
import numpy as np
import os
import sys
import cv2
from pathlib import Path
import time
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from constants import SIM_TASK_CONFIGS, XML_DIR, DT, GYM_ALOHA_GRIPPER_TO_ACT_FN
from sim_env import make_sim_env

def load_insertion_episode(episode_path):
    """Load a single insertion episode from HDF5 file"""
    with h5py.File(episode_path, 'r') as f:
        data = {
            'qpos': f['observations/qpos'][()],
            'qvel': f['observations/qvel'][()], 
            'action': f['action'][()],
            'images': {}
        }
        
        # Load all camera images if available
        if 'observations/images' in f:
            for cam_name in f['observations/images'].keys():
                data['images'][cam_name] = f[f'observations/images/{cam_name}'][()]
                
    return data

def convert_14d_to_16d_qpos(qpos_14d):
    """Convert 14D data to 16D joint positions for the XML model"""
    if len(qpos_14d) != 14:
        return qpos_14d
        
    # Create 16D joint position array
    qpos_16d = np.zeros(16)
    
    # Left arm joints (0-5)
    qpos_16d[0:6] = qpos_14d[0:6]
    
    # Left gripper: convert single normalized value to finger position
    # Apply range validation - both fingers move together via equality constraint
    left_gripper_norm = np.clip(qpos_14d[6], 0.0, 1.0)  # validated normalized value from gym_aloha
    left_gripper_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(left_gripper_norm)
    qpos_16d[6] = left_gripper_joint     # left_left_finger (main actuated joint)
    qpos_16d[7] = left_gripper_joint     # left_right_finger (equality constraint syncs this)
    
    # Right arm joints (8-13)
    qpos_16d[8:14] = qpos_14d[7:13]
    
    # Right gripper: convert single normalized value to finger position
    # Apply range validation - both fingers move together via equality constraint
    right_gripper_norm = np.clip(qpos_14d[13], 0.0, 1.0)  # validated normalized value from gym_aloha
    right_gripper_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(right_gripper_norm)
    qpos_16d[14] = right_gripper_joint    # right_left_finger (main actuated joint)
    qpos_16d[15] = right_gripper_joint    # right_right_finger (equality constraint syncs this)
    
    return qpos_16d

def replay_episode_dmcontrol(episode_data, save_frames=True, enable_physics_interaction=False, force_open_gripper=False):
    """Replay episode using dm_control environment with optional physics interaction
    
    Args:
        episode_data: Episode data containing actions
        save_frames: Whether to save video frames
        enable_physics_interaction: If True, use position control + physics interaction.
                                   If False, use direct qpos setting (original mode)
    """
    
    # Create environment for bimanual slot insertion
    task_name = 'bimanual_aloha_slot_insertion'
    env = make_sim_env(task_name)
    
    if env is None:
        print("Failed to create environment")
        return None
        
    # Reset environment
    ts = env.reset()
    physics = env.physics
    
    action_sequence = episode_data['action']
    print(f"Replaying episode with {len(action_sequence)} timesteps")
    print(f"Environment DOF: {physics.model.nq}, Episode DOF: {action_sequence.shape[1]}")
    print(f"Environment actuators: {physics.model.nu}")
    
    
    # Required cameras for bimanual ALOHA
    camera_names = ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right']
    
    frames = {cam_name: [] for cam_name in camera_names}
    
    for i, action_14d in enumerate(action_sequence):
        # Both modes should use the same 14D normalized action data (0-1 for grippers)
        # This maintains consistency with gym-aloha standard format
        
        if enable_physics_interaction:
            # Physics interaction mode: Optionally force grippers to stay open
            if force_open_gripper:
                action_14d_physics = action_14d.copy()
                action_14d_physics[6] = -1.0   # Force left gripper fully open
                action_14d_physics[13] = -1.0  # Force right gripper fully open
            else:
                action_14d_physics = action_14d  # Use original gripper values
            
            # Use env.step() for true physics interaction
            # This goes through BimanualAlohaTask.before_step() which properly handles gripper conversion
            ts = env.step(action_14d_physics)
            
        else:  
            # Kinematic mode: Convert 14D normalized action to 16D joint positions for direct setting
            qpos_16d = convert_14d_to_16d_qpos(action_14d)
            
            # Set robot joint positions directly for kinematic control
            with physics.reset_context():
                physics.data.qpos[:16] = qpos_16d
                physics.data.qvel[:16] = np.zeros_like(qpos_16d)
            
            # Step physics to update simulation
            physics.step()
        
        # Capture frames from all cameras
        if save_frames and i % 2 == 0:  # Save every 2nd frame
            for cam_name in camera_names:
                try:
                    frame = physics.render(height=480, width=640, camera_id=cam_name)
                    frames[cam_name].append(frame)
                except Exception as e:
                    print(f"Warning: Failed to render {cam_name}: {e}")
        
        # Print debug info every 50 steps
        if i % 50 == 0:
            if enable_physics_interaction:
                # Physics mode: show original and potentially modified values
                orig_left_norm = np.clip(action_14d[6], 0.0, 1.0)
                orig_right_norm = np.clip(action_14d[13], 0.0, 1.0)
                
                if force_open_gripper:
                    # Show forced values (always 1.0 for open)
                    left_norm = 1.0
                    right_norm = 1.0
                    left_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(left_norm)
                    right_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(right_norm)
                    print(f"Step {i}: Original grippers=[{orig_left_norm:.3f}, {orig_right_norm:.3f}] → Forced open=[{left_norm:.3f}, {right_norm:.3f}] → joint=[{left_joint:.4f}, {right_joint:.4f}]")
                else:
                    # Show original values
                    left_norm = orig_left_norm
                    right_norm = orig_right_norm  
                    left_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(left_norm)
                    right_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(right_norm)
                    print(f"Step {i}: Physics grippers=[{left_norm:.3f}, {right_norm:.3f}] → joint=[{left_joint:.4f}, {right_joint:.4f}]")
            else:
                # Kinematic mode: show original values
                left_norm = np.clip(action_14d[6], 0.0, 1.0)
                right_norm = np.clip(action_14d[13], 0.0, 1.0)
                left_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(left_norm)
                right_joint = GYM_ALOHA_GRIPPER_TO_ACT_FN(right_norm)
                print(f"Step {i}: Grippers norm=[{left_norm:.3f}, {right_norm:.3f}] joint=[{left_joint:.4f}, {right_joint:.4f}]")
                
                # Show actual physics positions for physics mode
                if enable_physics_interaction:
                    object_positions = physics.data.qpos[16:22]  # 6 object joints
                    actual_left_left = physics.data.qpos[6]   # left_left_finger actual position
                    actual_left_right = physics.data.qpos[7]  # left_right_finger actual position  
                    actual_right_left = physics.data.qpos[14] # right_left_finger actual position
                    actual_right_right = physics.data.qpos[15] # right_right_finger actual position
                    print(f"       Actual qpos: left=[{actual_left_left:.4f}, {actual_left_right:.4f}] right=[{actual_right_left:.4f}, {actual_right_right:.4f}] Objects={object_positions[:3]}")
    
    # Print final object analysis for physics interaction mode
    if enable_physics_interaction:
        print("\n=== Physics Interaction Analysis ===")
        final_object_positions = physics.data.qpos[16:22]
        initial_positions = [0.0, 0.1, -0.0005886, 0.0, 0.0, 0.0]  # Approximate initial values
        
        print("Object movement analysis:")
        object_names = ['slot', 'stick', 'adverse', 'distractor1', 'distractor2', 'distractor3']
        for i, (name, final, initial) in enumerate(zip(object_names, final_object_positions, initial_positions)):
            movement = abs(final - initial)
            status = "MOVED" if movement > 1e-6 else "static"
            print(f"  {name:12s}: {final:8.6f} (Δ={movement:8.6f}) [{status}]")
    
    return frames

def save_multiview_video(frames_dict, output_dir):
    """Save videos from multiple camera views"""
    if not frames_dict:
        print("No frames to save")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    saved_videos = []
    
    for cam_name, frames in frames_dict.items():
        if not frames:
            print(f"No frames for camera {cam_name}")
            continue
            
        output_path = os.path.join(output_dir, f"insertion_replay_{cam_name}.mp4")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        saved_videos.append(output_path)
        print(f"✓ {cam_name} video saved: {output_path}")
    
    return saved_videos

def create_combined_view(frames_dict, output_path):
    """Create a combined 2x2 view from all 4 cameras"""
    camera_names = ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right']
    
    # Check if we have frames from all cameras
    if not all(cam_name in frames_dict and frames_dict[cam_name] for cam_name in camera_names):
        print("Missing frames from some cameras, skipping combined view")
        return
    
    # Get the minimum number of frames across all cameras
    min_frames = min(len(frames_dict[cam_name]) for cam_name in camera_names)
    
    if min_frames == 0:
        print("No frames available for combined view")
        return
    
    # Create combined video
    frame_height, frame_width = frames_dict[camera_names[0]][0].shape[:2]
    combined_height = frame_height * 2
    combined_width = frame_width * 2
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (combined_width, combined_height))
    
    for i in range(min_frames):
        # Create 2x2 combined frame
        combined_frame = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # Position frames: overhead(top-left), worms_eye(top-right), 
        #                 wrist_left(bottom-left), wrist_right(bottom-right)
        combined_frame[0:frame_height, 0:frame_width] = frames_dict['overhead_cam'][i]
        combined_frame[0:frame_height, frame_width:] = frames_dict['worms_eye_cam'][i]
        combined_frame[frame_height:, 0:frame_width] = frames_dict['wrist_cam_left'][i]
        combined_frame[frame_height:, frame_width:] = frames_dict['wrist_cam_right'][i]
        
        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        out.write(combined_frame_bgr)
    
    out.release()
    print(f"✓ Combined 4-view video saved: {output_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Restore and replay slot insertion episodes')
    parser.add_argument('--episode', '-e', type=int, default=0, 
                       help='Episode number to replay (default: 0)')
    parser.add_argument('--data-dir', '-d', type=str, 
                       default="/home/zzt/actnew/data/converted_with_velocities",
                       help='Directory containing episode files')
    parser.add_argument('--output-dir', '-o', type=str,
                       default="/home/zzt/actnew/insertion_replay_videos", 
                       help='Output directory for videos')
    parser.add_argument('--physics', '-p', action='store_true',
                       help='Enable physics interaction mode (robot interacts with objects)')
    parser.add_argument('--mode', '-m', type=str, choices=['kinematic', 'physics'], 
                       default='kinematic',
                       help='Replay mode: kinematic (direct qpos) or physics (position control)')
    parser.add_argument('--force-open-gripper', action='store_true',
                       help='Force grippers to stay open in physics mode for better object interaction')
    
    args = parser.parse_args()
    
    # Construct episode file path
    episode_filename = f"episode_{args.episode}.hdf5"
    episode_path = os.path.join(args.data_dir, episode_filename)
    
    if not os.path.exists(episode_path):
        print(f"Episode file not found: {episode_path}")
        # List available episodes
        episode_files = [f for f in os.listdir(args.data_dir) if f.endswith('.hdf5')]
        if episode_files:
            print("Available episodes:")
            for f in sorted(episode_files):
                print(f"  {f}")
        return
    
    print(f"Loading episode: {episode_path}")
    
    # Load episode data
    episode_data = load_insertion_episode(episode_path)
    
    # Determine physics interaction mode
    enable_physics = args.physics or (args.mode == 'physics')
    mode_str = "physics interaction" if enable_physics else "kinematic"
    if enable_physics and args.force_open_gripper:
        mode_str += " (forced open grippers)"
    
    # Replay episode using dm_control
    print(f"Starting trajectory replay with dm_control ({mode_str} mode)...")
    frames_dict = replay_episode_dmcontrol(episode_data, save_frames=True, 
                                         enable_physics_interaction=enable_physics,
                                         force_open_gripper=args.force_open_gripper)
    
    if frames_dict is None:
        print("Failed to replay episode")
        return
    
    # Save individual camera videos
    saved_videos = save_multiview_video(frames_dict, args.output_dir)
    
    # Create combined 2x2 view
    combined_output = os.path.join(args.output_dir, f"insertion_replay_episode_{args.episode}_combined_4view.mp4")
    create_combined_view(frames_dict, combined_output)
    
    print("✓ Insertion trajectory replay with dm_control completed successfully!")
    print(f"✓ Videos saved to: {args.output_dir}")

if __name__ == "__main__":
    main()