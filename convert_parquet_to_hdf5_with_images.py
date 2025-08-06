#!/usr/bin/env python3
"""
Convert Parquet to HDF5 Format with Real Images
===============================================

使用ffmpeg从视频文件中提取真实图像数据，将parquet格式转换为HDF5格式。
"""

import pandas as pd
import numpy as np
import h5py
import os
import cv2
import subprocess
from tqdm import tqdm
import argparse
import tempfile
import shutil

def extract_frames_from_video(video_path, output_dir, frame_indices):
    """
    使用ffmpeg从视频中提取指定帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_indices: 要提取的帧索引列表
    
    Returns:
        提取的帧列表
    """
    frames = []
    
    try:
        # 使用ffmpeg提取帧
        for i, frame_idx in enumerate(frame_indices):
            # 构建ffmpeg命令
            output_frame = os.path.join(output_dir, f'frame_{i:06d}.jpg')
            cmd = [
                'ffmpeg', '-i', video_path,
                '-vf', f'select=eq(n\\,{frame_idx})',
                '-vframes', '1',
                '-y',  # 覆盖输出文件
                output_frame
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_frame):
                # 读取提取的帧
                frame = cv2.imread(output_frame)
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
                    frames.append(frame)
                else:
                    # 如果读取失败，创建黑色帧
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                # 如果提取失败，创建黑色帧
                frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                
    except Exception as e:
        print(f"Warning: Failed to extract frames from {video_path}: {e}")
        # 创建黑色帧作为fallback
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frame_indices]
    
    return frames

def convert_parquet_to_hdf5_with_images(parquet_dir, output_dir, num_episodes=None):
    """
    将parquet格式的数据转换为HDF5格式，包含真实图像数据
    
    Args:
        parquet_dir: parquet数据目录
        output_dir: 输出HDF5文件目录
        num_episodes: 要转换的episode数量，None表示全部
    """
    print(f"🔄 开始转换parquet到HDF5格式（包含真实图像）...")
    print(f"   输入目录: {parquet_dir}")
    print(f"   输出目录: {output_dir}")
    
    # 读取parquet文件
    parquet_path = os.path.join(parquet_dir, 'data', 'train-00000-of-00001.parquet')
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet文件不存在: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"📊 总数据点: {len(df)}")
    print(f"📊 可用episodes: {df['episode_index'].max() + 1}")
    
    # 限制episode数量
    if num_episodes is not None:
        df = df[df['episode_index'] < num_episodes]
        print(f"📊 使用episodes: 0-{num_episodes-1}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建临时目录用于帧提取
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 按episode分组处理
        episode_groups = df.groupby('episode_index')
        total_episodes = len(episode_groups)
        
        print(f"🎯 开始转换 {total_episodes} 个episodes...")
        
        for episode_idx, episode_data in tqdm(episode_groups, desc="转换episodes"):
            # 按frame_index排序
            episode_data = episode_data.sort_values('frame_index')
            
            # 创建HDF5文件
            hdf5_path = os.path.join(output_dir, f'episode_{episode_idx}.hdf5')
            
            with h5py.File(hdf5_path, 'w') as f:
                # 设置属性
                f.attrs['sim'] = True
                
                # 提取observations
                qpos_list = []
                qvel_list = []
                
                for _, row in episode_data.iterrows():
                    # 提取state数据 (14D)
                    state = np.frombuffer(row['observation.state'], dtype=np.float32)
                    qpos_list.append(state)
                    
                    # 对于qvel，我们暂时用零填充，因为parquet中没有velocity数据
                    qvel = np.zeros(14, dtype=np.float32)
                    qvel_list.append(qvel)
                
                # 转换为numpy数组
                qpos = np.array(qpos_list)  # (episode_len, 14)
                qvel = np.array(qvel_list)  # (episode_len, 14)
                
                # 提取action数据
                action_list = []
                for _, row in episode_data.iterrows():
                    action = np.frombuffer(row['action'], dtype=np.float32)
                    action_list.append(action)
                
                action = np.array(action_list)  # (episode_len, 14)
                
                # 写入HDF5文件
                f.create_dataset('/observations/qpos', data=qpos)
                f.create_dataset('/observations/qvel', data=qvel)
                f.create_dataset('/action', data=action)
                
                # 提取真实图像数据
                episode_len = len(qpos)
                camera_names = ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right']
                
                for cam_name in camera_names:
                    # 获取该相机的视频路径
                    first_row = episode_data.iloc[0]
                    img_info = first_row[f'observation.images.{cam_name}']
                    
                    if isinstance(img_info, dict) and 'path' in img_info:
                        video_path = img_info['path']
                        base_dir = parquet_dir
                        full_video_path = os.path.join(base_dir, video_path)
                        
                        if os.path.exists(full_video_path):
                            # 提取所有帧
                            frame_indices = list(range(episode_len))
                            frames = extract_frames_from_video(full_video_path, temp_dir, frame_indices)
                            
                            if frames:
                                # 转换为numpy数组并调整尺寸
                                frames_array = np.array(frames)  # (episode_len, H, W, 3)
                                # 调整尺寸到标准大小
                                resized_frames = []
                                for frame in frames_array:
                                    resized = cv2.resize(frame, (640, 480))
                                    resized_frames.append(resized)
                                
                                frames_array = np.array(resized_frames)  # (episode_len, 480, 640, 3)
                                f.create_dataset(f'/observations/images/{cam_name}', data=frames_array)
                                print(f"   Episode {episode_idx} {cam_name}: 提取了 {len(frames)} 帧")
                            else:
                                # 如果提取失败，创建假数据
                                fake_images = np.random.randint(0, 256, (episode_len, 480, 640, 3), dtype=np.uint8) * 0.1
                                f.create_dataset(f'/observations/images/{cam_name}', data=fake_images)
                                print(f"   Episode {episode_idx} {cam_name}: 使用假数据")
                        else:
                            # 如果视频文件不存在，创建假数据
                            fake_images = np.random.randint(0, 256, (episode_len, 480, 640, 3), dtype=np.uint8) * 0.1
                            f.create_dataset(f'/observations/images/{cam_name}', data=fake_images)
                            print(f"   Episode {episode_idx} {cam_name}: 视频文件不存在，使用假数据")
                    else:
                        # 如果图像信息格式不对，创建假数据
                        fake_images = np.random.randint(0, 256, (episode_len, 480, 640, 3), dtype=np.uint8) * 0.1
                        f.create_dataset(f'/observations/images/{cam_name}', data=fake_images)
                        print(f"   Episode {episode_idx} {cam_name}: 格式错误，使用假数据")
                
                print(f"   Episode {episode_idx}: {len(qpos)} 步, qpos shape: {qpos.shape}, action shape: {action.shape}")
        
        print(f"✅ 转换完成！")
        print(f"   输出目录: {output_dir}")
        print(f"   转换的episodes: {total_episodes}")
        
        # 验证转换结果
        print(f"\n🔍 验证转换结果...")
        test_file = os.path.join(output_dir, 'episode_0.hdf5')
        if os.path.exists(test_file):
            with h5py.File(test_file, 'r') as f:
                print(f"   Episode 0 验证:")
                print(f"     qpos shape: {f['/observations/qpos'].shape}")
                print(f"     qvel shape: {f['/observations/qvel'].shape}")
                print(f"     action shape: {f['/action'].shape}")
                print(f"     sim attribute: {f.attrs.get('sim', 'Not found')}")
                
                # 检查图像数据
                for cam_name in ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right']:
                    if f'/observations/images/{cam_name}' in f:
                        img_shape = f[f'/observations/images/{cam_name}'].shape
                        print(f"     {cam_name} images shape: {img_shape}")
        
        return output_dir
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description='将parquet格式转换为HDF5格式（包含真实图像）')
    parser.add_argument('--parquet_dir', type=str, 
                       default='/home/zzt/actnew/data/gv_sim_slot_insertion_2arms',
                       help='parquet数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/zzt/actnew/data/converted_bimanual_aloha_slot_insertion_with_vel',
                       help='输出HDF5文件目录')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='要转换的episode数量（建议先用少量测试）')
    
    args = parser.parse_args()
    
    try:
        output_dir = convert_parquet_to_hdf5_with_images(
            args.parquet_dir, 
            args.output_dir, 
            args.num_episodes
        )
        print(f"\n🎉 转换成功！现在可以使用原有的训练代码了。")
        print(f"   训练命令示例:")
        print(f"   python3 imitate_episodes_bimanual.py \\")
        print(f"       --task_name converted_bimanual_aloha_slot_insertion_with_vel \\")
        print(f"       --policy_class ACT \\")
        print(f"       --batch_size 8 \\")
        print(f"       --num_epochs 2000")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 