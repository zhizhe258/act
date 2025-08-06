#!/usr/bin/env python3
"""
Convert Parquet to HDF5 Format
==============================

将parquet格式的gym-aloha数据集转换为HDF5格式，以便与现有训练代码兼容。
"""

import pandas as pd
import numpy as np
import h5py
import os
import cv2
from tqdm import tqdm
import argparse

def convert_parquet_to_hdf5(parquet_dir, output_dir, num_episodes=None):
    """
    将parquet格式的数据转换为HDF5格式
    
    Args:
        parquet_dir: parquet数据目录
        output_dir: 输出HDF5文件目录
        num_episodes: 要转换的episode数量，None表示全部
    """
    print(f"🔄 开始转换parquet到HDF5格式...")
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
            
            # 创建假图像数据（用于训练测试）
            episode_len = len(qpos)
            for cam_name in ['overhead_cam', 'worms_eye_cam', 'wrist_cam_left', 'wrist_cam_right']:
                # 为每个相机创建假图像数据
                fake_images = np.random.randint(0, 256, (episode_len, 480, 640, 3), dtype=np.uint8) * 0.1
                f.create_dataset(f'/observations/images/{cam_name}', data=fake_images)
            
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
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='将parquet格式转换为HDF5格式')
    parser.add_argument('--parquet_dir', type=str, 
                       default='/home/zzt/actnew/data/gv_sim_slot_insertion_2arms',
                       help='parquet数据目录')
    parser.add_argument('--output_dir', type=str,
                       default='/home/zzt/actnew/data/converted_bimanual_aloha_slot_insertion_with_vel',
                       help='输出HDF5文件目录')
    parser.add_argument('--num_episodes', type=int, default=50,
                       help='要转换的episode数量')
    
    args = parser.parse_args()
    
    try:
        output_dir = convert_parquet_to_hdf5(
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