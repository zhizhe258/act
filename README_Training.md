# Bimanual ALOHA Training Guide

这个指南介绍如何使用优化的训练脚本来训练bimanual ALOHA任务，无需MuJoCo依赖。

## 🎯 新的训练架构

### 文件结构
- `imitate_episodes_bimanual.py` - 纯训练脚本（无MuJoCo依赖）
- `train_bimanual.py` - 训练启动器（预定义配置）
- `evaluate_bimanual.py` - 单独的评估脚本
- `README_Training.md` - 本指南

## 🚀 快速开始

### 1. 使用预定义配置训练

```bash
# 训练ACT策略在slot insertion任务上
python3 train_bimanual.py --config slot_insertion_act

# 训练RTC_Improved策略
python3 train_bimanual.py --config slot_insertion_rtc

# 启用temporal aggregation
python3 train_bimanual.py --config slot_insertion_act --temporal_agg

# 预览命令但不执行
python3 train_bimanual.py --config slot_insertion_act --dry_run
```

### 2. 手动训练配置

```bash
python3 imitate_episodes_bimanual.py \
    --task_name converted_bimanual_aloha_slot_insertion_with_vel \
    --policy_class ACT \
    --batch_size 8 \
    --seed 0 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --backbone resnet18
```

### 3. 评估训练好的模型

```bash
python3 evaluate_bimanual.py \
    --ckpt_dir /path/to/checkpoint/directory \
    --task_name converted_bimanual_aloha_slot_insertion_with_vel \
    --policy_class ACT \
    --num_rollouts 50 \
    --save_video
```

## 📋 支持的任务和配置

### 支持的任务
- `converted_bimanual_aloha_slot_insertion_with_vel` - 推荐使用（包含真实速度）
- `bimanual_aloha_slot_insertion`
- `bimanual_aloha_cube_transfer`
- `bimanual_aloha_peg_insertion`
- `bimanual_aloha_color_cubes`
- `bimanual_aloha_hook_package`
- `bimanual_aloha_pour_test_tube`
- `bimanual_aloha_thread_needle`

### 支持的策略
- `ACT` - Action Chunking Transformer
- `RTC_Improved` - RTC 改进版本  
- `CNNMLP` - CNN+MLP基线

### 预定义配置
- `slot_insertion_act` - ACT策略用于slot insertion
- `slot_insertion_rtc` - RTC策略用于slot insertion
- `cube_transfer_act` - ACT策略用于cube transfer
- `peg_insertion_act` - ACT策略用于peg insertion

## 🔧 主要改进

### 1. 训练时无MuJoCo依赖
```python
# ❌ 旧版本 - 不必要的依赖
from sim_env import BOX_POSE
from constants import PUPPET_GRIPPER_JOINT_OPEN

# ✅ 新版本 - 只导入必需的模块
from utils import load_data, compute_dict_mean, set_seed
from policy import ACTPolicy, CNNMLPPolicy
```

### 2. 正确的状态维度配置
```python
# ❌ 旧版本 - 硬编码
state_dim = 14

# ✅ 新版本 - 任务特定配置
task_config = get_task_config(task_name)
state_dim = task_config['state_dim']  # 动态获取
action_dim = task_config['action_dim']
```

### 3. 清晰的训练/评估分离
- **训练阶段**: 完全不依赖MuJoCo环境
- **评估阶段**: 单独脚本处理环境交互

### 4. 数据维度验证
```python
# 训练前验证数据维度
if qpos_data.shape[-1] != state_dim:
    raise ValueError(f"QPos dimension mismatch: got {qpos_data.shape[-1]}, expected {state_dim}")
```

## 📊 训练监控

### 自动生成的输出
- `checkpoints/` - 模型检查点
- `dataset_stats.pkl` - 数据集统计信息
- `train_val_*.png` - 训练曲线
- `policy_best.ckpt` - 最佳模型

### 训练进度示例
```
🤖 Bimanual ALOHA Training Configuration:
   Task: converted_bimanual_aloha_slot_insertion_with_vel
   State dimension: 14
   Action dimension: 14
   Dataset: /home/zzt/actnew/data/converted_with_velocities
   Episodes: 50
   Episode length: 301

📊 Training for 2000 epochs...
📈 Epoch 0
   Val loss:   0.12345
   Train loss: 0.12678
```

## ⚡ 性能优化建议

### 训练参数调优
- **Batch size**: 根据GPU内存调整（建议8-16）
- **Learning rate**: ACT通常用1e-5，RTC可以尝试更高
- **Chunk size**: 100是一个好的起点
- **Backbone**: resnet18速度快，resnet50准确性高

### 数据准备
1. 确保数据集已正确转换为14D格式
2. 检查gripper数据是否在[0,1]范围内
3. 验证相机图像数据质量

## 🐛 常见问题

### 问题：维度不匹配错误
```
ValueError: QPos dimension mismatch: got 16, expected 14
```
**解决方案**: 使用转换后的数据集（`converted_bimanual_aloha_slot_insertion_with_vel`）

### 问题：CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 减小batch_size，例如从8改为4

### 问题：收敛缓慢
**解决方案**: 
- 检查数据质量
- 调整学习率
- 增加训练epoch数
- 尝试不同的backbone

## 📝 最佳实践

1. **使用预定义配置开始**: 先用`train_bimanual.py`的预设配置
2. **监控训练曲线**: 观察loss是否正常下降
3. **定期评估**: 每100-200个epoch评估一次模型性能
4. **保存检查点**: 训练长时间任务时定期保存
5. **数据验证**: 训练前确保数据格式正确

## 🎯 下一步

1. 训练完成后使用`evaluate_bimanual.py`评估模型
2. 如果需要，可以继续在`restore_insertion_dmcontrol.py`中测试trained policy
3. 根据评估结果调整超参数重新训练