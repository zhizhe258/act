# Reward设计移植总结

## 概述
从 `/home/zzt/gym_av_aloha/gym_av_aloha/env/task` 中的五个任务移植了完整的reward设计到actnew中。

## 移植的任务

### 1. Slot Insertion (插槽插入)
**文件**: `slot_insertion_env.py` → `BimanualAlohaSlotInsertionTask`

**Reward逻辑**:
- **Level 0**: 初始状态
- **Level 1**: 左右夹爪都接触到stick
- **Level 2**: 左右夹爪都接触stick且stick不接触桌面
- **Level 3**: stick接触slot且不接触桌面
- **Level 4**: pin-stick接触pin-slot (成功插入)

**最大Reward**: 4

### 2. Peg Insertion (销钉插入)
**文件**: `peg_insertion_env.py` → `BimanualAlohaPegInsertionTask`

**Reward逻辑**:
- **Level 0**: 初始状态
- **Level 1**: 左右夹爪都接触到目标
- **Level 2**: 左右夹爪都接触目标且peg和hole都不接触桌面
- **Level 3**: peg接触hole且都不接触桌面
- **Level 4**: peg接触pin (成功插入)

**最大Reward**: 4

### 3. Hook Package (钩子包装)
**文件**: `hook_package_env.py` → `BimanualAlohaHookPackageTask`

**Reward逻辑**:
- **Level 0**: 初始状态
- **Level 1**: 左右夹爪都接触到package
- **Level 2**: 左右夹爪都接触package且package不接触桌面
- **Level 3**: package接触hook且不接触桌面
- **Level 4**: pin-package接触pin-hook (成功钩住)

**最大Reward**: 4

### 4. Pour Test Tube (倒试管)
**文件**: `transfer_tube_env.py` → `BimanualAlohaPourTestTubeTask`

**Reward逻辑**:
- **Level 0**: 初始状态
- **Level 1**: 左右夹爪都接触到试管
- **Level 2**: 左右夹爪都接触试管且试管都不接触桌面
- **Level 3**: ball接触pin (成功倒入)

**最大Reward**: 3

### 5. Thread Needle (穿针)
**文件**: `thread_needle_env.py` → `BimanualAlohaThreadNeedleTask`

**Reward逻辑**:
- **Level 0**: 初始状态
- **Level 1**: 右夹爪接触needle
- **Level 2**: 右夹爪接触needle且needle不接触桌面
- **Level 3**: needle接触wall且不接触桌面
- **Level 4**: needle成功穿过wall (threaded_needle = True)
- **Level 5**: 左夹爪在另一侧接触needle且needle已穿过

**最大Reward**: 5

## 通用设计模式

### Contact检测
所有reward都使用相同的contact检测模式：
```python
contact_pairs = []
for i_contact in range(physics.data.ncon):
    id_geom_1 = physics.data.contact[i_contact].geom1
    id_geom_2 = physics.data.contact[i_contact].geom2
    geom1 = physics.model.id2name(id_geom_1, 'geom')
    geom2 = physics.model.id2name(id_geom_2, 'geom')
    contact_pairs.append((geom1, geom2))
    contact_pairs.append((geom2, geom1))
```

### Reward递增设计
- 每个level都基于前一个level的条件
- 确保任务完成的渐进性
- 避免接触桌面等失败状态

### 特殊处理
- **Thread Needle**: 使用`self.threaded_needle`属性跟踪needle是否已穿过
- **Pour Test Tube**: 只有3个level，因为任务相对简单
- **其他任务**: 都是4个level的标准设计

## 文件修改
- `sim_env.py`: 为所有5个任务实现了完整的reward逻辑
- 保持了与gym_av_aloha完全一致的reward设计
- 确保了reward的连续性和合理性

## 测试状态
✅ 所有reward函数已成功移植并测试通过
✅ 环境可以正常创建和运行
✅ Reward逻辑与原始gym_av_aloha保持一致 