# ACT: Action Chunking with Transformers



### Repo Structure
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

Here need to rename the mian folder as actnew.
### Installation
    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    pip install pandas
    cd actnew/detr && pip install -e .


### Example Usages

To set up a new terminal, run:

    conda activate aloha
    cd <path to act repo>

### Data download and convertion
First download data from https://github.com/Soltanilara/av-aloha.git

To convert data to HDF5:
    
    python3 convert_parquet_to_hdf5.py \
    --base_data_dir /path/to/your/data \
    --num_episodes 50




## Converted Tasks

- `converted_bimanual_aloha_hook_package`
- `converted_bimanual_aloha_thread_needle`
- `converted_bimanual_aloha_peg_insertion`
- `converted_bimanual_aloha_slot_insertion`
- `converted_bimanual_aloha_cube_transfer`



To train ACT:
    
    # converted_bimanual_aloha_slot_insertion
    python3 imitate_episodes.py \
    --task_name converted_bimanual_aloha_slot_insertion \
    --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 \
    --seed 0


To evaluate the policy, run the same command but add ``--eval``. This loads the best validation checkpoint.
To enable temporal ensembling, add flag ``--temporal_agg``.
Videos will be saved to ``<ckpt_dir>`` for each rollout.
You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

