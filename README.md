# ACT Baseline for RH20T Dataset
Example for training ACT on RH20T dataset. This project is built on top of [ACT](https://github.com/tonyzhaozh/act/tree/main).

## Installation

```bash
git clone https://github.com/rh20t/act_baseline.git
cd act_baseline
pip install -r requirements.txt
```

## Loading data with cleaned data list
Sometimes you may encounter data loading errors due to NaN annotations in a few amount of scenes. We recommand loading data with ``rh20t_cleaned_data.json`` to reduce program errors and accelerate data processing.

For each scene, this file contains a subset of the original video, which is implemented by removing redundant frames according to their tcp differences. The selected frames is saved as timestamps like "1629432627550".

The structure in the file is as follows:
```bash
{
    "RH20T_cfg1": {
        "task_0001_user_0001_scene_0001_cfg_0001": {
            "cam_035622060973": [
                1629432627550,
                1629432629589,
                1629432629779,
                1629432630013,
                1629432630132,
                1629432630233,
                ...
            ],
            "cam_038522062547": [
                1629432627038,
                1629432629519,
                1629432629791,
                1629432629904,
                1629432630136,
                1629432630260,
                ...
            ],
            ...
        },
        "task_0001_user_0001_scene_0002_cfg_0001": {
            ...
        },
        ...
    },
    "RH20T_cfg2": {
        ...
    },
    ...
    "RH20T_cfg7": {
        ...
    }
}

```

Download ``rh20t_cleaned_data.json`` [here]() and put it under [dataset](dataset) folder. The usage is shown in [dataset/rh20t.py](dataset/rh20t.py).

## Pretrain on RH20T
Train the ACT model on RH20T by running the following command:
```bash
python train.py --task_name [TASK_NAME] --ckpt_dir [YOUR_LOG_DIR] --dataset_root [RH20T_DIR] --batch_size 24 --seed 233 --num_epoch 50 --save_epoch 5 --lr 1e-5 --kl_weight 10 --chunk_size 20 --hidden_dim 512 --dim_feedforward 3200
```
``[TASK_NAME]`` is the task ids in RH20T, like "task_0001". You can modify this variable to change the task to train on. The file [command_pretrain.sh](command_pretrain.sh) shows a more detailed example.

## Finetune on your own data
To conduct real robot experiments in your environment, a finetune step is highly recommanded. You can modify the dataset code and training code, and fintune the pretrained model on your own data. **Note that action trunking step is required during inference, and you can refer to the original [ACT implementation](https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py#L251-L259).**