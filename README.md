# What is this?
Source code to replicate experiments provided in [``Dropout Q-Functions for Doubly Efficient Reinforcement Learning.''](https://openreview.net/forum?id=xCVJMsPv3RT)


**NOTE**

In the following part of this source code, code contained in [1] (MIT license) is used without any major changes: ./KUCodebase/code/customenvs

The main part of this source code is implemented by modifying the source code (MIT license) of [2] and [3].

[1] https://github.com/JannerM/mbpo/tree/master/mbpo/env

[2] https://github.com/ku2482/soft-actor-critic.pytorch

[3] https://github.com/watchernyu/REDQ

## Requirements
You can install libraries using `pip install -r requirements.txt` except `mujoco_py`.

Note that you need a licence to install `mujoco_py`. For installation, please follow instructions [here](https://github.com/openai/mujoco-py).

## How to use?
### First, you need to choose codebase:
If you want to use agents implemented on the top of [KU codebase](https://github.com/ku2482/soft-actor-critic.pytorch). 
```
cd KUCodebase/code
```
If you want to use agents implemented on the top of [REDQ original codebase](https://github.com/watchernyu/REDQ), 
```
cd OriginalREDQCodebase
```


### Then, you can train agents as following examples.
#### SAC agent
```
python main.py -info sac -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 
```

#### REDQ agent
```
Python main.py -info redq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0
```

#### Droq agent (also called Dr.Q agent in the first version of my Arxiv and ICLR papers)
```
python main.py -info drq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 7 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
```

## Results

The experimental results (records of returns and estimation errors) are created under ``runs'' directory.

## Citation
If you use this repo or find it useful, please consider citing:
```
@inproceedings{hiraoka2022dropout,
title={Dropout Q-Functions for Doubly Efficient Reinforcement Learning},
author={Takuya Hiraoka and Takahisa Imagawa and Taisei Hashimoto and Takashi Onishi and Yoshimasa Tsuruoka},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=xCVJMsPv3RT}
}
```

## Tips
1: DroQ is sensitive especially to target entropy (target_entropy) and (target_drop_rate). 

If you use KU codebase, you can reproduce the results of my paper by setting target_entropy and target_drop_rate as follows.
|Environment| target_entropy | target_dropout_rate |
| --------- | -------------- | ------------------- |
| Hopper    | -1.0           | 0.01                | 
| Walker2d  | -3.0           | 0.01                | 
| Ant       | -4.0           | 0.01                | 
| Humanoid  | -2.0           | 0.01                | 

If you use REDQ original codebase, you can reproduce the results of my paper by setting target_drop_rate as follows (the value for target entropy is automatically assigned in core.py). 
|Environment| target_dropout_rate |
| --------- | ------------------- |
| Hopper    | 0.0001              | 
| Walker2d  | 0.005               | 
| Ant       | 0.01                | 
| Humanoid  | 0.1                 | 

2: Overall, methods implemented on REDQ original codebase work better than KU codebase counterparts. 

## Update
2022/01/15 Add agents implemented on the top of REDQ original codebase. 

2022/03/08 Add citation and tips.


