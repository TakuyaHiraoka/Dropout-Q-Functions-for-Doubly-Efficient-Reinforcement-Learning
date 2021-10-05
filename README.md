# What is this?
Source code to replicate experiments provided in ``Dropout Q-Functions for Doubly Efficient Reinforcement Learning.''


**NOTE**

In the following part of this source code, code contained in [1] (MIT license) is used without any major changes: ./code/customenvs

The main part of this source code is implemented by modifying the source code (MIT license) of [2].

[1] https://github.com/JannerM/mbpo/tree/master/mbpo/env

[2] https://github.com/ku2482/soft-actor-critic.pytorch


## Requirements
You can install liblaries using `pip install -r requirements.txt` except `mujoco_py`.

Note that you need a licence to install `mujoco_py`. For installation, please follow instructions [here](https://github.com/openai/mujoco-py).

## Examples
You can train agents as following examples.

### SAC agent
```
python main.py -info sac -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 
```

### REDQ agent
```
Python main.py -info redq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method redq -target_entropy -1.0
```

### Dr.Q agent
```
python main.py -info drq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 7 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
```


## Results

The experimental results (records of returns and estimation errors) are created under the code/runs directory.



