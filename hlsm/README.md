# HLSM
This is the code repository of the paper [A Persistent Spatial Semantic Representation for High-level Natural Language Instruction Execution](https://arxiv.org/abs/2107.05612) for reproduction in [ReALFRED](https://github.com/snumprlab/realfred).

## Setup

Tested on Ubuntu 20.04.

### Setup Python Environment
```
conda env create -f hlsm-realfred.yml
conda activate hlsm-realfred
```

### Setup Workspace
1. Create a workspace directory somewhere on your system to store models, checkpoints, data, alfred source, etc.
Collecting training data requires ~700GB of space. SSD preferred for faster training.
```
mkdir <workspace dir>
```

2. Update the WS_DIR variable in init.sh to point to `<workspace_dir>`.  

3. Clone ALFRED modified to ai2thor version 4.3.0 into sub-directory `alfred_src` in the workspace:
```
cd <workspace dir>
mkdir alfred_src
cd alfred_src
git clone https://huggingface.co/ch-min-ys/alfred
```

4. Define environment variables. **Do this before running every script in this repo.**
```
source init.sh
```

5. (Optional) Download pre-trained models by cloning them in to `<workspace_dir>`.
```
cd <workspace dir>
git clone https://huggingface.co/ch-min-ys/models
```


### Configuration Files
Most scripts in this repo are parameterized by a single argument that specifies a json
configuration file stored in `lgp/experiment_definitions`, excluding the .json file extension.
To change hyperparameters, datasets, or models, you can modify the existing .json configurations
or create your own. The configuration files support a special @include directive that allows recursively including other
configuration files (useful to share parameters among multiple configurations).

## Collect Training Data
Training data consists of two datasets extracted from oracle rollouts in the ReALFRED environment.
The subgoal dataset consists of examples of semantic maps at the start of each navigation+manipulation sequence
labelled with subgoals. The navigation dataset consists of RGB images, ground-truth depth and segmentation, and
2D affordance feature maps at every timestep labelled with navigation goal poses.

The following command will execute the oracle actions on every training environment in ReALFRED,
and store the data into `<workspace_dir>/data/rollouts`. It will take a few days to run.
Incase the process is interrupted, run it again to resume data collection.
```
python main/collect_universal_rollouts.py alfred/collect_universal_rollouts
```

## Training
Trained models are stored in <workspace_dir>/models.
If you downloaded the models above, you can skip these training steps and proceed to evaluation.

1. Train the high-level controller (subgoal model) for 6 epochs:
```
python main/train_supervised.py alfred/train_subgoal_model
```

2. Train the low-level controller's navigation model for 6 epochs:
```
python main/train_supervised.py alfred/train_navigation_model
```

3. Train the semantic segmentation model for 5 epochs:
```
python main/train_supervised.py alfred/train_segmentation_model
```

4. Train the depth prediction model for 4 epochs:
```
python main/train_supervised.py alfred/train_depth_model
```

Tensorboard summaries are written to `workspace_dir/data/runs`.

## Evaluation
`main/rollout_and_evaluate.py` is the main evaluation script and you can call it with different configurations to evaluate on the different data splits.

We recommend you to use shell script files below made for easier and customizable use.
Set from_idx and to_idx by referring the number of episodes for each split (in the table below).

To evaluate on tests_seen, run:
```
bash eval_tests_seen.sh
```

To evaluate on valid_unseen, run:
```
bash eval_valid_unseen.sh
```

To evaluate on valid_unseen with multi gpu, run:
```
bash eval_valid_unseen_multi.sh
```

Modify the above shell script file according to your conditions.

Explore the other configurations available at `experiment_definitions/alfred/eval`.


**Expected results**:

| Data split      | SR          | GC          | # of episodes |
| --------------- | ----------- | ----------- | ------------- |
| valid_unseen    |  1.08%      |  6.12%      | 1200          |
| valid_seen      |  4.23%      |  9.14%      | 1229          |
| tests_unseen    |  0.49%      |  4.28%      | 1996          |
| tests_seen      |  6.27%      | 10.44%      | 2043          |
