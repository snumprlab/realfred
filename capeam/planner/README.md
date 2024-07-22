# Context-aware planner
This README provides step-by-step instructions for training the meta planner and both the sub-policy planners as part of our implementation in [ReALFRED](https://github.com/snumprlab/realfred).

## Preprocess
### If you followed instruction provided [here](htttps;?/)
```
ln -s  $CAPEAM/alfred_data_all/Re_json_2.1.0 data/Re_json_2.1.0
ln -s  $CAPEAM/alfred_data_all/splits data/splits
```

Them run `preprocess.sh`
```
bash preprocess.sh
```
This will create `data.json` and `data-raw.json` under data
```
planner/               # This directory
│
├── README.md          # Overview and setup instructions
├── data/
│   ├── Re_json_2.1.0/ 
│   │   ├ train/
│   │   └ ...
│   ├── splits/
│   │   └ oct24.json
│   ├── data.json       # Created after running preproces.sh   
│   └── data-raw.json   # Created after running preproces.sh   
│
...
```

## Train
Training is separated into two major components:
1. **Meta Planner**
2. **Subpolicy Planners**

### Training the Meta Planner
To train the Meta planner, you only need to run the following command in the terminal:

```sh
python main_meta.py
```

This will process the training data and save the model weights to a `weight/MetaController/`.

### Training the Subpolicy Planners
We have multiple Subpolicy planners, each responsible for a different task. To train each of these planners, run the corresponding command:

```sh
python main_subpolicy.py --subgoal CleanObject --rawClass
python main_subpolicy.py --subgoal CoolObject --rawClass
python main_subpolicy.py --subgoal HeatObject --rawClass
python main_subpolicy.py --subgoal PickupObject --rawClass
python main_subpolicy.py --subgoal PutObject --rawClass
python main_subpolicy.py --subgoal SliceObject --rawClass
python main_subpolicy.py --subgoal ToggleObject --rawClass
```

Each command will train a specific sub-policy model on the designated task and save the trained model to a sub-directory within the `weight/subpolicy/` directory created earlier.

## Inference

Once training is complete, inference can be performed using the trained weights of both the meta planner and the sub-policy planners.

```sh
python main_hierarchical.py --metaWeight /path/to/meta/weight --subWeight /path/to/subpolicy/weights
```

Replace `/path/to/meta/weight` and `/path/to/subpolicy/weights` with the actual paths to the trained meta planner and sub-policy weights, respectively.


## Results

The output from the inference process (i.e., the generated plans) will be saved automatically in the `results/` directory. You can use them by editting `read_test_dict` in `$CAPEAM/models/instructions_processed_LP/ALFRED_task_helper.py` to use your files.
