from typing import List
import os


def get_alfred_root_path():
    assert 'ALFRED_ROOT' in os.environ, "ALFRED_ROOT environment variable not defined!"
    return os.environ['ALFRED_ROOT']


def get_task_traj_data_path(data_split: str, task_id: str) -> str:
    alfred_root = get_alfred_root_path()
    # REALFRED
    traj_data_path = os.path.join(alfred_root, "data", "Re_json_2.1.0", data_split, task_id, "traj_data.json")
    from main.rollout_and_evaluate import switched
    if switched:
        traj_data_path = os.path.join(alfred_root, "data", "Re_json_2.1.0_switched", data_split, task_id, "traj_data.json")
    return traj_data_path


def get_traj_data_paths(data_split: str) -> List[str]:
    alfred_root = get_alfred_root_path()
    # REALFRED
    traj_data_root = os.path.join(alfred_root, "data", "Re_json_2.1.0", data_split)
    from main.rollout_and_evaluate import switched
    if switched:
        traj_data_root = os.path.join(alfred_root, "data", "Re_json_2.1.0_switched", data_split)
    all_tasks = os.listdir(traj_data_root)
    traj_data_paths = []
    for task in all_tasks:
        trials = os.listdir(os.path.join(traj_data_root, task))
        for trial in trials:
            traj_data_paths.append(os.path.join(traj_data_root, task, trial, "traj_data.json"))
    return traj_data_paths


def get_task_dir_path(data_split: str, task_id: str) -> str:
    alfred_root = get_alfred_root_path()
    # REALFRED
    task_dir_path = os.path.join(alfred_root, "data", "Re_json_2.1.0", data_split, task_id.split("/")[0])
    from main.rollout_and_evaluate import switched
    if switched:
        task_dir_path = os.path.join(alfred_root, "data", "Re_json_2.1.0_switched", data_split, task_id.split("/")[0])
    return task_dir_path



def get_splits_path():
    # REALFRED
    splits_path = os.path.join(get_alfred_root_path(), "data", "splits", "oct24.json")
    from main.rollout_and_evaluate import switched
    if switched:
        splits_path = os.path.join(get_alfred_root_path(), "data", "splits", "oct24_val_test_switched.json")
    return splits_path