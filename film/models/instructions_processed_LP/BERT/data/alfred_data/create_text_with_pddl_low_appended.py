import os
import pandas as pd
import argparse
import json
import pickle
import string
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/media/user/data/FILM/alfred_data_all/json_new", help="where to look for the generated data")
parser.add_argument('--split', type=str, default="tests_unseen")
parser.add_argument('-o','--output_name', type=str)
args = parser.parse_args()
data_path = args.data_path
split = args.split
exclude = set(string.punctuation)
result = dict()
traj_data_path = "/media/user/data/alfred_4.3.0/gen/dataset/Finished"
traj_data_path = "/media/user/data/FILM/alfred_data_all/json_2.1.0/valid_unseen"
task_to_path = dict()
desc_to_gt_params = dict()     # 'wash the brown vegetable and put it on the counter’: 
JSON_FILENAME = "traj_data.json"
for dir_name, _, _ in os.walk(traj_data_path):
    if "trial_" in dir_name and (not "raw_images" in dir_name) and (not "pddl_states" in dir_name) and (not "video" in dir_name):
        json_file = os.path.join(dir_name, JSON_FILENAME)
        if not os.path.isfile(json_file):
            continue
        task = json_file.split('/')[-2]
        task_to_path[task] = json_file
# print(task_to_path)

task_types = {"pick_cool_then_place_in_recep": 0, "pick_and_place_with_movable_recep": 1, "pick_and_place_simple": 2, "pick_two_obj_and_place": 3, "pick_heat_then_place_in_recep": 4, "look_at_obj_in_light": 5, "pick_clean_then_place_in_recep": 6}

# 'task_desc': task
# {'wash the brown vegetable and put it on the counter’: 'trial_T20230514_192232_882070',
# 'wash the brown vegetable and put it on the counter’: 'trial_T20230514_192232_882070', ...}
x_task = dict()

result['x'] = []; result['x_low'] = []
n = 0
d = 0
# with open('../../../../../alfred_data_small/splits/REALFRED_splits.json', 'r') as f:
# with open('alfred_data_small/splits/REALFRED_splits.json', 'r') as f:
with open('../../../../../alfred_data_small/splits/oct21.json', 'r') as f:


    splits = json.load(f)
tasks = list()
for i in splits[split]:
    task = i["task"]
    if task not in tasks:
        tasks.append(task)
        
for task in tasks:   
    with open(os.path.join(data_path,task,'pp','ann_0.json')) as f:
        ann_0 = json.load(f)
    
    anns = ann_0['turk_annotations']['anns']    # anns = [{"assignment_id", "high_descs", "task_desc"},{},{}]
    for j in anns:
        task_desc = j['task_desc']
        if task_desc[-1] == '.':
            task_desc = task_desc[:-1]
        task_desc = task_desc.lower()
        task_desc = ''.join(ch for ch in task_desc if ch not in exclude)
        if task_desc in result['x']:
            print(task, task_desc)
            d += 1        
        result['x'].append(task_desc)
        
        x_low = ''
        for k in j['high_descs']:
            if k[-1] == '.':
                k = k[:-1]
            k = k.lower()
            k = ''.join(ch for ch in k if ch not in exclude)
            x_low = x_low + k + '[SEP]'
        x_low = x_low[:-6]
        result['x_low'].append(x_low)
        n += 1
        
        # x_task
        x_task[task_desc] = task
        path = task_to_path[task]
        with open(path) as f:
            traj_data = json.load(f)
        params = dict()
        task_type = task_types[traj_data['task_type']]
        params['task_type'] = task_type
        if traj_data['pddl_params']['mrecep_target'] != "":
            params['mrecep_target'] = traj_data['pddl_params']['mrecep_target']
        else:
            params['mrecep_target'] = None
        params['object_target'] = traj_data['pddl_params']['object_target']
        if traj_data['pddl_params']['parent_target'] != "":
            params['parent_target'] = traj_data['pddl_params']['parent_target']
        else:
            params['parent_target'] = None
        if traj_data['pddl_params']['object_sliced']:
            params['sliced'] = 1
        else:
            params['sliced'] = 0
                    
        desc_to_gt_params[task_desc] = params
        
        
        
        
# pickle.dump(result, open(args.output_name + ".p", "wb"))
pickle.dump(result, open(args.split + '_text_with_ppdl_low_appended.p', 'wb'))
# pickle.dump(x_task, open(args.split + '_task_desc_to_task_id.p', 'wb'))
pickle.dump(desc_to_gt_params, open('../../../instruction2_params_test_unseen_noappended_GT.p', 'wb'))
print(f"num of annotations: {n}")
print(f"num of duplication: {d}")
print(f"num of not duplication: {n-d}")