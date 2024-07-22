import os
import pandas as pd
import argparse
import json
import pickle
import string
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_1.3.2", help="where to look for the generated data")
parser.add_argument('--split', type=str, default='train', choices=['train','valid_seen', 'valid_unseen', 'tests_unseen', 'tests_seen'])
# parser.add_argument('-o','--output_name', type=str)

args = parser.parse_args()
data_path = args.data_path
split = args.split
exclude = set(string.punctuation)
result = dict()


#########################    Before voting data   ############################################
if args.split == 'tests_seen' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_1.3.2/tests_seen"  # before voting 
elif args.split == 'tests_unseen' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_1.3.2/tests_unseen"  # before voting 
elif args.split == 'valid_seen' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_1.3.2/valid_seen"  # before voting 
elif args.split == 'valid_unseen' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_1.3.2/valid_unseen"  # before voting     
elif args.split == 'train' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_1.3.2/train"  # before voting     
#######################################################################################################
    


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

task_types = {"pick_cool_then_place_in_recep":      0, 
              "pick_and_place_with_movable_recep":  1, 
              "pick_and_place_simple":              2, 
              "pick_two_obj_and_place":             3, 
              "pick_heat_then_place_in_recep":      4, 
              "look_at_obj_in_light":               5,
              "pick_clean_then_place_in_recep":     6}

# 'task_desc': task
# {'wash the brown vegetable and put it on the counter’: 'trial_T20230514_192232_882070',
# 'wash the brown vegetable and put it on the counter’: 'trial_T20230514_192232_882070', ...}
x_task = dict()

result['x'] = []; result['x_low'] = []
n = 0
d = 0

with open('alfred_data_small/splits/oct14_131_debug.json', 'r') as f:

    splits = json.load(f)
tasks = list()
for i in splits[split]:
    task = i["task"]
    if task not in tasks and task not in ['pick_and_place_simple-RemoteControl-None-Bed-216/trial_T20230518_211119_385729']: # error dataset
        tasks.append(task)
        
for task in tasks:   
    # print(task)
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
            # print(j['high_descs']) ##
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
        # path = task_to_path[task]
        path = task_to_path[task.split('/')[-1]]
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
        
        
        
## argument for training 
pickle.dump(result, open(args.split + '_realfred_appended.p', 'wb'))
## argument GT
pickle.dump(desc_to_gt_params, open(args.split +'_realfred_appended_GT.p', 'wb'))
print(f"num of annotations: {n}")
print(f"num of duplication: {d}")
print(f"num of not duplication: {n-d}")