import os
import pandas as pd
import argparse
import json
import pickle
import string
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/", help="where to look for the generated data")
parser.add_argument('--split', type=str, default='valid_seen', choices=['train','valid_seen', 'valid_unseen', 'tests_unseen', 'tests_seen'])

# parser.add_argument('-o','--output_name', type=str)
args = parser.parse_args()
data_path = args.data_path 
split = args.split
exclude = set(string.punctuation)
result = dict()

#########################    after voting data   ############################################
if args.split == 'tests_seen' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/tests_seen"  # after voting 
elif args.split == 'tests_unseen' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/tests_unseen"  # after voting 
elif args.split == 'valid_seen' :
    # traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/valid_seen"  # after voting 
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/tests_seen" # after voting switch
elif args.split == 'valid_unseen' :
    # traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/valid_unseen"  # after voting     
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/tests_unseen"  # after voting switch
elif args.split == 'train' :
    traj_data_path = "/media/user/Second_partition/ReALRED/231019_FILM_GTdepth/alfred_data_all/Re_json_2.1.0/train"  # after voting     
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

task_types = {"pick_cool_then_place_in_recep": 0,
              "pick_and_place_with_movable_recep": 1, 
              "pick_and_place_simple": 2, 
              "pick_two_obj_and_place": 3, 
              "pick_heat_then_place_in_recep": 4, 
              "look_at_obj_in_light": 5, 
              "pick_clean_then_place_in_recep": 6}


x_task = dict()

if args.split in ['tests_seen', 'tests_unseen']:
    result['x'] = []; result['x_low'] = []
if args.split in ['train', 'valid_seen', 'valid_unseen']:
    result['x'] = []; result['y'] = []; result['s'] = []; result['mrecep_targets'] = []; result['object_targets'] = []; result['parent_targets'] = []; result['toggle_targets'] = []; result['x_low'] = []

n = 0
d = 0


with open('alfred_data_small/splits/oct24_val_test_switched.json', 'r') as f:
    splits = json.load(f)
tasks = list()
for i in splits[split]:
    task = i["task"]
    if task not in tasks and task not in ['pick_and_place_simple-RemoteControl-None-Bed-216/trial_T20230518_211119_385729'] :
        tasks.append(task)

for task in tasks:
    with open(os.path.join(data_path,task,'pp','ann_0.json')) as f:
        ann_0 = json.load(f)

    # anns = ann_0['turk_annotations']['anns']    # anns = [{"assignment_id", "high_descs", "task_desc"},{},{}]

    ##### Without annotation #####
    # anns = [ann_0['template']]
    anns = ann_0['turk_annotations']['anns']  

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


        ##### without template #####
        # for k in j['high_descs']:

        ##### Without annotation #####
        for k in j['high_descs'][:-1]:
            if k[-1] == '.':
                k = k[:-1]
            k = k.lower()
            k = ''.join(ch for ch in k if ch not in exclude)
            x_low = x_low + k + '[SEP]'
        x_low = x_low[:-5]
        result['x_low'].append(x_low)
        n += 1
        
        
        # Extract from traj_data.json (GT)
        # x_task
        x_task[task_desc] = task

        task = task[-29:]
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
        if traj_data['pddl_params']['toggle_target'] != "":
            params['toggle_target'] = traj_data['pddl_params']['toggle_target']
        else:
            params['toggle_target'] = None
        
        desc_to_gt_params[task_desc] = params
        
        if args.split in ['train', 'valid_seen', 'valid_unseen']:
            # obj2idx_new_split, recep2idx_new_split
            import pickle
            obj2idx = pickle.load(open('models/instructions_processed_LP/BERT/data/alfred_data/alfred_dicts/obj2idx_new_split.p', 'rb'))
            recep2idx = pickle.load(open('models/instructions_processed_LP/BERT/data/alfred_data/alfred_dicts/recep2idx_new_split.p', 'rb'))
            toggle2idx = pickle.load(open('models/instructions_processed_LP/BERT/data/alfred_data/alfred_dicts/toggle2idx.p', 'rb'))
            
            result['y'].append(task_type)
            
            if traj_data['pddl_params']['object_sliced']:
                result['s'].append(1)
            else:
                result['s'].append(0)
            
            if traj_data['pddl_params']['mrecep_target'] != "":
                result['mrecep_targets'].append(obj2idx[traj_data['pddl_params']['mrecep_target']])
            else:
                result['mrecep_targets'].append(obj2idx[None])                

            result['object_targets'].append(obj2idx[traj_data['pddl_params']['object_target']])
            
            if traj_data['pddl_params']['parent_target'] != "":
                result['parent_targets'].append(recep2idx[traj_data['pddl_params']['parent_target']])
            else:
                result['parent_targets'].append(recep2idx[None])
                
            if traj_data['pddl_params']['toggle_target'] != "":
                result['toggle_targets'].append(toggle2idx[traj_data['pddl_params']['toggle_target']])
            else:
                result['toggle_targets'].append(toggle2idx[None])


# pickle.dump(result, open(args.output_name + ".p", "wb"))
# pickle.dump(result, open(args.split + '_text_with_ppdl_low_appended_GT_oct14_131.p', 'wb'))
# pickle.dump(x_task, open(args.split + '_task_desc_to_task_id_GT_oct14_131.p', 'wb'))
# pickle.dump(desc_to_gt_params, open( args.split + '_new_split_GT_oct14_131_debug_t.p', 'wb'))


pickle.dump(result, open(args.split + '_text_with_ppdl_low_appended_GT_after_voting.p', 'wb'))
pickle.dump(x_task, open(args.split + '_task_desc_to_task_id_GT_after_voting.p', 'wb'))
pickle.dump(desc_to_gt_params, open( args.split + '_new_split_GT_after_voting_debug_t.p', 'wb'))



print(f"num of annotations: {n}")
print(f"num of duplication: {d}")
print(f"num of not duplication: {n-d}")