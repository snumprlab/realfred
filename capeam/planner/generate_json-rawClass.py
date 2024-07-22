import os
import json
import torch
import constants
from tqdm import tqdm


root = 'data/Re_json_2.1.0'
splits = ['train', 'valid_seen', 'valid_unseen', 'tests_seen', 'tests_unseen']

res = {
    k: {
        'MetaController': [],
        'GotoLocation': [],
        'PickupObject': [],
        'PutObject': [],
        'CoolObject': [],
        'HeatObject': [],
        'CleanObject': [],
        'SliceObject': [],
        'ToggleObject': [],
    } for k in splits
}


classes_big = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
classes_small = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
classes_small = [o.lower() for o in classes_small]


def get_triplet(pddl_params, high_task, low_actions):
    action = high_task['discrete_action']['action']
    target = '0'
    receptacle = '0'

    if action in ['GotoLocation']:
        try:
            target = classes_small.index(high_task['discrete_action']['args'][0])
            target = classes_big[target]
        except Exception as e:
            print(e)
            target = 0

    elif action in ['PickupObject']:
        target = high_task['planner_action']['objectId'].split('|')
        target = target[-1] if len(target) > 4 else target[0]
        target = target.split('_')[0] if 'Slice' in target else target
        if 'coordinateReceptacleObjectId' in high_task['planner_action']:
            receptacle = high_task['planner_action']['coordinateReceptacleObjectId'][0]


    elif action in ['PutObject']:
        target = high_task['planner_action']['objectId'].split('|')
        target = target[-1] if len(target) > 4 else target[0]
        target = target.split('_')[0] if 'Slice' in target else target
        receptacle = high_task['planner_action']['receptacleObjectId'].split('|')[0]

    elif action in ['CoolObject']:
        for low_action in low_actions:
            if low_action['api_action']['action'] == 'PutObject':
                target = low_action['api_action']['objectId'].split('|')
                target = target[-1] if len(target) > 4 else target[0]
                target = target.split('_')[0] if 'Slice' in target else target
                break
        if pddl_params['object_sliced']:
            target = pddl_params['object_target'] + 'Sliced'
        else:
            target = pddl_params['object_target']
        receptacle = 'Fridge'


    elif action in ['HeatObject']:

        for low_action in low_actions:
            if low_action['api_action']['action'] == 'PutObject':
                target = low_action['api_action']['objectId'].split('|')
                target = target[-1] if len(target) > 4 else target[0]
                target = target.split('_')[0] if 'Slice' in target else target
                break
        if pddl_params['object_sliced']:
            target = pddl_params['object_target'] + 'Sliced'
        else:
            target = pddl_params['object_target']
        receptacle = 'Microwave'

    elif action in ['CleanObject']:
        # format check
        action_template = [
            'PutObject', 'ToggleObjectOn', 'ToggleObjectOff', 'PickupObject'
        ]
        cs = []
        cs.append(len(low_actions) == len(action_template))
        for i in range(len(low_actions)):
            cs.append(low_actions[i]['api_action']['action'] == action_template[i])
        if not all(cs):
            print('inconsistent format in {}: {}'.format(action, cs))
            exit(0)
        if pddl_params['object_sliced']:
            target = pddl_params['object_target'] + 'Sliced'
        else:
            target = pddl_params['object_target']
        receptacle = "Sink"

    elif action in ['SliceObject']:
        target = high_task['planner_action']['objectId'].split('|')
        target = target[-1] if len(target) > 4 else target[0]
        target = target.split('_')[0] if 'Slice' in target else target
        if 'coordinateReceptacleObjectId' in high_task['planner_action']:
            receptacle = high_task['planner_action']['coordinateReceptacleObjectId'][0]
    elif action in ['ToggleObject']:
        # format check
        action_template = [
            'ToggleObjectOn',
        ]
        cs = []
        cs.append(len(low_actions) == len(action_template))
        for i in range(len(low_actions)):
            cs.append(low_actions[i]['api_action']['action'] == action_template[i])
        if not all(cs):
            print('inconsistent format in {}: {}'.format(action, cs))
            exit(0)

        target = low_actions[0]['api_action']['objectId'].split('|')[0]

    for k in ['mrecep_target', 'object_target', 'toggle_target']:
        if target == pddl_params[k] :
            target = k
        if target == pddl_params[k]+'Sliced':
            target = k+'_sliced'

    return action, target, receptacle

tasks = json.load(open('data/splits/oct24.json', 'r'))
for split in splits:
    tasks_split = tasks[split]

    for t in tqdm(tasks_split):
        task_path = os.path.join(root, t['task'], 'pp', 'ann_{}.json'.format(t['repeat_idx']))
        task = json.load(open(task_path, 'r'))

        high_actions = []

        if 'test' not in split:
            # low level action extraction
            subgoal_indices = [p['high_idx'] for p in task['plan']['high_pddl']]
            actions = [p for p in task['plan']['low_actions']]
            for subgoal_idx in subgoal_indices:
                # high actions
                high_action = task['plan']['high_pddl'][subgoal_idx]['discrete_action']['action']

                # low actions
                low_actions = [a for a in actions if a['high_idx'] == subgoal_idx]

                action, target, receptacle = get_triplet(
                    task['pddl_params'],
                    task['plan']['high_pddl'][subgoal_idx],
                    low_actions
                )

                if action == 'SliceObject':
                    targetId = task['plan']['high_pddl'][subgoal_idx]['planner_action']['objectId']
                    for i, h_action in enumerate(task['plan']['high_pddl']):
                        if 'coordinateReceptacleObjectId' not in h_action['planner_action']:
                            continue
                        if i < subgoal_idx and h_action['planner_action']['objectId'] == targetId:
                            receptacle = h_action['planner_action']['coordinateReceptacleObjectId'][0]
                        if i > subgoal_idx and targetId in h_action['planner_action']['objectId']:
                            receptacle = h_action['planner_action']['coordinateReceptacleObjectId'][0]
                            break
                        
                if receptacle == task['pddl_params']['parent_target'] and action == 'PutObject' and 'target' in target:
                    receptacle = 'parent_target'
                    try:
                        pIdx = high_actions[-1].index(task['pddl_params']['parent_target'])
                        high_actions[-1][pIdx] = 'parent_target'
                    except:
                        pass 
                high_actions.append([action, target, receptacle])

                if high_action in ['GotoLocation']:
                    _actions = [a['discrete_action']['action'] for a in low_actions] + ['<<stop>>']
                    res[split][high_action].append({
                        'root': task['root'],
                        'repeat_idx': task['ann']['repeat_idx'],
                        'high_idx': subgoal_idx,
                        'triplets': high_actions[-1],
                        'actions': _actions,
                    })

                elif high_action in ['NoOp']:
                    continue

                else:
                    _actions = [a['discrete_action']['action'] for a in low_actions] + ['<<stop>>']
                    _classes = []
                    for a in low_actions:
                        if a['api_action']['action'] == 'PutObject':
                            _low_class = a['api_action']['objectId'].split('|')[0]
                        else:
                            _low_class = a['api_action']['objectId'].split('|')
                            _low_class = _low_class[-1] if len(_low_class) > 4 else _low_class[0]
                            _low_class = _low_class.split('_')[0] if 'Slice' in _low_class else _low_class
                
                        _classes.append(_low_class)
                    _classes.append('0')

                    res[split][high_action].append({
                        'root': task['root'],
                        'repeat_idx': task['ann']['repeat_idx'],
                        'high_idx': subgoal_idx,
                        'triplets': high_actions[-1],
                        'actions': _actions,
                        'classes': _classes,
                    })

            res[split]['MetaController'].append({
                'root': task['root'],
                'repeat_idx': task['ann']['repeat_idx'],
                'lang_goal': task['num']['lang_goal'],
                'lang_instr': task['num']['lang_instr'],
                'triplets': high_actions,
            })
        else:
            res[split]['MetaController'].append({
                'root': task['root'],
                'repeat_idx': task['ann']['repeat_idx'],
                'lang_goal': task['num']['lang_goal'],
                'lang_instr': task['num']['lang_instr'],
            })

with open('data/data-raw.json'.format(split), 'w') as f:
    json.dump(res, f, indent=4)
