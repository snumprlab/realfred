import os
import json
import torch
import numpy as np
import constants
import collections
from tqdm import trange, tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pickle
import string

class Dataset_MetaController:

    def __init__(self, root='data', split='train', batch_size=16, appended=True, factorize=True, nohier=False):
        self.root = root
        self.split = split
        self.batch_size = batch_size
        self.pad = 0
        self.vocab = torch.load(root+'/pp.vocab')
        self.appended = appended
        self.factorize = factorize
        self.nohier = nohier

        self.ACTIONS = [
            'GotoLocation',
            'PickupObject', 'PutObject',
            'CoolObject', 'HeatObject', 'CleanObject',
            'SliceObject', 'ToggleObject',
            '<<stop>>',
        ]
        if nohier:
            self.ACTIONS.extend(['OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff'])
        self.TARGETS = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        if factorize:
            self.TARGETS.extend(['mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced'])

        if nohier:
            data_path = os.path.join(root, 'data-nohier.json')
        else:
            if factorize:
                data_path = os.path.join(root, 'data.json')
            else:
                data_path = os.path.join(root, 'data-raw.json')
        
        self.trajectories = json.load(open(data_path, 'r'))[split]['MetaController']
        print(split, 'trajectories loaded:', len(self.trajectories))

    def featurize(self, batch):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda')
        feat = collections.defaultdict(list)

        def _get_param(root, idx):
            traj_path = os.path.join(root, 'pp', 'ann_%d.json'%idx)
            traj_data = json.load(open(traj_path, 'r'))
            return traj_data['pddl_params']

        def _find_slice_recep(root, idx):
            traj_path = os.path.join(root, 'pp', 'ann_%d.json'%idx)
            traj_data = json.load(open(traj_path, 'r'))
            slice_idx = [i for i, a in enumerate(traj_data['plan']['high_pddl']) if a['discrete_action']['action'] == 'SliceObject'][0]
            objectId = traj_data['plan']['high_pddl'][slice_idx]['planner_action']['objectId']
            receptacle = '0'
            for i, h_action in enumerate(traj_data['plan']['high_pddl']):
                if 'coordinateReceptacleObjectId' not in h_action['planner_action']:
                    continue
                if i < slice_idx and h_action['planner_action']['objectId'] == objectId:
                    receptacle = h_action['planner_action']['coordinateReceptacleObjectId'][0]
                if i > slice_idx and objectId in h_action['planner_action']['objectId']:
                    receptacle = h_action['planner_action']['coordinateReceptacleObjectId'][0]
                    break
            if receptacle in constants.OPENABLE_CLASS_SET - {'Box'}:
                return receptacle
            else:
                return '0'

        for ex in batch:
            feat['root'].append(ex['root'])

            # language (input)
            feat['lang_goal'].append(ex['lang_goal'])
            if self.appended:
                feat['lang_instr'].append([])
                for instr in ex['lang_instr']:
                    feat['lang_instr'][-1] = feat['lang_instr'][-1] + instr

            feat['goal_natural'].append([self.vocab['word'].index2word(w) for w in feat['lang_goal'][-1]])

            if 'test' in self.split:
                continue
                
            param = _get_param(ex['root'], ex['repeat_idx'])

            # triplet encoding
            actions    , actions_mask     = [], []
            targets    , targets_mask     = [], []
            receptacles, receptacles_mask = [], []
            for t, triplet in enumerate(ex['triplets']):
                action, target, receptacle = triplet
                
                if action == 'GotoLocation':
                    continue

                if action == 'NoOp':
                    actions.append(self.ACTIONS.index('<<stop>>'))
                    actions_mask.append(1)
                    targets.append(0)
                    targets_mask.append(0)
                    receptacles.append(0)
                    receptacles_mask.append(0)
                    break

                # Change object name to pddl parameter key
                if self.factorize:
                    if action == 'SliceObject':
                        receptacle = _find_slice_recep(ex['root'], ex['repeat_idx'])
                        

                    for k in ['mrecep_target', 'object_target', 'toggle_target']:
                        if target == param[k]:
                            target = k
                        if target == param[k]+'Sliced':
                            target = k+'_sliced'
                        if receptacle == param[k]:
                            receptacle = k

                    if action == 'PutObject' and 145 - 1 < targets[-1]:
                        if receptacle == param['parent_target']:
                            receptacle = 'parent_target'
                        if target == param['parent_target']:
                            target = 'parent_target'

                actions.append(self.ACTIONS.index(action))
                actions_mask.append(1)
                targets.append(self.TARGETS.index(target))
                targets_mask.append(1)
                if self.nohier:
                    receptacles.append(0)
                    receptacles_mask.append(0)
                else:
                    receptacles.append(self.TARGETS.index(receptacle))
                    receptacles_mask.append(1)
            
            feat['actions'].append(actions)
            feat['actions_mask'].append(actions_mask)
            feat['targets'].append(targets)
            feat['targets_mask'].append(targets_mask)
            feat['receptacles'].append(receptacles)
            feat['receptacles_mask'].append(receptacles_mask)

        for k, v in feat.items():
            if k in {'root', 'lang_goal', 'lang_instr', 'goal_natural'}:
                continue
            seqs = [torch.tensor(vv, device=device) for vv in v]
            seqs = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
            feat[k] = seqs

        return feat

    def iterate(self):
        for i in trange(0, len(self.trajectories), self.batch_size):
            batch = self.trajectories[i:i+self.batch_size]
            feat = self.featurize(batch)
            yield feat



class Dataset_Manipulator:

    def __init__(self, root='data', split='train', subgoal=None, batch_size=16, factorize=True):
        self.root = root
        self.split = split
        self.subgoal = subgoal
        self.batch_size = batch_size
        self.pad = 0
        self.factorize = factorize

        self.vocab = torch.load(root+'/pp.vocab')
        self.TRIPLETS = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        if factorize:
            self.TRIPLETS.extend([ 'mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced' ])
        self.TRIPLETS.extend(
            [
                'GotoLocation',
                'PickupObject', 'PutObject',
                'CoolObject', 'HeatObject', 'CleanObject',
                'SliceObject', 'ToggleObject',
            ]
        )

        self.LOW_ACTIONS = self.vocab['action_low'].to_dict()['index2word']
        self.LOW_CLASSES = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        if factorize:
            self.LOW_CLASSES.extend([ 'mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced' ])

        if factorize:
            data_path = os.path.join(root, 'data.json')
        else:
            data_path = os.path.join(root, 'data-raw.json')
            
        self.trajectories = json.load(open(data_path, 'r'))[split][subgoal]
        for _subgoal in tqdm(subgoal.split(','), desc='Data Loading'):
           self.trajectories = self.trajectories + json.load(open(data_path, 'r'))[split][_subgoal]


    def featurize(self, batch):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda')
        feat = collections.defaultdict(list)

        for ex in batch:
            feat['root'].append(ex['root'])

            feat['lang_triplets'].append([
                self.TRIPLETS.index(ex['triplets'][0]),
                self.TRIPLETS.index(ex['triplets'][1]),
                self.TRIPLETS.index(ex['triplets'][2]) if ex['triplets'][2] is not None else 0,
            ])

            actions = [self.LOW_ACTIONS.index(a) for a in ex['actions']]
            actions = actions + [self.LOW_ACTIONS.index('<<stop>>')]
            actions_mask = [1] * len(actions)

            classes = [self.LOW_CLASSES.index(c) for c in ex['classes']] + [0]
            classes_mask = [1] * len(classes)
            classes_mask[-1] = 0

            feat['low_actions'].append(actions)
            feat['low_actions_mask'].append(actions_mask)
            feat['low_classes'].append(classes)
            feat['low_classes_mask'].append(classes_mask)

        for k, v in feat.items():
            if k in {'root'}:
                continue
            seqs = [torch.tensor(vv, device=device) for vv in v]
            seqs = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
            feat[k] = seqs

        return feat

    def iterate(self):
        for i in trange(0, len(self.trajectories), self.batch_size):
            batch = self.trajectories[i:i+self.batch_size]
            feat = self.featurize(batch)
            yield feat



def get_triplet(high_task, low_actions):
    action = high_task['discrete_action']['action']
    target = '0'
    receptacle = '0'

    if action in ['GotoLocation']:
        pass

    elif action in ['PickupObject']:
        pattern = ','.join([a['api_action']['action'] for a in low_actions])
        if pattern == 'CloseObject,PickupObject':
            for low_action in low_actions:
                if low_action['api_action']['action'] == 'PickupObject':
                    target = low_action['api_action']['objectId'].split('|')
                    target = target[-1] if len(target) > 4 else target[0]
                    target = target.split('_')[0] if 'Slice' in target else target
                    break
        else:
            for low_action in low_actions:
                if low_action['api_action']['action'] == 'PickupObject':
                    target = low_action['api_action']['objectId'].split('|')
                    target = target[-1] if len(target) > 4 else target[0]
                    target = target.split('_')[0] if 'Slice' in target else target
                    break

            for low_action in low_actions:
                if low_action['api_action']['action'] in ['OpenObject', 'CloseObject']:
                    receptacle = low_action['api_action']['objectId'].split('|')
                    receptacle = receptacle[-1] if len(receptacle) > 4 else receptacle[0]
                    receptacle = receptacle.split('_')[0] if 'Slice' in receptacle else receptacle
                    break

    elif action in ['PutObject']:
        for low_action in low_actions:
            if low_action['api_action']['action'] == 'PutObject':
                target = low_action['api_action']['objectId'].split('|')
                target = target[-1] if len(target) > 4 else target[0]
                target = target.split('_')[0] if 'Slice' in target else target
                break

        for low_action in low_actions:
            if low_action['api_action']['action'] in ['OpenObject', 'CloseObject']:
                receptacle = low_action['api_action']['objectId'].split('|')
                receptacle = receptacle[-1] if len(receptacle) > 4 else receptacle[0]
                receptacle = receptacle.split('_')[0] if 'Slice' in receptacle else receptacle
                break

    elif action in ['CoolObject']:
        for low_action in low_actions:
            if low_action['api_action']['action'] == 'PutObject':
                target = low_action['api_action']['objectId'].split('|')
                target = target[-1] if len(target) > 4 else target[0]
                target = target.split('_')[0] if 'Slice' in target else target
                break

        for low_action in low_actions:
            if low_action['api_action']['action'] in ['OpenObject', 'CloseObject']:
                receptacle = low_action['api_action']['objectId'].split('|')
                receptacle = receptacle[-1] if len(receptacle) > 4 else receptacle[0]
                receptacle = receptacle.split('_')[0] if 'Slice' in receptacle else receptacle
                break

    elif action in ['HeatObject']:
        for low_action in low_actions:
            if low_action['api_action']['action'] == 'PutObject':
                target = low_action['api_action']['objectId'].split('|')
                target = target[-1] if len(target) > 4 else target[0]
                target = target.split('_')[0] if 'Slice' in target else target
                break
        receptacle = 'Microwave'

    elif action in ['CleanObject']:
        target = low_actions[0]['api_action']['objectId'].split('|')
        target = target[-1] if len(target) > 4 else target[0]
        target = target.split('_')[0] if 'Slice' in target else target

        receptacle = low_actions[1]['api_action']['objectId'].split('|')[0]

    elif action in ['SliceObject']:
        for low_action in low_actions:
            if low_action['api_action']['action'] in ['SliceObject']:
                target = low_action['api_action']['objectId'].split('|')
                target = target[-1] if len(target) > 4 else target[0]
                target = target.split('_')[0] if 'Slice' in target else target
                break

        for low_action in low_actions:
            if low_action['api_action']['action'] in ['OpenObject', 'CloseObject']:
                receptacle = low_action['api_action']['objectId'].split('|')
                receptacle = receptacle[-1] if len(receptacle) > 4 else receptacle[0]
                receptacle = receptacle.split('_')[0] if 'Slice' in receptacle else receptacle
                break

    elif action in ['ToggleObject']:
        target = low_actions[0]['api_action']['objectId'].split('|')
        target = target[-1] if len(target) > 4 else target[0]
        target = target.split('_')[0] if 'Slice' in target else target

    return action, target, receptacle



class Dataset_HierarchicalAgent:

    def __init__(self, root='data/Re_json_2.1.0', split='train', batch_size=16, appended=True):
        self.root = root
        self.split = split
        self.batch_size = batch_size
        self.pad = 0
        self.appended = appended

        self.vocab = torch.load(root+'/pp.vocab')
        self.ACTIONS = [
            'GotoLocation',
            'PickupObject', 'PutObject',
            'CoolObject', 'HeatObject', 'CleanObject',
            'SliceObject', 'ToggleObject',
            '<<stop>>',
        ]
        self.TRIPLETS = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        + [
            'GotoLocation',
            'PickupObject', 'PutObject',
            'CoolObject', 'HeatObject', 'CleanObject',
            'SliceObject', 'ToggleObject',
        ]
        self.LOW_ACTIONS = self.vocab['action_low'].to_dict()['index2word']
        self.LOW_CLASSES = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"
        ]

        tasks = json.load(open('data/splits/oct24.json', 'r'))[split]
        self.trajectories = []
        for t in tasks:
            path = os.path.join(root, t['task'], 'pp', 'ann_{}.json'.format(t['repeat_idx']))
            self.trajectories.append(json.load(open(path, 'r')))

    def featurize(self, batch):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda')
        feat = collections.defaultdict(list)

        for ex in batch:
            feat['root'].append(ex['root'])

            repeat_idx = ex['ann']['repeat_idx']
            feat['repeat_idx'].append(repeat_idx)
            feat['goal_natural'].append(ex['turk_annotations']['anns'][repeat_idx]['task_desc'])
            if self.appended:
                feat['instr_natural'].append(ex['turk_annotations']['anns'][repeat_idx]['high_descs'])

            # language (input)
            feat['lang_goal'].append(ex['num']['lang_goal'])
            if self.appended:
                feat['lang_instr'].append([])
                for instr in ex['num']['lang_instr']:
                    feat['lang_instr'][-1] = feat['lang_instr'][-1] + instr

            if self.split in ['valid_seen', 'valid_unseen']:
                # GT high actions
                triplets = []
                subgoal_indices = [p['high_idx'] for p in ex['plan']['high_pddl']]
                actions = [p for p in ex['plan']['low_actions']]
                for subgoal_idx in subgoal_indices:
                    action, target, receptacle = get_triplet(
                        ex['plan']['high_pddl'][subgoal_idx],
                        [a for a in actions if a['high_idx'] == subgoal_idx],
                    )
                    triplets.append((action, target, receptacle))
                feat['triplets'].append(triplets)

                # GT low actions
                low_actions = [self.LOW_ACTIONS.index(a['discrete_action']['action']) for a in ex['plan']['low_actions']]
                feat['low_actions'].append(low_actions)

                # GT low classes
                nav_actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'LookDown', 'LookUp']
                low_classes = []
                low_classes_mask = []
                for a in ex['plan']['low_actions']:
                    if a['api_action']['action'] in nav_actions:
                        low_class = 0
                        low_class_mask = 0
                    else:
                        if a['api_action']['action'] in ['PutObject']:
                            low_class = a['api_action']['objectId'].split('|')[0]
                            low_class = self.LOW_CLASSES.index(low_class)
                        else:
                            low_class = a['api_action']['objectId'].split('|')
                            low_class = low_class[-1] if len(low_class) > 4 else low_class[0]
                            low_class = low_class.split('_')[0] if 'Slice' in low_class else low_class
                            low_class = self.LOW_CLASSES.index(low_class)
                        low_class_mask = 1

                    low_classes.append(low_class)
                    low_classes_mask.append(low_class_mask)
                feat['low_classes'].append(low_classes)
                feat['low_classes_mask'].append(low_classes_mask)

        return feat

    def iterate(self):
        for i in trange(0, len(self.trajectories), self.batch_size):
            batch = self.trajectories[i:i+self.batch_size]
            feat = self.featurize(batch)
            yield feat
