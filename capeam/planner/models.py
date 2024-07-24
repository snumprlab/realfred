import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import constants
import pickle
import string

class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class ScaledDotAttn(nn.Module):
    def __init__(self, dim_key_in=1024, dim_key_out=128, dim_query_in=1024 ,dim_query_out=128):
        super().__init__()
        self.fc_key = nn.Linear(dim_key_in, dim_key_out)
        self.fc_query = nn.Linear(dim_query_in, dim_query_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, value, h): # key: lang_feat_t_instr, query: h_tm1_instr
        key = F.relu(self.fc_key(value))
        query = F.relu(self.fc_query(h)).unsqueeze(-1)

        scale_1 = np.sqrt(key.shape[-1])
        scaled_dot_product = torch.bmm(key, query) / scale_1
        softmax = self.softmax(scaled_dot_product)
        element_wise_product = value*softmax
        weighted_lang_t_instr = torch.sum(element_wise_product, dim=1)

        return weighted_lang_t_instr, softmax.squeeze(-1)



# Meta Controller, LSTM
class MetaController_LSTM(nn.Module):
    def __init__(self, dhid=1024, demb=100, appended=True, factorize=True, large=False, nohier=False):
        super(MetaController_LSTM, self).__init__()

        self.dhid = dhid
        self.demb = demb

        self.appended = appended
        self.factorize = factorize
        self.nohier = nohier

        # Embeddings
        self.vocab = torch.load('data/pp.vocab')
        self.emb_word = nn.Embedding(len(self.vocab['word']), demb)

        # Language encoder
        self.lang_dropout = nn.Dropout(0.2, inplace=True)

        self.language_encoder_goal = nn.LSTM(demb, dhid//2, bidirectional=True, batch_first=True)
        self.language_attn_goal = SelfAttn(dhid)
        self.scale_dot_attn_goal = ScaledDotAttn(dhid, 128, dhid, 128)

        if appended:
            self.language_encoder_instr = nn.LSTM(demb, dhid//2, bidirectional=True, batch_first=True)
            self.language_attn_instr = SelfAttn(dhid)
            self.scale_dot_attn_instr = ScaledDotAttn(dhid, 128, dhid, 128)
            lang_input_size = dhid*2
        else:
            lang_input_size = dhid
        # high-level plan decoder

        self.decoder = nn.LSTMCell(lang_input_size if appended else dhid, dhid)
        if factorize:
            num_objs = 145 +5

        else:
            num_objs = 145
        if nohier:
            num_actions = 9+4
        else:
            num_actions = 9
        self.decoder_action = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(inplace=True),
            nn.Linear(dhid//2, num_actions),
        )
        self.decoder_receptacle = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(inplace=True),
            nn.Linear(dhid//2, num_objs),
        )
        self.decoder_target = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(inplace=True),
            nn.Linear(dhid//2, num_objs),
        )
        
    def step(self, enc_lang_goal=None, enc_lang_instr=None, state_prev=None):
        # language attention
        weighted_lang_t_goal, _ = self.scale_dot_attn_goal(enc_lang_goal, state_prev[0])
        if self.appended:
            weighted_lang_t_instr, _ = self.scale_dot_attn_instr(enc_lang_instr, state_prev[0])
            inp = torch.cat([weighted_lang_t_goal, weighted_lang_t_instr], dim=1)
        else:
            inp = weighted_lang_t_goal

        # decoder update
        state_t = self.decoder(inp, state_prev)

        # high-level action
        action_t = self.decoder_action(state_t[0])

        # high-level target receptacle
        receptacle_t = self.decoder_receptacle(state_t[0])

        # high-level target object
        target_t = self.decoder_target(state_t[0])

        return action_t, receptacle_t, target_t, state_t

    def encode_lang(self, feat, goal=False):
        # language embedding
        v = feat['lang_goal'] if goal else feat['lang_instr']
        seqs = [torch.tensor(vv, device=torch.device('cuda')) for vv in v]
        pad_seq = pad_sequence(seqs, batch_first=True, padding_value=0)
        seq_lengths = np.array(list(map(len, v)))
        embed_seq = self.emb_word(pad_seq)
        packed_input = pack_padded_sequence(
            embed_seq, seq_lengths, batch_first=True, enforce_sorted=False
        )
        emb_lang = packed_input

        # language encoding
        if goal:
            self.lang_dropout(emb_lang.data)
            enc_lang, _ = self.language_encoder_goal(emb_lang)
            enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
            self.lang_dropout(enc_lang)
            cont_lang = self.language_attn_goal(enc_lang)
        else:
            self.lang_dropout(emb_lang.data)
            enc_lang, _ = self.language_encoder_instr(emb_lang)
            enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
            self.lang_dropout(enc_lang)
            cont_lang = self.language_attn_instr(enc_lang)

        return cont_lang, enc_lang


    def forward(self, feat):
        # encode language
        cont_lang_goal, enc_lang_goal = self.encode_lang(feat, goal=True)
        if self.appended:
            cont_lang_instr, enc_lang_instr = self.encode_lang(feat, goal=False)

        # initialize hidden state
        state_prev = cont_lang_goal, torch.zeros_like(cont_lang_goal)

        # inference
        actions = []
        receptacles = []
        targets = []
        slices = []
        states = []
        last_action = np.zeros((16, ))
        try:
            for t in range(feat['actions'].size(1)):
                action_t, receptacle_t, target_t, state_t = \
                    self.step(
                        enc_lang_goal=enc_lang_goal,
                        enc_lang_instr=enc_lang_instr \
                            if self.appended else None,
                        state_prev=state_prev)
                actions.append(action_t)
                receptacles.append(receptacle_t)
                targets.append(target_t)
                states.append(state_t[0])
                state_prev = state_t
        except:
            while (last_action != 8).any():
                if len(actions) > 30:
                    break
                action_t, receptacle_t, target_t, state_t = \
                    self.step(
                        enc_lang_goal=enc_lang_goal,
                        enc_lang_instr=enc_lang_instr \
                            if self.appended else None,
                        state_prev=state_prev)
                actions.append(action_t)
                last_action = np.argmax(action_t.cpu(), axis=1)
                receptacles.append(receptacle_t)
                targets.append(target_t)
                states.append(state_t[0])
                state_prev = state_t

        return {
            'out_actions': torch.stack(actions, dim=1),
            'out_receptacles': torch.stack(receptacles, dim=1),
            'out_targets': torch.stack(targets, dim=1),
            'out_states': torch.stack(states, dim=1),
        }


    def inference(self, feat):
        # encode language
        cont_lang_goal, enc_lang_goal = self.encode_lang(feat, goal=True)
        if self.appended:
            cont_lang_instr, enc_lang_instr = self.encode_lang(feat, goal=False)

        # initialize hidden state
        state_prev = cont_lang_goal, torch.zeros_like(cont_lang_goal)

        # triplets
        triplets = []
        for b in range(cont_lang_goal.size(0)):
            triplet = []
            _state_prev = state_prev[0][b:b+1], state_prev[1][b:b+1]
            for _ in range(50):
                action_t, receptacle_t, target_t, state_t = \
                    self.step(
                        enc_lang_goal=enc_lang_goal[b:b+1],
                        enc_lang_instr=enc_lang_instr[b:b+1] \
                            if self.appended else None,
                        state_prev=_state_prev
                    )
                _state_prev = state_t

                triplet.append((
                    action_t[0].argmax().item(),
                    target_t[0].argmax().item(),
                    receptacle_t[0].argmax().item(),
                ))

                if triplet[-1][0] == 8:
                    break

            triplets.append(triplet)

        return triplets
    
class Manipulator(nn.Module):

    def __init__(self, dhid=1024, demb=100, factorize=False):
        super(Manipulator, self).__init__()

        self.TRIPLETS = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        if factorize:
            self.TRIPLETS.extend([ 'mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced' ])
        self.TRIPLETS.extend([
            'GotoLocation',
            'PickupObject', 'PutObject',
            'CoolObject', 'HeatObject', 'CleanObject',
            'SliceObject', 'ToggleObject',
        ])
        self.emb_triplet = nn.Embedding(len(self.TRIPLETS), demb)

        # triplet encoder (high-action, target, receptacle)
        self.language_encoder = nn.LSTM(demb, dhid//2, bidirectional=True, batch_first=True)
        self.language_attn = SelfAttn(dhid)
        self.scale_dot_attn = ScaledDotAttn(dhid, 128, dhid, 128)

        # low-level decoder
        num_objs = 145+5 if factorize else 145
        self.lang_dropout = nn.Dropout(0.2, inplace=True)
        self.decoder = nn.LSTMCell(dhid*3, dhid)
        self.decoder_action = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(inplace=True),
            nn.Linear(dhid//2, 15),
        )
        self.decoder_class = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(inplace=True),
            nn.Linear(dhid//2, num_objs),
        )

    def encode_lang(self, feat):
        # language embedding
        v = feat['lang_triplets']
        seqs = [torch.tensor(vv, device=torch.device('cuda')) for vv in v]
        pad_seq = pad_sequence(seqs, batch_first=True, padding_value=0)
        seq_lengths = np.array(list(map(len, v)))
        embed_seq = self.emb_triplet(pad_seq)
        packed_input = pack_padded_sequence(
            embed_seq, seq_lengths, batch_first=True, enforce_sorted=False
        )
        emb_lang = packed_input

        # language encoding
        enc_lang, _ = self.language_encoder(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        cont_lang = self.language_attn(enc_lang)

        return cont_lang, enc_lang

    def step(self, enc_triplet=None, state_prev=None):
        # language attention
        weighted_enc_triplet_t = torch.cat([enc_triplet[:,i] for i in range(3)], dim=1)

        inp = weighted_enc_triplet_t

        # decoder update
        state_t = self.decoder(inp, state_prev)

        # high-level action
        action_t = self.decoder_action(state_t[0])

        # high-level target object
        class_t = self.decoder_class(state_t[0])

        return action_t, class_t, state_t

    def forward(self, feat):
        # encode language
        cont_triplet, enc_triplet = self.encode_lang(feat)

        # initialize hidden state
        state_prev = cont_triplet, torch.zeros_like(cont_triplet)

        # inference
        actions = []
        classes = []
        states = []
        for t in range(feat['low_actions'].size(1)):
            action_t, class_t, state_t = \
                self.step(
                    enc_triplet=enc_triplet,
                    state_prev=state_prev)
            actions.append(action_t)
            classes.append(class_t)
            states.append(state_t[0])
            state_prev = state_t

        return {
            'out_low_actions': torch.stack(actions, dim=1),
            'out_low_classes': torch.stack(classes, dim=1),
            'out_states': torch.stack(states, dim=1),
        }


    def inference(self, feat):
        # encode language
        cont_triplet, enc_triplet = self.encode_lang(feat)

        # initialize hidden state
        state_prev = cont_triplet, torch.zeros_like(cont_triplet)

        # inference
        low_actions = []
        low_classes = []
        states = []

        for b in range(cont_triplet.size(0)):
            actions = []
            classes = []
            _state_prev = state_prev[0][b:b+1], state_prev[1][b:b+1]
            while True:
                action_t, class_t, state_t = \
                    self.step(
                        enc_triplet=enc_triplet[b:b+1],
                        state_prev=_state_prev
                    )
                _state_prev = state_t

                if action_t[0].argmax().item() == 2 or len(actions) > 100:
                    break

                actions.append(action_t[0].argmax().item())
                classes.append(class_t[0].argmax().item())

            low_actions.append(actions)
            low_classes.append(classes)

        return low_actions, low_classes

class HierarchicalAgent(nn.Module):
    def __init__(self, meta_weight, sub_weight, factorize_meta=True, factorize_subpolicy=False, large=False):
        super(HierarchicalAgent, self).__init__()

        self.factorize_meta = factorize_meta
        self.factorize_subpolicy = factorize_subpolicy

        self.vocab = torch.load('data/pp.vocab')
        self.ACTIONS = [
            'GotoLocation',
            'PickupObject', 'PutObject',
            'CoolObject', 'HeatObject', 'CleanObject',
            'SliceObject', 'ToggleObject',
            '<<stop>>',
        ]
        self.TRIPLETS = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        if factorize_subpolicy:
            self.TRIPLETS.extend([ 'mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced' ])
        self.TRIPLETS.extend([
            'GotoLocation',
            'PickupObject', 'PutObject',
            'CoolObject', 'HeatObject', 'CleanObject',
            'SliceObject', 'ToggleObject',
        ])
        self.LOW_ACTIONS = self.vocab['action_low'].to_dict()['index2word']
        self.LOW_CLASSES = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
        if factorize_subpolicy:
            self.LOW_CLASSES.extend([ 'mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced' ])

        # Meta Controller
        meta_controller_path = f'weight/MetaController/{meta_weight}/best_unseen_acc_actions.pth'
        print(f"Loading MetaController from {meta_controller_path}")
        if large:
            dhid = 1024*2
            demb = 100
        else:
            dhid = 1024
            demb = 100
        self.meta_controller = MetaController_LSTM(dhid=dhid, demb=demb, factorize=(self.factorize_meta))
        self.meta_controller.load_state_dict(
            torch.load(meta_controller_path)
        )

        self.goto = None
        if large:
            dhid = 1024*2
            demb = 100*2
        else:
            dhid = 1024
            demb = 100
        weight_name = 'best_unseen_acc_classes.pth'
        pickup_path = os.path.join('weight', 'PickupObject', sub_weight, weight_name)
        print(f"Loading MetaController from {pickup_path}")
        self.pickup = Manipulator(demb=demb, dhid=dhid, factorize=(self.factorize_subpolicy))
        self.pickup.load_state_dict(
            torch.load(pickup_path)
        )
        self.pickup.eval()

        put_path = os.path.join('weight', 'PutObject', sub_weight, weight_name) 
        print(f"Loading MetaController from {put_path}")
        self.put = Manipulator(dhid=dhid, demb=demb, factorize=(self.factorize_subpolicy))
        self.put.load_state_dict(
            torch.load(put_path)
        )
        self.put.eval()

        cool_path = os.path.join('weight', 'CoolObject', sub_weight, weight_name) 
        print(f"Loading MetaController from {cool_path}")
        self.cool = Manipulator(demb=demb, dhid=dhid, factorize=(self.factorize_subpolicy))
        self.cool.load_state_dict(
            torch.load(cool_path)
        )
        self.cool.eval()

        heat_path = os.path.join('weight', 'HeatObject', sub_weight, weight_name) 
        print(f"Loading MetaController from {heat_path}")
        self.heat = Manipulator(demb=demb, dhid=dhid, factorize=(self.factorize_subpolicy))
        self.heat.load_state_dict(
            torch.load(heat_path)
        )
        self.heat.eval()

        clean_path = os.path.join('weight', 'CleanObject', sub_weight, weight_name) 
        print(f"Loading MetaController from {clean_path}")
        self.clean = Manipulator(demb=demb, dhid=dhid, factorize=(self.factorize_subpolicy))
        self.clean.load_state_dict(
            torch.load(clean_path)
        )
        self.clean.eval()

        slice_path = os.path.join('weight', 'SliceObject', sub_weight, weight_name) 
        print(f"Loading MetaController from {slice_path}")
        self.slice = Manipulator(demb=demb, dhid=dhid, factorize=(self.factorize_subpolicy))
        self.slice.load_state_dict(
            torch.load(slice_path)
        )
        self.slice.eval()

        toggle_path = os.path.join('weight', 'ToggleObject', sub_weight, weight_name) 
        print(f"Loading MetaController from {toggle_path}")
        self.toggle = Manipulator(demb=demb, dhid=dhid, factorize=(self.factorize_subpolicy))
        self.toggle.load_state_dict(
            torch.load(toggle_path)
        )
        self.toggle.eval()


    def inference(self, feat):
        def preprocess(s):
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = s.strip()
            s = s.lower()
            return s
            
        def get_param(goal_natural):
            meta_valid_seen = pickle.load(open('FILM_lang/intruct2params/instruction2_params_valid_seen_appended_new_split_oct24.p', 'rb'))
            meta_valid_unseen = pickle.load(open('FILM_lang/intruct2params/instruction2_params_valid_unseen_appended_new_split_oct24.p', 'rb'))
            meta_test_seen = pickle.load(open('FILM_lang/intruct2params/instruction2_params_tests_seen_appended_new_split_oct24.p', 'rb'))
            meta_test_unseen = pickle.load(open('FILM_lang/intruct2params/instruction2_params_tests_unseen_appended_new_split_oct24.p', 'rb'))
            for meta_dict in [meta_valid_seen, meta_valid_unseen, meta_test_seen, meta_test_unseen]:
                for goal, param in meta_dict.items():
                    goal = goal.split('[SEP]')[0]
                    if preprocess(goal) == preprocess(goal_natural):
                        return param
            raise Exception('Goal Not Found')

        def token2idx(token_idx, goal):
            if token_idx < 145:
                return self.TRIPLETS.index(self.LOW_CLASSES[token_idx])

            keys = ['mrecep_target', 'object_target', 'parent_target', 'toggle_target', 'object_target_sliced']
            key = keys[token_idx-145]

            if key == 'toggle_target':
                return self.TRIPLETS.index('FloorLamp')
            
            param = get_param(goal)
            if key == 'object_target_sliced':
                try:
                    obj = param['object_target']+'Sliced'
                    self.TRIPLETS.index(obj)
                except:
                    # ex) EggSliced
                    obj = param['object_target']
            elif param[key] == None:
                obj = '0'
            else:
                obj = param[key]
            return self.TRIPLETS.index(obj)
            
        out_meta = self.meta_controller.inference(feat)



        nets = {
            'GotoLocation': self.goto,
            'PickupObject': self.pickup,
            'PutObject': self.put,
            'CoolObject': self.cool,
            'HeatObject': self.heat,
            'CleanObject': self.clean,
            'SliceObject': self.slice,
            'ToggleObject': self.toggle,
        }

        high_triplets = []
        low_actions = []
        low_classes = []
        high_idxs = []
        for b in range(len(out_meta)):
            _triplets = []
            _high_idxs = []
            _low_actions = []
            _low_classes = []
            #print('Meta Controller')
            for n, triplet in enumerate(out_meta[b]):

                if self.ACTIONS[triplet[0]] == 'GotoLocation':
                    continue
                elif self.ACTIONS[triplet[0]] == '<<stop>>':
                    break

                subpolicy = nets[self.ACTIONS[triplet[0]]]

                # embedding translation (meta -> subpolicy)
                # replace pddl parameter key to FILM predicted parameter
                _triplet = list(triplet)
                _triplet[0] = self.TRIPLETS.index(self.ACTIONS[triplet[0]])
                if not self.factorize_meta or self.factorize_subpolicy:
                    _triplet[1] = self.TRIPLETS.index(self.LOW_CLASSES[triplet[1]])
                    _triplet[2] = self.TRIPLETS.index(self.LOW_CLASSES[triplet[2]])
                else:
                    _triplet[1] = token2idx(triplet[1], feat['goal_natural'][b])
                    _triplet[2] = token2idx(triplet[2], feat['goal_natural'][b])
                _triplet = tuple(_triplet)

                with torch.no_grad():
                    out_low_actions, out_low_classes = subpolicy.inference({
                        'lang_triplets': [_triplet]
                    })
                _triplets.append(_triplet)
                _high_idxs = _high_idxs + [n]*len(out_low_actions[0])
                _low_actions = _low_actions + out_low_actions[0]
                _low_classes = _low_classes + out_low_classes[0]
            
            high_triplets.append(_triplets)
            low_actions.append(_low_actions)
            low_classes.append(_low_classes)
            high_idxs.append(_high_idxs)

        return {
            'out_triplets': high_triplets,
            'out_low_actions': low_actions,
            'out_low_classes': low_classes,
            'out_h_idxs': high_idxs,
        }

    def inference_at_eval(self, feat):
        out_meta = self.meta_controller.inference(feat)[0]

        nets = {
            'GotoLocation': self.goto,
            'PickupObject': self.pickup,
            'PutObject': self.put,
            'CoolObject': self.cool,
            'HeatObject': self.heat,
            'CleanObject': self.clean,
            'SliceObject': self.slice,
            'ToggleObject': self.toggle,
        }

        res = [] #  [ [(a1, c1), (a2, c2), ...], [...], ... ]
        for n, triplet in enumerate(out_meta):
            _res = []

            if self.ACTIONS[triplet[0]] == 'GotoLocation':
                high_action, subgoal_idx, actions = feat['gt_actions'][n]
                if high_action != 'GotoLocation':
                    print(high_action)
                    print('Inconsistent')
                    exit(0)
                _res = actions
            elif self.ACTIONS[triplet[0]] == '<<stop>>':
                _res = [['<<stop>>', '0']]
            else:
                subpolicy = nets[self.ACTIONS[triplet[0]]]

                # embedding translation (meta -> subpolicy)
                _triplet = list(triplet)
                _triplet[0] = self.TRIPLETS.index(self.ACTIONS[triplet[0]])
                _triplet[1] = self.TRIPLETS.index(self.LOW_CLASSES[triplet[1]])
                _triplet[2] = self.TRIPLETS.index(self.LOW_CLASSES[triplet[2]])
                _triplet = tuple(_triplet)

                with torch.no_grad():
                    out_low_actions, out_low_classes = subpolicy.inference({
                        'lang_triplets': [_triplet]
                    })

                for a, c in zip(out_low_actions[0], out_low_classes[0]):
                    _res.append([self.LOW_ACTIONS[a], self.LOW_CLASSES[c]])

            res.append(_res)

        return res



