import os
import json
from unittest import result
from pandas import json_normalize
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import pprint
import time
from models import HierarchicalAgent

from dataset import Dataset_HierarchicalAgent as Dataset


def compute_loss(out, feat, factorize=False):
    loss = 0
    num_low_classes = 119+5 if factorize else 119
    for k, SPACE in [('low_actions', 15), ('low_classes', num_low_classes)]:
        preds = out['out_'+k].view(-1, SPACE)
        labels = feat[k].view(-1)
        valid = feat[k+'_mask'].view(-1)

        _loss = F.cross_entropy(preds, labels, reduction='none') * valid.float()
        loss = loss + _loss.mean()

    return loss


def train(args, net, dataset, optimizer):
    net.train()

    for feat in tqdm(dataset.iterate(), desc='train'):
        out = net(feat)
        loss = compute_loss(out, feat, factorize=(args.rawSub))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(args, net, dataset, save_path=None):

    net.eval()

    pp = pprint.PrettyPrinter()

    correct, correct_actions, correct_classes = 0., 0., 0.
    total, total_actions, total_classes = 0., 0., 0.
    goals = set()
    result = {}
    repr = {}
    with torch.no_grad():
        for feat in dataset.iterate():
            out = net.inference(feat)
            for i in range(len(feat['root'])):
                output = {}; res = {}
                # visualization
                print('='*50)
                print('Root: ', feat['root'][i])
                print()
                res['root'] = feat['root'][i]

                print('Goal: ', feat['goal_natural'][i])
                goal = feat['goal_natural'][i]
                output['goal'] =  goal
                
                if not args.noappended:
                    print('Instruction')
                    for n, instr in enumerate(feat['instr_natural'][i]):
                        print(' {:02d}: {}'.format(n+1, instr))
                    output['instr_natural'] = res['instr_natural'] = feat['instr_natural'][i]

                print()
                print('-'*50)
                print()

                if not args.rawSub and 'test' not in dataset.split:
                    pddl_params = json.load(open(os.path.join(feat['root'][i], 'pp', 'ann_%d.json'%(feat['repeat_idx'][i])), 'r'))['pddl_params']
                    pddl_params['object_target_sliced'] = pddl_params['object_target']+'Sliced'
                    if pddl_params['parent_target'] in ['Sink', 'Bathtub']:
                        pddl_params['parent_target'] = pddl_params['parent_target']+'Basin'
                    pp.pprint(pddl_params)

                print()
                print('-'*50)
                print()

                out_triplets = []
                for triplet in out['out_triplets'][i]:
                    out_triplets.append((
                        dataset.TRIPLETS[triplet[0]],
                        dataset.TRIPLETS[triplet[1]],
                        dataset.TRIPLETS[triplet[2]],
                    ))

                # Replace to (Put obj recep) in repr file
                temp = []
                for triplet in out_triplets:
                    if triplet[0] == 'PutObject':
                        if temp[-1][0] == 'SliceObject':
                            temp.append((triplet[0], temp[-2][1], triplet[2]))
                        else:
                            temp.append((triplet[0], temp[-1][1], triplet[2]))
                    else:
                        temp.append(triplet)
                
                output['triplet'] = out_triplets
                res['triplet'] = temp

                if dataset.split in ['valid_seen', 'valid_unseen']:
                    gt_triplets = [t for t in feat['triplets'][i] if t[0] != 'GotoLocation' and t[0] != 'NoOp']
                
                    print(' [Meta Controller]')
                    for k in range(max(len(gt_triplets), len(out_triplets))):
                        if not args.rawSub:
                            if k < len(out_triplets):
                                triplet_pred = []
                                for arg in out_triplets[k]:
                                    triplet_pred.append(pddl_params[arg] if arg in pddl_params else arg)
                            else:
                                triplet_pred = ['-']*3
                        else:
                            triplet_pred = out_triplets[k] if k < len(out_triplets) else ['-']*3    
                        triplet_label = gt_triplets[k] if k < len(gt_triplets) else ['-']*3
                        wrong = '' if all(p == l for p, l in zip(triplet_pred, triplet_label)) else 'V'
                        print('   {:02d}: {:15s}, {:13s}, {:13s} ({:15s}, {:13s}, {:13s}) {}'.format(
                            k + 1,
                            *triplet_pred,
                            *triplet_label,
                            wrong,
                        ))
                else:
                    dummy = ['-', '-', '-']

                    print(' [Meta Controller]')
                    for k in range(len(out_triplets)):
                        triplet_pred = out_triplets[k] if k < len(out_triplets) else ['-']*3
                        print('   {:02d}: {:15s}, {:13s}, {:13s} ({:15s}, {:13s}, {:13s}) {}'.format(
                            k + 1,
                            *triplet_pred,
                            *dummy,
                            0,
                        ))


                print()
                print('-'*50)
                print()


                nav_actions = ['MoveAhead_25', 'RotateRight_90', 'RotateLeft_90', 'LookDown_15', 'LookUp_15']
                out_actions = [dataset.LOW_ACTIONS[a] for a in out['out_low_actions'][i]]
                out_classes = [dataset.LOW_CLASSES[c] for c in out['out_low_classes'][i]]

                output['low_actions'] = res['low_actions'] = out_actions
                output['low_classes'] = res['low_classes'] = out_classes
                res['high_idxs'] = out['out_h_idxs'][i]
                
                if dataset.split in ['valid_seen', 'valid_unseen']:
                    gt_actions = [dataset.LOW_ACTIONS[a] for a in feat['low_actions'][i] if dataset.LOW_ACTIONS[a] not in nav_actions]
                    gt_classes = [dataset.LOW_CLASSES[c] for c in feat['low_classes'][i] if c != 0]
                    print(' [Subpolicies]')
                    for t in range(max(len(gt_actions), len(out_actions))):
                        a_pred = out_actions[t] if t < len(out_actions) else '-'
                        if not args.rawSub:
                            if t < len(out_classes):
                                c_pred = pddl_params[out_classes[t]] if out_classes[t] in pddl_params else out_classes[t]
                            else:
                                c_pred = '-'
                        else:
                            c_pred = out_classes[t] if t < len(out_classes) else '-'        
                        a_label = gt_actions[t] if t < len(gt_actions) else '-'
                        c_label = gt_classes[t] if t < len(gt_classes) else '-'
                        wrong = '' if a_pred == a_label and c_pred == c_label else 'V'
                        print('   {:02d}: {:15s}, {:13s} ({:15s}, {:13s}) {}'.format(
                            t + 1,
                            a_pred,
                            c_pred,
                            a_label,
                            c_label,
                            wrong
                        ))

                    _correct = 1
                    if len(gt_actions) != len(out_actions):
                        _correct = 0
                    else:
                        for t in range(max(len(gt_actions), len(out_actions))):
                            if not (out_actions[t] == gt_actions[t] and out_classes[t] == gt_classes[t]):
                                _correct = 0
                                break
                    correct += _correct
                    total += 1

                    print()
                    print('-'*50)
                    print()

                    print('Success / Fail: ' + ('Success' if _correct == 1 else 'Fail'))

                    print()
                    #acc_actions = correct_actions / total_actions
                    #acc_classes = correct_classes / total_classes

                    #print('Acc: {:.3f}% = {}/{}'.format((correct / total)*100, correct, total))
                else:
                    print(' [Subpolicies]')

                    for t in range(len(out_actions)):
                        print('   {:02d}: {:15s}, {:13s} ({:15s}, {:13s}) {}'.format(
                            t + 1,
                            out_actions[t],
                            out_classes[t],
                            '-',
                            '-',
                            0
                        ))

                print('='*50)
                print()
                result[os.path.join(feat['root'][i], 'pp', 'ann_%d.json'%(feat['repeat_idx'][i]))] = output
                repr[goal] = res
        
        # print('Acc: {:.3f}% = {}/{}'.format((correct / total)*100, correct, total))
        json.dump(repr, open((os.path.join(os.path.split(save_path)[0]+'-repr', os.path.split(save_path)[1])), 'w'), indent=4)

# function to save eval output in json file
def evaluate_json(args, net, dataset):
    net.eval()
    
    with torch.no_grad():
        result = dict()
        goals = dict()
        cnt = 0
        rcnt = 0
        for feat in tqdm(dataset.iterate()):
            out = net.inference(feat)
            for i in range(len(feat['root'])):

                goal = feat['goal_natural'][i]
                if goal in goals.keys():
                    print('Goal: ', goal)
                    print('root 1: ', feat['root'][i])
                    print('root 2: ', goals[goal])
                    if feat['root'][i] == goals[goal]:
                        print('same root')
                        rcnt += 1
                    print()
                    cnt += 1
                else:
                    goals[goal] = feat['root'][i]

                out_actions = [dataset.LOW_ACTIONS[a] for a in out['out_low_actions'][i]]
                out_classes = [dataset.LOW_CLASSES[c] for c in out['out_low_classes'][i]]
                result[goal] = []

                for t in range(len(out_actions)):
                    a_pred = out_actions[t] if t < len(out_actions) else '-'
                    c_pred = out_classes[t] if t < len(out_classes) else '-'
                    pred = (a_pred, c_pred)
                    result[goal].append(pred)
    print('duplication: ', cnt)
    print('root concided: ', rcnt)
    return result

            
def main():
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--epochs', help='random seed', default=1, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', help='batch size', default=16, type=int)
    parser.add_argument('--metaWeight', help='path to meta planner weight', required=True, type=str)
    parser.add_argument('--subWeight', help='path to subpolicy planner weight', required=True, type=str)

    ## args
    parser.add_argument('--noappended', help='Use only high level language',action='store_true')
    parser.add_argument('--rawMeta', help='factorize',action='store_true')
    parser.add_argument('--rawSub', help='factorize',action='store_true')
    parser.add_argument('--large', help='Large Model',action='store_true')

    args = parser.parse_args()

    save_folder = "result/%s%s%s%s%s"%(
        time.strftime('%m-%d-%H-%M', time.localtime(time.time())),
        '_large' if args.large else '',
        '_rawClass(Meta)' if args.rawMeta else '',
        '_rawClass(Sub)' if args.rawSub else '',
        '_noappended' if args.noappended else '')

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder+'-repr', exist_ok=True)
    # meta controller
    hierarchical_agent = HierarchicalAgent(meta_weight=args.metaWeight, sub_weight=args.subWeight, factorize_meta=(not args.rawMeta), factorize_subpolicy=(not args.rawSub), large=args.large).cuda()

    # dataset
    dataset_valid_seen = Dataset(split='valid_seen', batch_size=args.batch_size, appended=(not args.noappended))
    dataset_valid_unseen = Dataset(split='valid_unseen', batch_size=args.batch_size, appended=(not args.noappended))
    dataset_tests_seen = Dataset(split='tests_seen', batch_size=args.batch_size, appended=(not args.noappended))
    dataset_tests_unseen = Dataset(split='tests_unseen', batch_size=args.batch_size, appended=(not args.noappended))


    print('valid seen')
    evaluate(args, hierarchical_agent, dataset_valid_seen, save_path=os.path.join(save_folder,'valid_seen.json'))

    print('valid unseen')
    evaluate(args, hierarchical_agent, dataset_valid_unseen, save_path=os.path.join(save_folder,'valid_unseen.json'))

    print('tests seen')
    evaluate(args, hierarchical_agent, dataset_tests_seen, save_path=os.path.join(save_folder,'test_seen.json'))
    
    print('tests unseen')
    evaluate(args, hierarchical_agent, dataset_tests_unseen, save_path=os.path.join(save_folder,'test_unseen.json'))



if __name__ == '__main__':
    main()
