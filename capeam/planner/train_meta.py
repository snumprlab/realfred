import os
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import time
from models import MetaController_LSTM as MetaController

from dataset import Dataset_MetaController as Dataset


def compute_loss(out, feat, factorize=True, nohier=False):
    loss = 0

    num_classes = 145+5 if factorize else 145
    num_actions = 9+4 if nohier else 9
    for k, SPACE in [('actions', num_actions), ('targets', num_classes), ('receptacles', num_classes)]:
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
        loss = compute_loss(out, feat, factorize=(not args.rawClass), nohier=(args.nohier))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(args, net, dataset):
    net.eval()

    correct, correct_actions, correct_targets, correct_receptacles = 0., 0., 0., 0.
    total, total_actions, total_targets, total_receptacles = 0., 0., 0., 0.
    with torch.no_grad():
        for feat in tqdm(dataset.iterate(), desc='eval'):
            out = net(feat)

            for i in range(feat['actions'].size(0)):
                # measure accuracy (actions)
                indices = feat['actions_mask'][i].nonzero().squeeze(-1)
                if len(indices) > 0:
                    p_alow = out['out_actions'][i][indices]
                    l_alow = feat['actions'][i][indices]

                    #total_actions += len(l_alow)
                    #correct_actions += (p_alow.argmax(1) == l_alow).sum()
                    action_correct = (p_alow.argmax(1) == l_alow).all()
                    total_actions += 1
                    correct_actions += 1 if action_correct else 0

                # measure accuracy (targets)
                indices = feat['targets_mask'][i].nonzero().squeeze(-1)
                if len(indices) > 0:
                    p_tlow = out['out_targets'][i][indices]
                    l_tlow = feat['targets'][i][indices]

                    #total_targets += len(l_tlow)
                    #correct_targets += (p_tlow.argmax(1) == l_tlow).sum()
                    target_correct = (p_tlow.argmax(1) == l_tlow).all()
                    total_targets += 1
                    correct_targets += 1 if target_correct else 0

                # measure accuracy (receptacles)
                indices = feat['receptacles_mask'][i].nonzero().squeeze(-1)
                if len(indices) > 0:
                    p_rlow = out['out_receptacles'][i][indices]
                    l_rlow = feat['receptacles'][i][indices]

                    #total_receptacles += len(l_rlow)
                    #correct_receptacles += (p_rlow.argmax(1) == l_rlow).sum()
                    receptacle_correct = (p_rlow.argmax(1) == l_rlow).all()
                    total_receptacles += 1
                    correct_receptacles += 1 if receptacle_correct else 0

                total += 1
                if args.nohier:
                    correct += 1 if all([action_correct, target_correct]) else 0    
                else:
                    correct += 1 if all([action_correct, target_correct, receptacle_correct]) else 0


                if not target_correct and i==0:
                    print('Root:', feat['root'][i])
                    print('Goal:', feat['goal_natural'][i])

                    vis = []
                    for t in range(feat['actions'].size(1)):
                        action = out['out_actions'][i][t]
                        target = out['out_targets'][i][t]
                        receptacle = out['out_receptacles'][i][t]

                        action = action.argmax().data.item()
                        target = target.argmax().data.item()
                        receptacle = receptacle.argmax().data.item()

                        action = dataset.ACTIONS[action]
                        target = dataset.TARGETS[target]
                        receptacle = dataset.TARGETS[receptacle]

                        l_action = dataset.ACTIONS[feat['actions'][i][t]]
                        l_target = dataset.TARGETS[feat['targets'][i][t]]
                        l_receptacle = dataset.TARGETS[feat['receptacles'][i][t]]

                        vis.append((
                            action if feat['actions_mask'][i][t] == 1 else '-',
                            target if feat['targets_mask'][i][t] == 1 else '-',
                            receptacle if feat['receptacles_mask'][i][t] == 1 else '-'
                        ))

                        print(' - {:02d}'.format(t+1), vis[-1], (l_action, l_target, l_receptacle))


                        if action == '<<stop>>':
                            break
                    print()

    acc = correct / total
    acc_actions = correct_actions / total_actions
    acc_targets = correct_targets / total_targets
    if not args.nohier:
        acc_receptacles = correct_receptacles / total_receptacles
    else:
        acc_receptacles = 0

    print(
        '''
            Acc: {:.3f}% = {}/{}
        Action Acc: {:.3f}% = {}/{}
        Target Acc: {:.3f}% = {}/{}
        Recep  Acc: {:.3f}% = {}/{}
        '''.format(
                acc*100, correct, total,
                acc_actions*100, correct_actions, total_actions,
                acc_targets*100, correct_targets, total_targets,
                acc_receptacles*100, correct_receptacles, total_receptacles,
            )
    )

    return acc_actions, acc_targets, acc_receptacles


def main():
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--epochs', help='random seed', default=100, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', help='batch size', default=16, type=int)
    parser.add_argument('--eval_only', help='evaluation only', default=False, action='store_true')
    parser.add_argument('--eval_weight', help='eval weight path', default='weight/MetaController/latest.pth', type=str)

    ## ARGS
    parser.add_argument('--noappended', action='store_true')
    parser.add_argument('--rawClass', action='store_true')
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--nohier', action='store_true')
    
    args = parser.parse_args()
    save_path = "weight/MetaController/%s%s%s%s%s"%(
        time.strftime('%m-%d-%H-%M', time.localtime(time.time())),
        '_large' if args.large else '',
        '_rawClass' if args.rawClass else '',
        '_noappended' if args.noappended else '',
        '_nohier' if args.nohier else '')
    print('-'*20)
    print(save_path)

    # meta controller
    if args.large:
        dhid = 1024*2
        demb = 100
    else:
        dhid = 1024
        demb = 100
    meta_controller = MetaController(
        dhid=dhid, demb=demb,
        appended=(not args.noappended),
        factorize=(not args.rawClass),
        nohier=(args.nohier)).cuda()

    # dataset
    dataset_train = Dataset(
        split='train', batch_size=args.batch_size, 
        appended=(not args.noappended), factorize=(not args.rawClass), nohier=(args.nohier))
    dataset_valid_seen = Dataset(
        split='valid_seen', batch_size=args.batch_size, 
        appended=(not args.noappended), factorize=(not args.rawClass), nohier=(args.nohier))
    dataset_valid_unseen = Dataset(
        split='valid_unseen', batch_size=args.batch_size, 
        appended=(not args.noappended), factorize=(not args.rawClass), nohier=(args.nohier))

    if args.eval_only:
        meta_controller.load_state_dict(torch.load(args.eval_weight))

        print('Valid Seen')
        evaluate(args, meta_controller, dataset_valid_seen)

        print('Valid Unseen')
        evaluate(args, meta_controller, dataset_valid_unseen)

        exit(0)

    # optimizer
    optimizer = torch.optim.Adam(meta_controller.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # save path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    best_seen_acc_actions, best_seen_acc_targets, best_seen_acc_receptacles = -1., -1., -1.
    best_unseen_acc_actions, best_unseen_acc_targets, best_unseen_acc_receptacles = -1., -1., -1.
    for epoch in range(args.epochs):

        # train
        train(args, meta_controller, dataset_train, optimizer)
        print('\nEpoch: {}'.format(epoch))
        acc_dict = {}

        # evaluate in valid seen
        acc_actions, acc_targets, acc_receptacles = evaluate(args, meta_controller, dataset_valid_seen)
        if acc_actions > best_seen_acc_actions:
            best_seen_acc_actions = acc_actions
            torch.save(meta_controller.state_dict(), os.path.join(save_path, 'best_seen_acc_actions.pth'))
        if acc_targets > best_seen_acc_targets:
            best_seen_acc_targets = acc_targets
            torch.save(meta_controller.state_dict(), os.path.join(save_path, 'best_seen_acc_targets.pth'))
        if acc_receptacles > best_seen_acc_receptacles:
            best_seen_acc_receptacles = acc_receptacles
            torch.save(meta_controller.state_dict(), os.path.join(save_path, 'best_seen_acc_receptacles.pth'))

        # # evaluate in valid unseen
        acc_actions, acc_targets, acc_receptacles = evaluate(args, meta_controller, dataset_valid_unseen)
        if acc_actions > best_unseen_acc_actions:
            best_unseen_acc_actions = acc_actions
            torch.save(meta_controller.state_dict(), os.path.join(save_path, 'best_unseen_acc_actions.pth'))
        if acc_targets > best_unseen_acc_targets:
            best_unseen_acc_targets = acc_targets
            torch.save(meta_controller.state_dict(), os.path.join(save_path, 'best_unseen_acc_targets.pth'))
        if acc_receptacles > best_unseen_acc_receptacles:
            best_unseen_acc_receptacles = acc_receptacles
            torch.save(meta_controller.state_dict(), os.path.join(save_path, 'best_unseen_acc_receptacles.pth'))

        # torch.save(meta_controller.state_dict(), os.path.join(save_path, 'latest.pth'))
        if epoch % 5 == 0:
            torch.save(meta_controller.state_dict(), os.path.join(save_path, f'epoch{epoch}.pth'))

        scheduler.step()
        



if __name__ == '__main__':
    main()
