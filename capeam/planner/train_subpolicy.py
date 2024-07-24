import os
import torch
from torch import nn
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from models import Manipulator

from dataset import Dataset_Manipulator as Dataset


def compute_loss(out, feat, factorize=True):
    loss = 0
    num_low_classes = 145+5 if factorize else  145
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
        loss = compute_loss(out, feat, factorize=(not args.rawClass))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(args, net, dataset):
    net.eval()

    correct, correct_actions, correct_classes = 0., 0., 0.
    total, total_actions, total_classes = 0., 0., 0.
    with torch.no_grad():
        for feat in tqdm(dataset.iterate(), desc='eval'):

            out = net(feat)
            for i in range(feat['low_actions'].size(0)):
                vis = []
                for t in range(feat['low_actions'].size(1)):
                    action = out['out_low_actions'][i][t]
                    target = out['out_low_classes'][i][t]

                    action = action.argmax().data.item()
                    target = target.argmax().data.item()

                    action = dataset.LOW_ACTIONS[action]
                    target = dataset.LOW_CLASSES[target]

                    vis.append((
                        action if feat['low_actions_mask'][i][t] == 1 else '-',
                        target if feat['low_classes_mask'][i][t] == 1 else '-',
                    ))


                    if action == '<<stop>>':
                        break

                # measure accuracy (actions)
                indices = feat['low_actions_mask'][i].nonzero().squeeze(-1)
                if len(indices) > 0:
                    p_alow = out['out_low_actions'][i][indices]
                    l_alow = feat['low_actions'][i][indices]

                    total_actions += len(l_alow)
                    correct_actions += (p_alow.argmax(1) == l_alow).sum()

                # measure accuracy (classes)
                indices = feat['low_classes_mask'][i].nonzero().squeeze(-1)
                if len(indices) > 0:
                    p_tlow = out['out_low_classes'][i][indices]
                    l_tlow = feat['low_classes'][i][indices]

                    total_classes += len(l_tlow)
                    correct_classes += (p_tlow.argmax(1) == l_tlow).sum()

                correct += 1 if (p_alow.argmax(1) == l_alow).all() and (p_tlow.argmax(1) == l_tlow).all() else 0
                total += 1

    acc = correct / total
    acc_actions = correct_actions / total_actions
    acc_classes = correct_classes / total_classes

    print(
'''
       Acc: {:.3f}% = {}/{}
Action Acc: {:.3f}% = {}/{}
Class  Acc: {:.3f}% = {}/{}
'''.format(
        acc*100, correct, total,
        acc_actions*100, correct_actions, total_actions,
        acc_classes*100, correct_classes, total_classes,
    )
    )

    return acc_actions, acc_classes


def main():
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--epochs', help='random seed', default=50, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--batch_size', help='batch size', default=16, type=int)
    parser.add_argument('--subgoal', help='subgoal type', type=str)
    parser.add_argument('--save_path', help='batch size', default='weight', type=str)
    # ARGS
    parser.add_argument('--rawClass', action='store_true')
    parser.add_argument('--large', action='store_true')

    args = parser.parse_args()

    args.save_path = "weight/%s/%s%s%s"%(
        args.subgoal,
        time.strftime('(%m-%d-%H-%M)', time.localtime(time.time())),
        '_large' if args.large else '',
        '_rawClass' if args.rawClass else '')

    print(f"Saving in {args.save_path}")

    # manipulator
    if args.large:
        dhid = 1024*2
        demb = 100*2
    else:
        dhid = 1024
        demb = 100
    subpolicy = Manipulator(
        dhid=dhid, demb=demb,
        factorize=(not args.rawClass)).cuda()

    # dataset
    dataset_train = Dataset(split='train', subgoal=args.subgoal, batch_size=args.batch_size, factorize=(not args.rawClass))
    dataset_valid_seen = Dataset(split='valid_seen', subgoal=args.subgoal, batch_size=args.batch_size, factorize=(not args.rawClass))
    dataset_valid_unseen = Dataset(split='valid_unseen', subgoal=args.subgoal, batch_size=args.batch_size, factorize=(not args.rawClass))

    # optimizer
    optimizer = torch.optim.Adam(subpolicy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # save path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    best_seen_acc_actions, best_seen_acc_classes = -1., -1.
    best_unseen_acc_actions, best_unseen_acc_classes = -1., -1.
    for epoch in range(args.epochs):

        # train
        train(args, subpolicy, dataset_train, optimizer)

        print(f"[{epoch}]")
        # evaluate in valid seen
        acc_actions, acc_classes = evaluate(args, subpolicy, dataset_valid_seen)
        if acc_actions > best_seen_acc_actions:
            print('New Best Seen Action: epoch {}'.format(epoch))
            best_seen_acc_actions = acc_actions
            torch.save(subpolicy.state_dict(), os.path.join(args.save_path, 'best_seen_acc_actions.pth'))
        if acc_classes > best_seen_acc_classes:
            print('New Best Seen Class: epoch {}'.format(epoch))
            best_seen_acc_classes = acc_classes
            torch.save(subpolicy.state_dict(), os.path.join(args.save_path, 'best_seen_acc_classes.pth'))

        # evaluate in valid unseen
        acc_actions, acc_classes = evaluate(args, subpolicy, dataset_valid_unseen)
        if acc_actions > best_unseen_acc_actions:
            print('New Best Unseen Action: epoch {}'.format(epoch))
            best_unseen_acc_actions = acc_actions
            torch.save(subpolicy.state_dict(), os.path.join(args.save_path, 'best_unseen_acc_actions.pth'))
        if acc_classes > best_unseen_acc_classes:
            print('New Best Unseen Class: epoch {}'.format(epoch))
            best_unseen_acc_classes = acc_classes
            torch.save(subpolicy.state_dict(), os.path.join(args.save_path, 'best_unseen_acc_classes.pth'))

        torch.save(subpolicy.state_dict(), os.path.join(args.save_path, f'epoch{epoch}.pth'))
        torch.save(subpolicy.state_dict(), os.path.join(args.save_path, 'latest.pth'))

        scheduler.step()


if __name__ == '__main__':
    main()
