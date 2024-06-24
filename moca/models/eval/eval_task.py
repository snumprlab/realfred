import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval import Eval
from env.thor_env import ThorEnv

import torch
import constants
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
####
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
###
import matplotlib.pyplot as plt
import random

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]
classes_objects = constants.OBJECTS_DETECTOR   + ['0']
classes_receptacles = constants.STATIC_RECEPTACLES  + ['0']


class EvalTask(Eval):
    '''
    evaluate overall task performance
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, successes, failures, results)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()


    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures, results):
        # reset model
        model.reset()
        
        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)

        # extract language features
        feat = model.featurize([(traj_data, False)], load_mask=False)

        # # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        print()
        print('[', goal_instr, ']')
        for n, instr in enumerate(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
            print('  -', n+1, instr)
        print()

        # maskrcnn_objects------------------------------------------------------------------
        maskrcnn_obj = maskrcnn_resnet50_fpn(pretrained=False)
        
        anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
        maskrcnn_obj.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        maskrcnn_obj.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # get number of input features for the classifier
        in_features = maskrcnn_obj.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        maskrcnn_obj.roi_heads.box_predictor = FastRCNNPredictor(in_features, 107+1)

        # now get the number of input features for the mask classifier
        in_features_mask = maskrcnn_obj.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        maskrcnn_obj.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        107+1)
        
        maskrcnn_obj.eval()
        maskrcnn_obj.load_state_dict(torch.load('mrcnn_realfred_objects_007.pth'))
        maskrcnn_obj = maskrcnn_obj.cuda()
        # ------------------------------------------------------------------ maskrcnn_objects
        
        # maskrcnn_receptacles------------------------------------------------------------------
        maskrcnn_rec = maskrcnn_resnet50_fpn(pretrained=False)
        
        anchor_generator = AnchorGenerator(
        sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
        aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
        maskrcnn_rec.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        maskrcnn_rec.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # get number of input features for the classifier
        in_features = maskrcnn_rec.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        maskrcnn_rec.roi_heads.box_predictor = FastRCNNPredictor(in_features, 34+1)

        # now get the number of input features for the mask classifier
        in_features_mask = maskrcnn_rec.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        maskrcnn_rec.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        34+1)
        
        maskrcnn_rec.eval()
        maskrcnn_rec.load_state_dict(torch.load('mrcnn_realfred_receptacles_007.pth'))
        maskrcnn_rec = maskrcnn_rec.cuda()
        # ------------------------------------------------------------------ maskrcnn_objects
        
        # maskrcnn_all ------------------------------------------------------------------
        maskrcnn_all = maskrcnn_resnet50_fpn(pretrained=True)

        anchor_generator = AnchorGenerator(
            sizes=tuple([(4, 8, 16, 32, 64, 128, 256, 512) for _ in range(5)]),
            aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))
        maskrcnn_all.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        maskrcnn_all.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])

        # get number of input features for the classifier
        in_features = maskrcnn_all.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        maskrcnn_all.roi_heads.box_predictor = FastRCNNPredictor(in_features, 142)

        # now get the number of input features for the mask classifier
        in_features_mask = maskrcnn_all.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        maskrcnn_all.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        142)
        maskrcnn_all.eval()
        maskrcnn_all.load_state_dict(torch.load('mrcnn_realfred_all_013.pth'))
        maskrcnn_all = maskrcnn_all.cuda()
        # ------------------------------------------------------------------maskrcnn_all
        prev_image = None
        prev_action = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']

        prev_class = 0
        prev_center = torch.zeros(2)

        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

            # forward model
            m_out = model.step(feat)
            m_pred = model.extract_preds(m_out, [(traj_data, False)], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            if args.save_pictures:
                img_save_path = os.path.join('ForIPIU_Oral_Presentation', 'pictures', traj_data['task_id'], str(r_idx))
                os.makedirs(img_save_path, exist_ok=True)
                curr_image.save(img_save_path+'/%09d.png' % t)

            # action prediction
            action = m_pred['action_low']
            if prev_image == curr_image and prev_action == action and prev_action in nav_actions and action in nav_actions and action == 'MoveAhead_25':
                dist_action = m_out['out_action_low'][0][0].detach().cpu()
                idx_rotateR = model.vocab['action_low'].word2index('RotateRight_90')
                idx_rotateL = model.vocab['action_low'].word2index('RotateLeft_90')
                action = 'RotateLeft_90' if dist_action[idx_rotateL] > dist_action[idx_rotateR] else 'RotateRight_90'

            if action == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break

            # mask prediction
            mask = None
            if model.has_interaction(action):
                class_dist = m_pred['action_low_mask'][0]
                pred_class = np.argmax(class_dist)
                target_label = classes[pred_class]

                # mask generation
                with torch.no_grad():
                    
                    if classes[pred_class] in classes_receptacles :
                        # print('######in recep list######')
                        detector_net = maskrcnn_rec
                        pred_class = classes_receptacles.index(classes[pred_class])
                        category = classes_receptacles
                        flag = '_rec'
                    
                    elif classes[pred_class] in  classes_objects :
                        # print('######in objec list######') 
                        detector_net = maskrcnn_obj
                        pred_class = classes_objects.index(classes[pred_class])
                        category = classes_objects
                        flag = '_obj'
                        
                    else :
                        # print('######in something else list######') 
                        detector_net = maskrcnn_all
                        category = classes
                        flag = '_tiny'
                        
                        
                    out = detector_net([to_tensor(curr_image).cuda()])[0]
                    for k in out:
                        out[k] = out[k].detach().cpu()

                if sum(out['labels'] == pred_class) == 0:
                    mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
                else:
                    masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                    scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                    # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
                    if prev_class != pred_class:
                        scores, indices = scores.sort(descending=True)
                        masks = masks[indices]
                        prev_class = pred_class
                        prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                    else:
                        cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                        distances = ((cur_centers - prev_center)**2).sum(dim=1)
                        distances, indices = distances.sort()
                        masks = masks[indices]
                        prev_center = cur_centers[0]

                    mask = np.squeeze(masks[0].numpy(), axis=0)

            # print action
            if args.debug:
                print(action)
                
            if model.has_interaction(action):
                print(t, action, target_label)
            else:
                print(t, action)
            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)

            if not t_success:
                fails += 1
                # print(f'failed {fails}; {err} ')
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

            prev_image = curr_image
            prev_action = action

        # check if goal was satisfied
        goal_satisfied = env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # goal_conditions
        pcs = env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / (float(t) + 1e-4))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / (float(t) + 1e-4))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        lock.acquire()
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'goal_instr': goal_instr,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            successes.append(log_entry)
        else:
            failures.append(log_entry)

        # overall results
        results['all'] = cls.get_metrics(successes, failures)

        print("-------------")
        print("SR: %d/%d = %.5f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("PLW SR: %.5f" % (results['all']['path_length_weighted_success_rate']))
        print("GC: %d/%d = %.5f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW GC: %.5f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                results[task_type] = cls.get_metrics(task_successes, task_failures)
            else:
                results[task_type] = {}

        lock.release()

    @classmethod
    def get_metrics(cls, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res

    def create_stats(self):
            '''
            storage for success, failure, and results info
            '''
            self.successes, self.failures = self.manager.list(), self.manager.list()
            self.results = self.manager.dict()

    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

