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
# from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import matplotlib.pyplot as plt
import random

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', "OnionSliced", "StrawberrySliced", "LemonSliced", "BananaSliced", "EggplantSliced"]

classes_objects = constants.OBJECTS_DETECTOR   + ['0']
classes_receptacles = constants.STATIC_RECEPTACLES  + ['0']

import random
def loop_detection(vis_feats, actions, window_size=10):

    # not enough vis feats for loop detection
    if len(vis_feats) < window_size*2:
        return False, None

    nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90'] #, 'LookDown_15', 'LookUp_15']
    random.shuffle(nav_actions)

    start_idx = len(vis_feats) - 1

    for end_idx in range(start_idx - window_size, window_size - 1, -1):
        if (vis_feats[start_idx] == vis_feats[end_idx]).all():
            if all((vis_feats[start_idx-i] == vis_feats[end_idx-i]).all() for i in range(window_size)):
                return True, nav_actions[1] if actions[end_idx] == nav_actions[0] else nav_actions[0]

    return False, None


import math
def get_orientation(d):
    if d == 'left':
        h, v = -math.pi/2, 0.0
    elif d == 'up':
        h, v = 0.0, -math.pi/12
    elif d == 'down':
        h, v = 0.0, math.pi/12
    elif d == 'right':
        h, v = math.pi/2, 0.0
    else:
        h, v = 0.0, 0.0

    orientation = torch.cat([
        torch.cos(torch.ones(1)*(h)),
        torch.sin(torch.ones(1)*(h)),
        torch.cos(torch.ones(1)*(v)),
        torch.sin(torch.ones(1)*(v)),
    ]).unsqueeze(-1).unsqueeze(-1).repeat(1,7,7).unsqueeze(0).unsqueeze(0)

    return orientation

def get_panoramic_views(env):
    horizon = np.round(env.last_event.metadata['agent']['cameraHorizon'])
    rotation = env.last_event.metadata['agent']['rotation']
    position = env.last_event.metadata['agent']['position']

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 270.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_left = Image.fromarray(np.uint8(env.last_event.frame))

    # Right
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": (rotation['y'] + 90.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_right = Image.fromarray(np.uint8(env.last_event.frame))

    # Up
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon - constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_up = Image.fromarray(np.uint8(env.last_event.frame))

    # Down
    env.step({
        "action": "TeleportFull",
        "horizon": np.round(horizon + constants.AGENT_HORIZON_ADJ),
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })
    curr_image_down = Image.fromarray(np.uint8(env.last_event.frame))

    # Left
    env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotateOnTeleport": True,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
    })

    return curr_image_left, curr_image_right, curr_image_up, curr_image_down


def get_panoramic_actions(env):
    action_pairs = [
        ['RotateLeft_90', 'RotateRight_90'],
        ['RotateRight_90', 'RotateLeft_90'],
        ['LookUp_15', 'LookDown_15'],
        ['LookDown_15', 'LookUp_15'],
    ]
    imgs = []
    actions = []

    curr_image = Image.fromarray(np.uint8(env.last_event.frame))

    for a1, a2 in action_pairs:
        t_success, _, _, err, api_action = env.va_interact(a1, interact_mask=None, smooth_nav=False)
        actions.append(a1)
        imgs.append(Image.fromarray(np.uint8(env.last_event.frame)))
        #if len(err) == 0:
        if curr_image != imgs[-1]:
            t_success, _, _, err, api_action = env.va_interact(a2, interact_mask=None, smooth_nav=False)
            actions.append(a2)
        else:
            #print(err)
            print('Error while {}'.format(a1))
    return actions, imgs



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
            
        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # goal_instr = traj_data['template']['task_desc']
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
                

        prev_vis_feat = None
        prev_action = None
        prev_image = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        man_actions = ['PickupObject', 'SliceObject', 'OpenObject', 'PutObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']

        prev_class = 0
        prev_OE_flag = 0
        OE_flag = 0
        action_mask = None
        prev_category = classes
        prev_center = torch.zeros(2)

        vis_feats = []
        pred_actions = []
        loop_count = 0


        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        tt = 0
        
        env.step({'action' : 'GetReachablePositions'})
        reachable_points = set()
        for point in env.last_event.metadata['actionReturn']:
            reachable_points.add((point['x'], point['z']))

        visited_points = list()        
        collided_points = list()

        while not done:
            visited_points.append((env.last_event.metadata['agent']['position']['x'], env.last_event.metadata['agent']['position']['z']))

            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            feat['frames'] = vis_feat
            vis_feats.append(vis_feat)
            if model.panoramic:
                #curr_image_left, curr_image_right, curr_image_up, curr_image_down = get_panoramic_views(env)
                panoramic_actions, imgs = get_panoramic_actions(env)
                curr_image_left, curr_image_right, curr_image_up, curr_image_down = imgs
                feat['frames_left'] = resnet.featurize([curr_image_left], batch=1).unsqueeze(0)
                feat['frames_right'] = resnet.featurize([curr_image_right], batch=1).unsqueeze(0)
                feat['frames_up'] = resnet.featurize([curr_image_up], batch=1).unsqueeze(0)
                feat['frames_down'] = resnet.featurize([curr_image_down], batch=1).unsqueeze(0)
                if args.save_pictures:
                    img_save_path = os.path.join('pictures', 'pictures', traj_data['task_id'], str(r_idx))
                    os.makedirs(img_save_path, exist_ok=True)
                    curr_image.save(img_save_path+'/%09d.png' % t)
                    for x, img in enumerate(imgs):
                        img.save(img_save_path+'/%09d_%d.png' % (t, x))

                if model.orientation:
                    feat['frames'] = torch.cat([feat['frames'], get_orientation('front').cuda()], dim=2)
                    feat['frames_left'] = torch.cat([feat['frames_left'], get_orientation('left').cuda()], dim=2)
                    feat['frames_up'] = torch.cat([feat['frames_up'], get_orientation('up').cuda()], dim=2)
                    feat['frames_down'] = torch.cat([feat['frames_down'], get_orientation('down').cuda()], dim=2)
                    feat['frames_right'] = torch.cat([feat['frames_right'], get_orientation('right').cuda()], dim=2)

            # forward model
            m_out = model.step(feat)
            m_pred = model.extract_preds(m_out, [(traj_data, False)], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # action prediction
            action = m_pred['action_low']
            
            # Loop detection
            isLoop, rand_action = loop_detection(vis_feats, pred_actions, 10)
            if isLoop:
                action = rand_action
                loop_count += 1

            if prev_vis_feat != None:
                od_score = ((prev_vis_feat - vis_feat)**2).sum().sqrt()
                epsilon = 0.9
                tmp=action
                OE_flag = 0
                if od_score < epsilon :
                    print('Obstruction Evasion')
                    OE_flag = 1
                    dist_action = m_out['out_action_low'][0][0].detach().cpu()
                    dist_action = F.softmax(dist_action)
                    if prev_OE_flag == 1 :
                        action_mask = prev_action_mask
                    else:
                        action_mask = torch.ones(len(model.vocab['action_low']), dtype=torch.float)
                    action_mask[model.vocab['action_low'].word2index(prev_action)] = -1
                    action = model.vocab['action_low'].index2word(torch.argmax(dist_action*action_mask))

            if m_pred['action_low'] == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break
            

            pred_actions.append(action)
            
            # mask prediction
            mask = None
            if model.has_interaction(action):
                class_dist = m_pred['action_low_mask'][0]
                pred_class = np.argmax(class_dist)
                target_label = classes[pred_class]


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
                    #mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
                    # resample action and mask should be None
                    mask = None
                    
                    action_dist = m_out['out_action_low'][0][0]
                    action_mask = torch.zeros_like(action_dist)
                    for i in range(len(model.vocab['action_low'])):
                        if model.vocab['action_low'].index2word(i) in nav_actions[1:3]:
                            action_mask[i] = 1
                    action = model.vocab['action_low'].index2word(((action_dist - action_dist.min()) * action_mask).argmax().item())
                    print('      Resampled action:', action)
                else:
                    masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                    scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                    # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
                    if prev_category[prev_class] != category[pred_class]:
                        scores, indices = scores.sort(descending=True)
                        masks = masks[indices]
                        prev_class = pred_class
                        prev_category = category
                        prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                    else:
                        cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                        distances = ((cur_centers - prev_center)**2).sum(dim=1)
                        distances, indices = distances.sort()
                        masks = masks[indices]
                        prev_center = cur_centers[0]

                    mask = np.squeeze(masks[0].numpy(), axis=0)


            if model.has_interaction(action):
                print(t, action, target_label, category[pred_class])
            else:
                
                print(t, action)

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            
            if not t_success:
                fails += 1
                if "MoveAhead" in action:
                    collided_points.append((env.last_event.metadata['agent']['position']['x'], env.last_event.metadata['agent']['position']['z']))
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break
            
            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

            prev_image = curr_image
            prev_vis_feat = vis_feat
            prev_action = action
            prev_OE_flag = OE_flag
            prev_action_mask = action_mask


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
                     'reward': float(reward),
                     'loop_count': loop_count,
                    'reachable_points' : len(reachable_points),
                    'visited_points' : visited_points,
                     'len_visited_points': len(set(visited_points)),
                     'scene_num' : traj_data['scene']['scene_num'],
                     'collided_points': collided_points}

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
        er = float((sum([entry['len_visited_points'] for entry in successes]) + sum([entry['len_visited_points'] for entry in failures])) / (sum([entry['reachable_points'] for entry in successes]) + sum([entry['reachable_points'] for entry in failures])))
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
        res['explore_rate'] = er
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
