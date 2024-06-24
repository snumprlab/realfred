import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from eval_task import EvalTask
from env.thor_env import ThorEnv
import torch.multiprocessing as mp

import visualize_maskrcnn
import torch
import constants
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


import cv2
import time
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvF
from torchvision.utils import make_grid
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tvF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


tiny_list = ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet', 'AlarmClock', 'ArmChair', 'Sofa']
classes_old = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
# classes_old = [0] + constants.ALL_DETECTOR

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

    # Back to original
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


class Leaderboard(EvalTask):
    '''
    dump action-sequences for leaderboard eval
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, splits, seen_actseqs, unseen_actseqs):
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
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, splits, seen_actseqs, unseen_actseqs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, splits, seen_actseqs, unseen_actseqs):
        # reset model
        model.reset()

        # setup scene
        cls.setup_scene(env, traj_data, r_idx, args)

        # extract language features
        feat = model.featurize([(traj_data, False)], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        
        
        
        
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
        maskrcnn_obj.roi_heads.box_predictor = FastRCNNPredictor(in_features, 73+1)

        # now get the number of input features for the mask classifier
        in_features_mask = maskrcnn_obj.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        maskrcnn_obj.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        73+1)
        
        maskrcnn_obj.eval()
        maskrcnn_obj.load_state_dict(torch.load('objects_lr5e-3_005.pth'))
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
        maskrcnn_rec.roi_heads.box_predictor = FastRCNNPredictor(in_features, 32+1)

        # now get the number of input features for the mask classifier
        in_features_mask = maskrcnn_rec.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        maskrcnn_rec.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        32+1)
        
        maskrcnn_rec.eval()
        maskrcnn_rec.load_state_dict(torch.load('receps_lr5e-3_003.pth'))
        maskrcnn_rec = maskrcnn_rec.cuda()
        # ------------------------------------------------------------------ maskrcnn_objects
        
        # maskrcnn_given ------------------------------------------------------------------
        maskrcnn_given = maskrcnn_resnet50_fpn(num_classes=119)
        maskrcnn_given.eval()
        maskrcnn_given.load_state_dict(torch.load('weight_maskrcnn.pt'))
        maskrcnn_given = maskrcnn_given.cuda()
        # ------------------------------------------------------------------maskrcnn_given
        
        prev_vis_feat = None
        prev_action = None
        prev_image = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        man_actions = ['PickupObject', 'SliceObject', 'OpenObject', 'PutObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']
        
        prev_class = 0
        prev_OE_flag = 0
        OE_flag = 0
        action_mask = None
        prev_category = classes_old
        prev_center = torch.zeros(2)

        vis_feats = []
        pred_actions = []

        done, success = False, False
        actions = list()
        fails = 0
        t = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break
            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            vis_feat = resnet.featurize([curr_image], batch=1).unsqueeze(0)
            feat['frames'] = vis_feat
            vis_feats.append(vis_feat)
            
            img_save_path  ='/home/user/code_lab/twoongg/abp_InstrVPM_moreDF_PMinVPM_noObjnoMan/exp/visualizer_0324_2/'
            if args.visualize :
                os.makedirs(img_save_path+goal_instr, exist_ok=True)
                curr_image.save(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego'+ '.png', 'png')
            
            # with torch.no_grad():                    
            #     out_given = maskrcnn_given([to_tensor(curr_image).cuda()])[0]
            #     out_rec = maskrcnn_rec([to_tensor(curr_image).cuda()])[0]
            #     out_obj = maskrcnn_obj([to_tensor(curr_image).cuda()])[0]

            #     for k in out_given:
            #         out_given[k] = out_given[k].detach().cpu()
            #         out_rec[k] = out_given[k].detach().cpu()
            #         out_obj[k] = out_given[k].detach().cpu()


            # result_given, output, top_predictions = visualize_maskrcnn.predict(curr_image, out_given, classes_old)
            # result_rec, output, top_predictions = visualize_maskrcnn.predict(curr_image, out_rec, classes_receptacles)
            # result_obj, output, top_predictions = visualize_maskrcnn.predict(curr_image, out_obj, classes_objects)
            # cv2.imwrite(img_save_path + goal_instr+'/{0:06d}_given'.format(t) + '.png', result_given)
            # cv2.imwrite(img_save_path + goal_instr+'/{0:06d}_rec'.format(t) + '.png', result_rec)
            # cv2.imwrite(img_save_path + goal_instr+'/{0:06d}_obj'.format(t) + '.png', result_obj)
            
            # cv2.imshow('re', result_given)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            if model.panoramic:
                #curr_image_left, curr_image_right, curr_image_up, curr_image_down = get_panoramic_views(env)
                panoramic_actions, imgs = get_panoramic_actions(env)
                curr_image_left, curr_image_right, curr_image_up, curr_image_down = imgs
                feat['frames_left'] = resnet.featurize([curr_image_left], batch=1).unsqueeze(0)
                feat['frames_right'] = resnet.featurize([curr_image_right], batch=1).unsqueeze(0)
                feat['frames_up'] = resnet.featurize([curr_image_up], batch=1).unsqueeze(0)
                feat['frames_down'] = resnet.featurize([curr_image_down], batch=1).unsqueeze(0)
                #for pa in panoramic_actions:
                #    actions.append({"action": pa[:-3], "forceAction": True})
                #t += len(panoramic_actions)
                #if t >= args.max_steps:
                #    break

                if model.orientation:
                    feat['frames'] = torch.cat([feat['frames'], get_orientation('front').cuda()], dim=2)
                    feat['frames_left'] = torch.cat([feat['frames_left'], get_orientation('left').cuda()], dim=2)
                    feat['frames_up'] = torch.cat([feat['frames_up'], get_orientation('up').cuda()], dim=2)
                    feat['frames_down'] = torch.cat([feat['frames_down'], get_orientation('down').cuda()], dim=2)
                    feat['frames_right'] = torch.cat([feat['frames_right'], get_orientation('right').cuda()], dim=2)

            # forward model
            # print('before step')
            # print(feat.keys())
            m_out = model.step(feat)
            # print('after step')
            # print(feat.keys())
            # print(feat['out_action_low'])
            m_pred = model.extract_preds(m_out, [(traj_data, False)], feat, clean_special_tokens=False)
            
            m_pred = list(m_pred.values())[0]

            # action prediction
            action = m_pred['action_low']
                
            # Loop detection
            isLoop, rand_action = loop_detection(vis_feats, pred_actions, 10)
            if isLoop:
                action = rand_action

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
                        
                    if args.visualize :
                        img=cv2.imread(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego'+ '.png', cv2.IMREAD_UNCHANGED)
                        cv2.putText(img, tmp+"OE"+action, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                        cv2.imwrite(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego_OE'+ '.png', img)
                        os.remove(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego'+ '.png')
                    
            # get action and mask
            #action = m_pred['action_low']
            # if prev_image == curr_image and prev_action == action and prev_action in nav_actions and action in nav_actions and action == 'MoveAhead_25':
            #         tmp=action
            #         dist_action = m_out['out_action_low'][0][0].detach().cpu()
            #         idx_rotateR = model.vocab['action_low'].word2index('RotateRight_90')
            #         idx_rotateL = model.vocab['action_low'].word2index('RotateLeft_90')
            #         action = 'RotateLeft_90' if dist_action[idx_rotateL] > dist_action[idx_rotateR] else 'RotateRight_90'
            #         if args.visualize :
            #             img=cv2.imread(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego'+ '.png', cv2.IMREAD_UNCHANGED)
            #             cv2.putText(img, tmp+"OE"+action, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            #             cv2.imwrite(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego_OE'+ '.png', img)
            #             os.remove(img_save_path + goal_instr+'/{0:06d}_'.format(t) +'ego'+ '.png')

            # check if <<stop>> was predicted
            if m_pred['action_low'] == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break
                
            pred_actions.append(action)

            mask = None
            if model.has_interaction(action):
                class_dist = m_pred['action_low_mask'][0]
                pred_class = np.argmax(class_dist)
                target_label = classes_old[pred_class]


                with torch.no_grad():
                    
                    if classes_old[pred_class] in tiny_list :
                        # print('######in tiny list######')
                        detector_net = maskrcnn_given
                        category = classes_old
                        flag = '_tiny'
                    
                    elif classes_old[pred_class] in classes_receptacles :
                        # print('######in recep list######')
                        detector_net = maskrcnn_rec
                        pred_class = classes_receptacles.index(classes_old[pred_class])
                        category = classes_receptacles
                        flag = '_rec'
                    
                    elif classes_old[pred_class] in  classes_objects :
                        # print('######in objec list######') 
                        detector_net = maskrcnn_obj
                        pred_class = classes_objects.index(classes_old[pred_class])
                        category = classes_objects
                        flag = '_obj'
                        
                    else :
                        # print('######in something else list######') 
                        detector_net = maskrcnn_given
                        category = classes_old
                        flag = '_tiny'
                        
                        
                    out = detector_net([to_tensor(curr_image).cuda()])[0]
                    for k in out:
                        out[k] = out[k].detach().cpu()
                        
                # segmented_result, output, top_predictions = visualize_maskrcnn.predict(curr_image, out, category, target=target_label, action = action)
                
                # if args.visualize :
                #     cv2.imwrite(img_save_path + goal_instr+'/{0:06d}'.format(t)+flag + '.png', segmented_result)
                #     pass
                
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

            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, api_action = env.va_interact(action, interact_mask=mask, smooth_nav=False)
            
            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # save action
            if api_action is not None:
                actions.append(api_action)

            # next time-step
            t += 1

            prev_image = curr_image
            prev_vis_feat = vis_feat
            prev_action = action
            prev_OE_flag = OE_flag
            prev_action_mask = action_mask

        # actseq
        seen_ids = [t['task'] for t in splits['tests_seen']]
        actseq = {traj_data['task_id']: actions}

        # log action sequences
        lock.acquire()

        if traj_data['task_id'] in seen_ids:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        lock.release()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']

        # add seen trajectories to queue
        for traj in seen_files:
            task_queue.put(traj)

        # add unseen trajectories to queue
        for traj in unseen_files:
            task_queue.put(traj)

        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        self.model.test_mode = True
        for n in range(self.args.num_threads):
            thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
                                                       self.splits, self.seen_actseqs, self.unseen_actseqs))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        storage for seen and unseen actseqs
        '''
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(self.seen_actseqs),
                   'tests_unseen': list(self.unseen_actseqs)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, args.dout + '_tests_actseqs_dump_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)
            


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_feat_2.1.0")
    parser.add_argument('--dout', type=str, default='10')
    parser.add_argument('--model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--x_display', type=str, default='0')
    parser.add_argument('--visualize', type=bool, default=False)


    # parse arguments
    args = parser.parse_args()

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # leaderboard dump
    eval = Leaderboard(args, manager)

    # start threads
    eval.spawn_threads()
