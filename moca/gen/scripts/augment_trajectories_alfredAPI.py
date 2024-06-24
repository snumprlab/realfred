import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import json
import glob
import os
import constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
from utils.video_util import VideoSaver
from utils.py_util import walklevel
from env.thor_env import ThorEnv


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
HIGH_RES_IMAGES_FOLDER = "raw_images_panoramic"
DEPTH_IMAGES_FOLDER = "depth_images_panoramic"
INSTANCE_MASKS_FOLDER = "instance_masks_panoramic"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = False
render_settings['renderObjectImage'] = False
render_settings['renderClassImage'] = False

video_saver = VideoSaver()


def get_image_index(save_path):
    return len(glob.glob(save_path + '/*.png'))


def save_image_with_delays(env, action,
                           save_path, direction=constants.BEFORE):
    im_ind = get_image_index(save_path)
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        save_image(env.last_event, save_path)
        env.noop()
    return im_ind


def save_image(event, save_path):
    # rgb
    rgb_save_path = os.path.join(save_path, HIGH_RES_IMAGES_FOLDER)
    rgb_image = event.frame[:, :, ::-1]

    # # depth
    # depth_save_path = os.path.join(save_path, DEPTH_IMAGES_FOLDER)
    # depth_image = event.depth_frame
    # depth_image = depth_image * (255 / 10000)
    # depth_image = depth_image.astype(np.uint8)

    # # masks
    # mask_save_path = os.path.join(save_path, INSTANCE_MASKS_FOLDER)
    # mask_image = event.instance_segmentation_frame

    # dump images
    im_ind = get_image_index(rgb_save_path)
    cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image)
    # cv2.imwrite(depth_save_path + '/%09d.png' % im_ind, depth_image)
    # cv2.imwrite(mask_save_path + '/%09d.png' % im_ind, mask_image)

    return im_ind


def save_images_in_events(events, root_dir):
    for event in events:
        save_image(event, root_dir)


def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # make directories
    root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

    orig_images_dir = os.path.join(root_dir, ORIGINAL_IMAGES_FORLDER)
    high_res_images_dir = os.path.join(root_dir, HIGH_RES_IMAGES_FOLDER)
    depth_images_dir = os.path.join(root_dir, DEPTH_IMAGES_FOLDER)
    instance_masks_dir = os.path.join(root_dir, INSTANCE_MASKS_FOLDER)
    augmented_json_file = os.path.join(root_dir, AUGMENTED_TRAJ_DATA_JSON_FILENAME)

    # fresh images list
    traj_data['images'] = list()

    clear_and_create_dir(high_res_images_dir)
    # clear_and_create_dir(depth_images_dir)
    # clear_and_create_dir(instance_masks_dir)

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    event = env.step(dict(traj_data['scene']['init_action']))
    print("Task: %s/%s" % (traj_data['task_type'], traj_data['task_id']))

    save_image(event, root_dir)

    # setup task
    env.set_task(traj_data, args, reward_type='dense')
    rewards = []

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        hl_action_idx, traj_api_cmd, traj_discrete_action = \
            ll_action['high_idx'], ll_action['api_action'], ll_action['discrete_action']

        # print templated low-level instructions & discrete action
        # print("HL Templ: %s, LL Cmd: %s" % (traj_data['template']['high_descs'][hl_action_idx],
        #                                     traj_discrete_action['action']))

        # Use the va_interact that modelers will have to use at inference time.
        action_name, action_args = traj_discrete_action['action'], traj_discrete_action['args']

        # three ways to specify object of interest mask
        # 1. create a rectangular mask from bbox
        # mask = env.bbox_to_mask(action_args['bbox']) if 'bbox' in action_args else None  # some commands don't require any arguments
        # 2. create a point mask from bbox
        # mask = env.point_to_mask(action_args['point']) if 'point' in action_args else None
        # 3. use full pixel-wise segmentation mask
        compressed_mask = action_args['mask'] if 'mask' in action_args else None
        if compressed_mask is not None:
            mask = env.decompress_mask(compressed_mask)
        else:
            mask = None

        success, event, target_instance_id, err, _ = env.va_interact(action_name, interact_mask=mask, smooth_nav=False)
        save_image(event, root_dir)

        # update image list
        new_img_idx = get_image_index(high_res_images_dir)
        last_img_idx = len(traj_data['images'])
        num_new_images = new_img_idx - last_img_idx
        for j in range(num_new_images):
            traj_data['images'].append({
                'low_idx': ll_idx,
                'high_idx': ll_action['high_idx'],
                'image_name': '%09d.png' % int(last_img_idx + j)
            })

        if not event.metadata['lastActionSuccess']:
            raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))

        # reward, _ = env.get_transition_reward()
        # rewards.append(reward)

    # save 10 frames in the end as per the training data
    for _ in range(10):
        save_image(env.last_event, root_dir)

    # store color to object type dictionary
    color_to_obj_id_type = {}
    all_objects = env.last_event.metadata['objects']
    for color, object_id in env.last_event.color_to_object_id.items():
        for obj in all_objects:
            if object_id == obj['objectId']:
                color_to_obj_id_type[str(color)] = {
                    'objectID': obj['objectId'],
                    'objectType': obj['objectType']
                }

    augmented_traj_data = copy.deepcopy(traj_data)
    augmented_traj_data['scene']['color_to_object_type'] = color_to_obj_id_type
    augmented_traj_data['task'] = {'rewards': rewards, 'reward_upper_bound': sum(rewards)}

    with open(augmented_json_file, 'w') as aj:
        json.dump(augmented_traj_data, aj, sort_keys=True, indent=4)

    """
    # save video
    images_path = os.path.join(high_res_images_dir, '*.png')
    video_save_path = os.path.join(high_res_images_dir, 'high_res_video.mp4')
    video_saver.save(images_path, video_save_path)

    # check if number of new images is the same as the number of original images
    if args.smooth_nav and args.time_delays:
        orig_img_count = get_image_index(high_res_images_dir)
        new_img_count = get_image_index(orig_images_dir)
        print ("Original Image Count %d, New Image Count %d" % (orig_img_count, new_img_count))
        if orig_img_count != new_img_count:
            raise Exception("WARNING: the augmented sequence length doesn't match the original")
    """


def run(done):
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv()
    skipped_files = []
    done = done
    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print (len(traj_list),"Left, Augmenting: " + json_file)
        try:
            augment_traj(env, json_file)
            with lock:
                done.append(json_file)
                with open('done_panoramic_imageGen.json', 'w') as f:
                    json.dump(done, f, indent = 4)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print ("Error: " + repr(e))
            print ("Skipping " + json_file)
            skipped_files.append(json_file)

    env.stop()
    print("Finished.")

    # skipped files
    if len(skipped_files) > 0:
        print("Skipped Files:")
        print(skipped_files)
        

if __name__ == '__main__':
    traj_list = []
    lock = threading.Lock()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/media/user/Data2/abp_Withgoal/data/Re_json_2.1.1")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true')
    parser.add_argument('--time_delays', dest='time_delays', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--reward_config', type=str, default='models/config/rewards.json')
    args = parser.parse_args()
    fail_trajs = []

    # # make a list of all the traj_data json files
    # for dir_name, subdir_list, file_list in walklevel(args.data_path, level=3):
    #     if "trial_" in dir_name and 'tests_' not in dir_name and 'train' in dir_name:
    #         json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
    #         if not os.path.isfile(json_file):
    #             continue
    #         traj_list.append(json_file)
    #         aug_json_file = os.path.join(dir_name, AUGMENTED_TRAJ_DATA_JSON_FILENAME)
    #         if not os.path.isfile(aug_json_file):
    #             fail_trajs.append(json_file)
    traj_list = json.load(open('replay_need.json'))
    print(len(traj_list))
    # traj_list = [t for t in traj_list if any(f in t for f in fail_trajs)]
    print(len(traj_list))
    lock = threading.Lock()
    done=[]
    # start threads
    threads = []
    for n in range(args.num_threads):
        thread = threading.Thread(target=run, args=(done,))
        threads.append(thread)
        thread.start()
        time.sleep(1)