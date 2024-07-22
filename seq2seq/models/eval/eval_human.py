import os
import json
import numpy as np
from PIL import Image
from datetime import datetime
from eval_ import Eval
from env.thor_env_ import ThorEnv
import tkinter as tk
import sys
import termios
import tty
import cv2 



class EvalTaskHuman(Eval):
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
        
        def get_xy_in_image(image):
            class MouseGesture():
                def __init__(self) -> None:
                    self.is_dragging = False 
                    self.x0, self.y0, self.w0, self.h0 = -1,-1,-1,-1
                    self.resulting_coordinate = None

                def on_mouse(self, event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.resulting_coordinate = [x, y]

                    return 

            mouse_class = MouseGesture()
            print("Click object you are interacting, then press \"Q\" ")
            while True:
                cv2.imshow('ReALFRED', image)
                cv2.setMouseCallback('ReALFRED', mouse_class.on_mouse, param=image)
                cv2.waitKey(0)
                if mouse_class.resulting_coordinate != None:
                    cv2.destroyAllWindows()   
                    break
            mask = np.zeros((900, 900))
            # print(mouse_class.resulting_coordinate)
            mask[mouse_class.resulting_coordinate[1]][mouse_class.resulting_coordinate[0]] = 1
            return mask

                
        def get_key():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                key = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                
            if key == 'w' or key == 'W':
                action = 'MoveAhead'
                mask = None
                
            elif key == 'a' or key == 'A':
                action = 'RotateLeft'
                mask = None
                
            elif key == 'd' or key == 'D':
                action = 'RotateRight'
                mask = None
                
            elif key == 'q' or key == 'Q':
                action = 'LookDown'
                mask = None
                
            elif key == 'e' or key == 'e':
                action = 'LookUp'
                mask = None

            elif key == '1' :
                action = 'PickupObject'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))
                
            elif key == '2' :
                action = 'PutObject'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))

            elif key == '3' :
                action = 'OpenObject'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))
            
            elif key == '4' :
                action = 'CloseObject'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))

            elif key == '5' :
                action = 'SliceObject'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))
                
            elif key == '6' :
                action = 'ToggleObjectOn'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))

            elif key == '7' :
                action = 'ToggleObjectOff'
                mask = get_xy_in_image(np.uint8(env.last_event.frame[:,:,::-1]))

            elif key == 'x' or key =='X':
                action = 'Stop'
                mask = None
            elif key == 'i' or key =='I':
                action = 'Instruction'
                mask = None
            elif key == 'm' or key =='M':
                action = 'Manual'
                mask = None


            else:
                action = None
                mask = None

            return action, mask
                            



        # setup scene
        reward_type = 'dense'
        cls.setup_scene(env, traj_data, r_idx, args, reward_type=reward_type)
        _, _, _, _, _ = env.va_interact("Pass", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)

        # # goal instr
        # goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        # Our traj ; twoongg.kim
        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # goal_instr = traj_data['template']['task_desc']
        print()
        print('[', goal_instr, ']')
        for n, instr in enumerate(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
            print('  -', n+1, instr)
        print()
        
        print("******************************************User Manual******************************************")
        print("General      [I : Display Instructions,  M : Display User Manual")
        print()
        print("Action Space [W : GoForward,  A : RotateLeft, D : RotateRight, Q : LookDown,     E : LookUp")
        print("             [1 : PickUp Obj, 2 : PutObject,  3 : OpenObject,  4 : CloseObject")
        print("             [5 : Slice Obj,  6 : Toggle On,  7 : Toggle Off,  X : Stop (task is finished.)")


        done, success = False, False
        fails = 0
        t = 0
        reward = 0
        
            


        while not done:
            action = None
            while True:
                print('Waiting input')
                action, mask = get_key()                
                if action == "Instruction":
                    print()
                    print('[', goal_instr, ']')
                    for n, instr in enumerate(traj_data['turk_annotations']['anns'][r_idx]['high_descs']):
                        print('  -', n+1, instr)
                    print()
                if action == "Manual":
                    print("******************************************User Manual******************************************")
                    print("General      [I : Display Instructions,  M : Display User Manual")
                    print()
                    print("Action Space [W : GoForward,  A : RotateLeft, D : RotateRight, Q : LookDown,     E : LookUp")
                    print("             [1 : PickUp Obj, 2 : PutObject,  3 : OpenObject,  4 : CloseObject")
                    print("             [5 : Slice Obj,  6 : Toggle On,  7 : Toggle Off,  X : Stop (task is finished.)")
                if action !=None and action not in ["Manual", "Instruction"]:
                    break


                
            if action == 'Stop':
                print("Predicted Stop!")
                break
            # use predicted action and mask (if available) to interact with the env
            
            print(f"Performing : {action}")
            
            t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)
            if not t_success:
                print(err)
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break
            
            _, _, _, _, _ = env.va_interact("Pass", interact_mask=None, smooth_nav=args.smooth_nav, debug=args.debug)
            # next time-step
            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1

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
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
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
        print("SR: %d/%d = %.3f" % (results['all']['success']['num_successes'],
                                    results['all']['success']['num_evals'],
                                    results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (results['all']['goal_condition_success']['completed_goal_conditions'],
                                    results['all']['goal_condition_success']['total_goal_conditions'],
                                    results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (results['all']['path_length_weighted_goal_condition_success_rate']))
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

