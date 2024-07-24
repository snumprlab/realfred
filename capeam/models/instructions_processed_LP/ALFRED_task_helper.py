#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 16:49:38 2021

@author: soyeonmin
"""
import pickle
import alfred_utils.gen.constants as constants
import string
import json

exclude = set(string.punctuation)
task_type_dict = {2: 'pick_and_place_simple',
 5: 'look_at_obj_in_light',
 1: 'pick_and_place_with_movable_recep',
 3: 'pick_two_obj_and_place',
 6: 'pick_clean_then_place_in_recep',
 4: 'pick_heat_then_place_in_recep',
 0: 'pick_cool_then_place_in_recep'}


def read_test_dict(test, appended, unseen):
    if test:
        if appended:
            if unseen:
                return json.load(open("models/pretrained/test_unseen.json"))
            else:
                return json.load(open("models/pretrained/test_seen.json"))
        else:
            assert()
            # if unseen:
            #     return json.load(open("models/pretrained/test_unseen-noappend.json"))

            # else:
            #     return json.load(open("models/pretrained/test_seen-noappend.json"))
                
    else:
        if appended:
            if unseen:
                return json.load(open("models/pretrained/valid_unseen.json"))
            else:
                return json.load(open("models/pretrained/valid_seen.json"))
        else:
            assert()
            # if unseen:
            #     return json.load(open("models/pretrained/valid_unseen-noappend.json"))
            
            # else:
            #     return json.load(open("models/pretrained/valid_seen-noappend.json"))

def exist_or_no(string):
    if string == '' or string == False:
        return 0
    else:
        return 1

def none_or_str(string):
    if string == '':
        return None
    else:
        return string

def get_arguments_test(test_dict, instruction):
    task_type, mrecep_target, object_target, parent_target, sliced = \
        test_dict[instruction]['task_type'],  test_dict[instruction]['mrecep_target'], test_dict[instruction]['object_target'], test_dict[instruction]['parent_target'],\
             test_dict[instruction]['sliced']

    if isinstance(task_type, int):
        task_type = task_type_dict[task_type]
    return instruction, task_type, mrecep_target, object_target, parent_target, sliced 
        

def get_arguments(traj_data):
    task_type = traj_data['task_type']
    try:
        r_idx = traj_data['repeat_idx']
    except:
        r_idx = 0
    language_goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
    
    sliced = exist_or_no(traj_data['pddl_params']['object_sliced'])
    mrecep_target = none_or_str(traj_data['pddl_params']['mrecep_target'])
    object_target = none_or_str(traj_data['pddl_params']['object_target'])
    parent_target = none_or_str(traj_data['pddl_params']['parent_target'])
    
    return language_goal_instr, task_type, mrecep_target, object_target, parent_target, sliced

def add_target(target, target_action, list_of_actions):
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "OpenObject"))
    list_of_actions.append((target, target_action))
    if target in [a for a in constants.OPENABLE_CLASS_LIST if not(a == 'Box')]:
        list_of_actions.append((target, "CloseObject"))
    return list_of_actions

def determine_consecutive_interx(list_of_actions, previous_pointer, sliced=False):
    returned, target_instance = False, None
    if previous_pointer <= len(list_of_actions)-1:
        if list_of_actions[previous_pointer][0] == list_of_actions[previous_pointer+1][0]:
            returned = True
            #target_instance = list_of_target_instance[-1] #previous target
            target_instance = list_of_actions[previous_pointer][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "OpenObject" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
        #Micorwave or Fridge
        elif list_of_actions[previous_pointer][1] == "PickupObject" and list_of_actions[previous_pointer+1][1] == "CloseObject":
            returned = True
            #target_instance = list_of_target_instance[-2] #e.g. Fridge
            target_instance = list_of_actions[previous_pointer-1][0]
        #Faucet
        elif list_of_actions[previous_pointer+1][0] == "Faucet" and list_of_actions[previous_pointer+1][1] in ["ToggleObjectOn", "ToggleObjectOff"]:
            returned = True
            target_instance = "Faucet"
        #Pick up after faucet 
        elif list_of_actions[previous_pointer][0] == "Faucet" and list_of_actions[previous_pointer+1][1] == "PickupObject":
            returned = True
            #target_instance = list_of_target_instance[0]
            target_instance = list_of_actions[0][0]
            if sliced:
                #target_instance = list_of_target_instance[3]
                target_instance = list_of_actions[3][0]
    return returned, target_instance


def get_list_of_highlevel_actions(traj_data, test=False, test_dict=None, args_nonsliced=False, appended=False):
    if not(test):
        language_goal, task_type, mrecep_target, obj_target, parent_target,  sliced = get_arguments(traj_data)
    if test:
        r_idx = traj_data['ann']['repeat_idx']
        instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        language_goal = instruction


    list_of_highlevel_actions = []
    second_object = []
    caution_pointers = []
    convert_pointers = []


    triplet_high_action =[]
    slice_idx = []
    pickup_idx = []
    triplet_low_action =[]
    low_actions =[]
    low_classes =[]
    high_idxs = []
    categories_in_inst=list(set(test_dict['low_classes']))
    sliced = 0
    check_put =[]
    last_action = '0'

    for category in categories_in_inst :
        if 'Sliced' in category :
            if category[0:-6] not in categories_in_inst :
                categories_in_inst.append(category[0:-6])
    

    for _,obj,recep in test_dict['triplet'] :
        if obj  not in categories_in_inst :
            categories_in_inst.append(obj)   
        if recep  not in categories_in_inst :
            categories_in_inst.append(recep)                        

    triplet_high_action= test_dict['triplet'] 


    for idx, low_action in enumerate( test_dict['low_actions'] ):
        if  low_action!='GotoLocation' :
            low_actions.append(low_action)
            high_idxs.append(test_dict['high_idxs'][idx])
            if test_dict['low_classes'][idx] =='Sink':
                low_classes.append('SinkBasin')
            elif test_dict['low_classes'][idx] =='Bathtub':
                low_classes.append('BathtubBasin')
            else :
                low_classes.append(test_dict['low_classes'][idx])
        if low_action == 'SliceObject':
            sliced=1

    new_low_action_idx =0
    low_action_idx =0
    high_action_idx =0
    flag= True
    for action,obj,recep in triplet_high_action :
        sequence =high_idxs.count(high_action_idx)
        if action in ['CoolObject','HeatObject','CleanObject','PutObject'] :
            caution_pointers.append(new_low_action_idx)
        elif recep in constants.OPENABLE_CLASS_LIST and recep!='Box':
            caution_pointers.append(new_low_action_idx)
            if action in ['PickupObject','SliceObject'] :
                convert_pointers.append(new_low_action_idx)

        for step in range(sequence):
            if high_action_idx in slice_idx :
                if flag :
                    triplet_low_action.append(['SinkBasin', 'PutObject',obj,'SinkBasin',high_action_idx])
                    second_object.append(False)
                    flag = False
                else :
                    new_low_action_idx-=1
            else:
                if high_action_idx not in slice_idx :
                    flag =True
                if high_action_idx in pickup_idx and low_actions[low_action_idx] =="CloseObject" :
                    new_low_action_idx-=1
                    pass
                else:
                    second_object.append(False)
                    triplet_low_action.append([low_classes[low_action_idx],low_actions[low_action_idx],obj,recep,high_action_idx])
            if len(triplet_low_action)>0 :
                if triplet_low_action[new_low_action_idx][1] == 'PutObject' :
                    if obj+recep in check_put:
                        for second_idx in range (new_low_action_idx,0,-1) :
                            if triplet_low_action[second_idx][1]=='PickupObject' and  triplet_low_action[second_idx][0] == obj:
                                second_object[second_idx] =True
                                break
                    else:
                        check_put.append(obj+recep)            

            low_action_idx +=1
            new_low_action_idx +=1
        high_action_idx+=1

    list_of_highlevel_actions =triplet_low_action
    print("instruction goal is ", language_goal)
    for action_len in range(len(list_of_highlevel_actions)):
        print('list_of_highlevel_actions:', list_of_highlevel_actions[action_len])

    # print(second_object)
    return list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers, convert_pointers,sliced