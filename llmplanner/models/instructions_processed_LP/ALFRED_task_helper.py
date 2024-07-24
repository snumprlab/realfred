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
                return json.load(open("planner/planner_results/all_examples/turbo-bias-tests_unseen_result_template_executable.json", "rb"))
            else:
                return json.load(open("planner/planner_results/all_examples/turbo-bias-tests_seen_result_template_executable.json", "rb"))
        else:
            if unseen:
                assert()
                # return json.load(open("planner/planner_results/all_examples/turbo-bias-tests_unseen_result_template_executable_noAppend.json", "rb"))
            else:
                assert()
                # return json.load(open("planner/planner_results/all_examples/turbo-bias-tests_seen_result_template_executable_noAppend.json", "rb"))
                
    else:
        if appended:
            if unseen:
                return json.load(open("planner/planner_results/all_examples/turbo-bias-valid_unseen_result_template_executable.json", "rb"))
            else:
                return json.load(open("planner/planner_results/all_examples/turbo-bias-valid_seen_result_template_executable.json", "rb"))
        else:
            if unseen:
                assert()
                # return json.load(open("planner/planner_results/all_examples/turbo-bias-valid_unseen_result_template_executable_noAppend.json", "rb"))
            else:
                assert()
                # return json.load(open("planner/planner_results/all_examples/turbo-bias-valid_seen_result_template_executable_noAppend.json", "rb"))
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
    return instruction, None, None, None, None, None 
        

def get_arguments(traj_data):
    task_type = traj_data['task_type']
    try:
        r_idx = traj_data['ann']['repeat_idx']
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

        #if appended:
        instruction = instruction.lower()
        instruction = ''.join(ch for ch in instruction if ch not in exclude)
        language_goal, task_type, mrecep_target, obj_target, parent_target,  sliced  = get_arguments_test(test_dict, instruction)
            
    
    
    categories_in_inst = []
    list_of_highlevel_actions = []
    second_object = []
    caution_pointers = []
    sliced = 0

    categories_in_inst=list(set(test_dict['low_classes']))

    for category in categories_in_inst :
        if 'Sliced' in category :
            if category[0:-6] not in categories_in_inst :
                categories_in_inst.append(category[0:-6])
    

    for _,obj,recep in test_dict['triplet'] :
        if obj  not in categories_in_inst :
            categories_in_inst.append(obj)   
        if recep  not in categories_in_inst :
            categories_in_inst.append(recep)

            
    for idx in range(len(test_dict['low_actions'])):
        list_of_highlevel_actions.append([test_dict['low_classes'][idx], test_dict['low_actions'][idx]])
        if test_dict['low_actions'][idx] == 'SliceObject':
            sliced=1
        if test_dict['low_actions'][idx] == 'PutObject':
            caution_pointers.append(len(list_of_highlevel_actions)-1)


    
    #return [(goal_category, interaction), (goal_category, interaction), ...]
    print("instruction goal is ", language_goal)
    #list_of_highlevel_actions = [ ('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject')]
    #list_of_highlevel_actions = [('Microwave', 'OpenObject'), ('Microwave', 'PutObject'), ('Microwave', 'CloseObject'), ('Microwave', 'ToggleObjectOn'), ('Microwave', 'ToggleObjectOff'), ('Microwave', 'OpenObject'), ('Apple', 'PickupObject'), ('Microwave', 'CloseObject'), ('Fridge', 'OpenObject'), ('Fridge', 'PutObject'), ('Fridge', 'CloseObject')]
    #categories_in_inst = ['Microwave', 'Fridge']
    return list_of_highlevel_actions, categories_in_inst, second_object, caution_pointers, sliced