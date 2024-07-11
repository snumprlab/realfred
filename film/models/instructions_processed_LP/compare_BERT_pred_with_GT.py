import pickle

splits = ["valid_seen","valid_unseen","tests_seen", "tests_unseen"]
split = "valid_unseen"

pred = pickle.load(open('instruction2_params_'+split+'_appended_new_split_oct24.p','rb'))
gt = pickle.load(open('instruction2_params_'+split+'_new_split_GT_oct24.p','rb'))

task_desc_list = list(pred.keys())
# gt = list(gt.items())
task_types = {"pick_cool_then_place_in_recep": 0, "pick_and_place_with_movable_recep": 1, "pick_and_place_simple": 2, "pick_two_obj_and_place": 3, "pick_heat_then_place_in_recep": 4, "look_at_obj_in_light": 5, "pick_clean_then_place_in_recep": 6}

c0=0; c1=0; c2=0; c3=0; c4=0; c5=0; c6=0
nt=0; nm=0; no=0; np=0; ns=0; nt=0
parent_pred = []; parent_gt=[]; parents = []
# num_of_tables_in_parent = 0
for i in range(len(pred)):
    n = 0
    task = task_desc_list[i]
    print("##########################################################################")
    print(task)
    if pred[task]['task_type'] == gt[task]['task_type']:
        nt += 1; n += 1
    else:
        print('Task type: ', pred[task]['task_type'], gt[task]['task_type'])
        
    if pred[task]['mrecep_target'] == gt[task]['mrecep_target']:
        nm += 1; n += 1
    else:
        print('mrecep: ', pred[task]['mrecep_target'], gt[task]['mrecep_target'])
        
    if pred[task]['object_target'] == gt[task]['object_target']:
        no += 1; n += 1
    else:
        print('object_target: ', pred[task]['object_target'], gt[task]['object_target'])
        
    if pred[task]['parent_target'] == gt[task]['parent_target']:
        np += 1; n += 1

    else:
        print('parent_target: ', pred[task]['parent_target'], gt[task]['parent_target'])
        parent_pred.append(pred[task]['parent_target'])
        parent_gt.append(gt[task]['parent_target'])
        parents.append([pred[task]['parent_target'],gt[task]['parent_target']])
        
    if pred[task]['sliced'] == gt[task]['sliced']:
        ns += 1; n += 1
    else:
        print('sliced: ', pred[task]['sliced'], gt[task]['sliced'])
    # if pred[task]['toggle_target'] == gt[task]['toggle_target']:
    #     nt += 1; n += 1
    print("##########################################################################")

    if n == 0:
        c0 += 1
    elif n == 1:
        c1 += 1
    elif n == 2:
        c2 += 1
    elif n == 3:
        c3 += 1
    elif n == 4:
        c4 += 1
    elif n == 5:
        c5 += 1

print("################################################")
print(split)
print(f"Total number of task_desc: {len(pred), len(gt)}\n")
print(f"task_type accuracy: {round(nt/len(pred)*100, 2)}")
print(f"mrecep_target accuracy: {round(nm/len(pred)*100, 2)}")
print(f"object_target accuracy: {round(no/len(pred)*100, 2)}")
print(f"parent_target accuracy: {round(np/len(pred)*100, 2)}")
print(f"sliced accuracy: {round(ns/len(pred)*100, 2)}")
print()
print(f"0 Correct: {round(c0/len(pred)*100, 2)}")
print(f"1 Correct: {round(c1/len(pred)*100, 2)}")
print(f"2 Correct: {round(c2/len(pred)*100, 2)}")
print(f"3 Correct: {round(c3/len(pred)*100, 2)}")
print(f"4 Correct: {round(c4/len(pred)*100, 2)}")
print(f"5 Correct: {round(c5/len(pred)*100, 2)}")
print('\nTotal all correct accuracy: ', str(round(c5/len(pred)*100, 2)))
print("################################################")