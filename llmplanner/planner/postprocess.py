import os, json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser



def main(sp, destination):
    path = f'planner_results/{destination}/turbo-bias-{sp}_result.json'
    result = json.load(open(path))
    for k in result.keys():    
        result[k]['triplet'][0] = result[k]['triplet'][0][1:]
        

    for k in result.keys():    
        tmp = []
        for i in range(len(result[k]['triplet'][0].split(','))):
            if 'Object' in result[k]['triplet'][0].split(',')[i]:
                action = result[k]['triplet'][0].split(',')[i].strip()
                obj = result[k]['triplet'][0].split(',')[i+1].strip()
                try:
                    recep = result[k]['triplet'][0].split(',')[i+2].strip()
                except:
                    recep = "0"
                tmp.append([action, obj, recep])
        result[k]['triplet_beforeDC'] = tmp
        result[k]['triplet'] = tmp


    OPENABLE = ['Fridge', 'Cabinet', 'Microwave', 'Drawer', 'Safe']
    for k in result.keys():    
        tmp_actions = []
        tmp_classes = []
        tmp_idxs = []
        skipped_cnt=0
        DC_indices = []
        for i in range(len(result[k]['triplet_beforeDC'])):
            action = result[k]['triplet_beforeDC'][i][0]
            target = result[k]['triplet_beforeDC'][i][1]
            recp = result[k]['triplet_beforeDC'][i][2]
            if action == 'HeatObject':
                tmp_actions.append('OpenObject')
                tmp_actions.append('PutObject')
                tmp_actions.append('CloseObject')
                tmp_actions.append('ToggleObjectOn')
                tmp_actions.append('ToggleObjectOff')
                tmp_actions.append('OpenObject')
                tmp_actions.append('PickupObject')
                tmp_actions.append('CloseObject')
                
                tmp_classes.append(recp) #Microwave
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(target)
                tmp_classes.append(recp)
                for _ in range(8):
                    tmp_idxs.append(i-skipped_cnt)

                
            elif action == 'CoolObject':
                tmp_actions.append('OpenObject')
                tmp_actions.append('PutObject')
                tmp_actions.append('CloseObject')
                tmp_actions.append('OpenObject')
                tmp_actions.append('PickupObject')
                tmp_actions.append('CloseObject')
                
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(target)
                tmp_classes.append(recp)
                for _ in range(6):
                    tmp_idxs.append(i-skipped_cnt)


            
            elif action == 'CleanObject':
                tmp_actions.append('PutObject')
                tmp_actions.append('ToggleObjectOn')
                tmp_actions.append('ToggleObjectOff')
                tmp_actions.append('PickupObject')
                
                tmp_classes.append('Sink')
                tmp_classes.append(recp)
                tmp_classes.append(recp)
                tmp_classes.append(target)
                for _ in range(4):
                    tmp_idxs.append(i-skipped_cnt)

            elif action == 'ToggleObject':
                tmp_actions.append('ToggleObjectOn')
                
                tmp_classes.append(target)
                for _ in range(1):
                    tmp_idxs.append(i-skipped_cnt)
                    
            elif action == 'PickupObject':
                if recp in OPENABLE:
                    tmp_actions.append('OpenObject')            
                    tmp_actions.append('PickupObject')
                    tmp_actions.append('CloseObject')            
                    tmp_classes.append(recp)
                    tmp_classes.append(target)
                    tmp_classes.append(recp)
                    for _ in range(3):
                        tmp_idxs.append(i-skipped_cnt)
                
                elif target == 'Laptop':  # Manually postprocess Laptop
                    tmp_actions.append('CloseObject')
                    tmp_actions.append('PickupObject')
                    tmp_classes.append(target)
                    tmp_classes.append(target)
                    for _ in range(2):
                        tmp_idxs.append(i-skipped_cnt)

                else:
                    tmp_actions.append('PickupObject')
                    tmp_classes.append(target)
                    for _ in range(1):
                        tmp_idxs.append(i-skipped_cnt)

            
            elif action == 'SliceObject':
                tmp_actions.append('SliceObject')
                
                tmp_classes.append(target)
                for _ in range(1):
                    tmp_idxs.append(i-skipped_cnt)
                    
            elif action == 'PutObject':
                if recp in OPENABLE:
                    tmp_actions.append('OpenObject')            
                    tmp_actions.append('PutObject')
                    tmp_actions.append('CloseObject')            
                    tmp_classes.append(recp)
                    tmp_classes.append(recp)
                    tmp_classes.append(recp)
                    for _ in range(3):
                        tmp_idxs.append(i-skipped_cnt)
                else:
                    tmp_actions.append('PutObject')
                    tmp_classes.append(recp)
                    for _ in range(1):
                        tmp_idxs.append(i-skipped_cnt)
            else:
                print(action, 'not supported')
                skipped_cnt+=1
            result[k]['low_actions'] = tmp_actions
            result[k]['low_classes'] = tmp_classes
            result[k]['high_idxs'] = tmp_idxs
        for j in reversed(DC_indices):
            del result[k]['triplet'][j]

    with open(f'turbo-bias-{sp}_result_template_executable.json', 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--dn', help='desination', default='all_examples', type=str)
    args = parser.parse_args()

    for split in ['tests_seen', 'tests_unseen', 'valid_seen', 'valid_unseen']:
        main(split, args.dn)
