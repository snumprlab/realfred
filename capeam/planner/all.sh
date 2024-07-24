CUDA_VISIBLE_DEVICES=0 python main_meta.py >> log_meta.txt &

CUDA_VISIBLE_DEVICES=1 python main_subpolicy.py --subgoal CleanObject --rawClass >> log_Clean.txt & 
CUDA_VISIBLE_DEVICES=2 python main_subpolicy.py --subgoal CoolObject --rawClass >> log_Cool.txt & 
CUDA_VISIBLE_DEVICES=3 python main_subpolicy.py --subgoal HeatObject --rawClass >> log_Heat.txt & 
CUDA_VISIBLE_DEVICES=4 python main_subpolicy.py --subgoal PickupObject --rawClass >> log_Pick.txt & 
CUDA_VISIBLE_DEVICES=5 python main_subpolicy.py --subgoal PutObject --rawClass >> log_Put.txt & 
CUDA_VISIBLE_DEVICES=6 python main_subpolicy.py --subgoal SliceObject --rawClass >> log_Slice.txt & 
CUDA_VISIBLE_DEVICES=7 python main_subpolicy.py --subgoal ToggleObject --rawClass >> log_Toggle.txt