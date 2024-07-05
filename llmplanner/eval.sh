SPLIT=$1
FROM_IDX=$2
TO_IDX=$3

python main.py \
        --max_episode_length 1000       \
        --num_local_step 25     \
        --num_processes 1       \
        --eval_split $SPLIT     \
        --from_idx $FROM_IDX    \
        --to_idx $TO_IDX        \
        --max_fails 10  \
        --debug_local   \
        --set_dn realfred-llmplanner-${SPLIT} \
        --which_gpu 0   \
        --sem_gpu_id 0  \
        --sem_seg_gpu 0 \
        --depth_gpu 0   \
        --x_display 1   \
        --data "alfred_data_all/Re_json_2.1.0"   \
        --splits "alfred_data_small/splits/oct24.json" \
        --learned_depth \
        --use_sem_policy  \
        --appended \
        --use_sem_seg   \
        --appended \


