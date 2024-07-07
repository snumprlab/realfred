GPU=$1
FROM=$2
TO=$3
DEF=$4

CUDA_VISIBLE_DEVICES=$GPU python main/rollout_and_evaluate.py\
        --from_idx $FROM \
        --to_idx $TO  \
        --def_name $DEF