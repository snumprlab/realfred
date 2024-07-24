(bash rollout_and_evaluate.sh 0 0 300 "alfred/eval/hlsm_full/eval_hlsm_valid_unseen" ) &
(bash rollout_and_evaluate.sh 1 300 600 "alfred/eval/hlsm_full/eval_hlsm_valid_unseen" ) &
(bash rollout_and_evaluate.sh 2 600 900 "alfred/eval/hlsm_full/eval_hlsm_valid_unseen" ) &
(bash rollout_and_evaluate.sh 3 900 1200 "alfred/eval/hlsm_full/eval_hlsm_valid_unseen" )