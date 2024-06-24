#export DISPLAY=:0.0
export ALFRED_ROOT=$(pwd)

GPU=0
SPLITS=('valid_unseen' 'valid_seen')
WEIGHT_PATH='moca'
mkdir logs
mkdir "logs/${WEIGHT_PATH}"
cp get_scores.py logs/"${WEIGHT_PATH}"
for ((epoch=0; epoch<50; epoch++))

for split in ${SPLITS[@]}
    do
        for ((;;))
        do
            if [ -f "exp/${WEIGHT_PATH}/net_epoch_${epoch}.pth" ]
            then
                echo "epoch ${epoch} found. start ${split} evaluation"
                sleep 60
                break
            fi
        done
        CUDA_VISIBLE_DEVICES=$GPU python models/eval/eval_seq2seq.py    \
            --model_path "exp/${WEIGHT_PATH}/net_epoch_${epoch}.pth"    \
            --eval_split "$split"      \
            --gpu       \
            --max_steps 1000 \
            --max_fails 10 \
            --num_threads 4 >> "logs/${WEIGHT_PATH}/log_${epoch}_${split}.txt"
    done
done
echo "Done"
