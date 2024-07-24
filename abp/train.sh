export ALFRED_ROOT=$(pwd)

python models/train/train_seq2seq.py --dout exp/abp --subgoal_aux_loss_wt 0 --lr 1e-3 --gpu --save_every_epoch --panoramic --panoramic_concat