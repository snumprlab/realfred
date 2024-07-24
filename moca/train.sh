export ALFRED_ROOT=$(pwd)
python models/train/train_seq2seq.py --gpu --dout exp/moca --save_every_epoch
