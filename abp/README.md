# ReALFRED - ABP
This is the code repository of the paper [Agent with the Big Picture: Perceiving Surroundings for Interactive Instruction Following](https://embodied-ai.org/papers/Agent-with-the-Big-Picture.pdf) for reproduction in [ReALFRED](https://github.com/snumprlab/realfred). \
Our code is largely built upon the codebase from [ABP](https://github.com/snumprlab/abp).


## Environment
### Clone repository
```
$ git clone https://github.com/snumprlab/realfred.git
$ cd realfred/abp
$ export ALFRED_ROOT=$(pwd)
```

### Install requirements
```
$ conda create -n reabp python=3.6
$ conda activate reabp

$ cd $ALFRED_ROOT
$ pip install --upgrade pip
$ pip install -r requirements.txt
```
You also need to install Pytorch depending on your system. e.g) PyTorch v1.10.0 + cuda 11.1 <br>
Refer [here](https://pytorch.kr/get-started/previous-versions/)
```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```



## Download
Download the ResNet-18 features and annotation files from <a href="https://huggingface.co/datasets/SNUMPR/realfred_feat">the Hugging Face repo</a>.
<br>
**Note**: It takes quite a large space (~2.3TB).
```
git clone https://huggingface.co/datasets/SNUMPR/realfred_feat data
```

## Training
To train ABP, run `train_seq2seq.py` with hyper-parameters below. <br>
```
python models/train/train_seq2seq.py --data <path_to_dataset> --model seq2seq_im_mask --dout <path_to_save_weight> --splits data/splits/oct24.json --gpu --batch <batch_size> --pm_aux_loss_wt <pm_aux_loss_wt_coeff> --subgoal_aux_loss_wt <subgoal_aux_loss_wt_coeff>
```
**Note**: As mentioned in the repository of <a href="https://github.com/askforalfred/alfred/tree/master/models">ALFRED</a>, run with `--preprocess` only once for preprocessed json files. 
**Note**: All hyperparameters used for the experiments in the paper are set as default.

For example, if you want train ABP and save the weights for all epochs in "exp/abp" with all hyperparameters used in the experiments in the paper, you may use the command below <br>
```
python models/train/train_seq2seq.py --gpu --dout exp/abp --save_every_epoch
```
or simply just run
```
bash train.sh
```
**Note**: The option, `--save_every_epoch`, saves weights for all epochs and therefore could take a lot of space.


## Evaluation
### Task Evaluation
To evaluate ABP, run `eval_seq2seq.py` with hyper-parameters below. <br>
To evaluate a model in the `seen` or `unseen` environment, pass `valid_seen` or `valid_unseen` or `tests_seen` or `tests_unseen` to `--eval_split`.
```
python models/eval/eval_seq2seq.py --data <path_to_dataset> --model models.model.seq2seq_im_mask --model_path <path_to_weight> --eval_split <eval_split> --gpu --num_threads <thread_num>
```
**Note**: All hyperparameters used for the experiments in the paper are set as default.

If you want to evaluate our pretrained model saved in `exp/pretrained/pretrained.pth` in the `seen` validation, you may use the command below.
```
python models/eval/eval_seq2seq.py --model_path "exp/pretrained/pretrained.pth" --eval_split valid_seen --gpu --num_threads 4
```


## Hardware 
Trained and Tested on:
- **GPU** - RTX A6000
- **CPU** - Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz
- **RAM** - 64GB
- **OS** - Ubuntu 20.04


## Citation
**ReALFRED**
```
@inproceedings{kim2024realfred,
  author    = {Kim, Taewoong and Min, Cheolhong and Kim, Byeonghwi and Kim, Jinyeon and Jeung, Wonje and Choi, Jonghyun},
  title     = {ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environment},
  booktitle = {ECCV},
  year      = {2024}
  }
```
**ABP**
```
@inproceedings{kim2021agent,
  author    = {Kim, Byeonghwi and Bhambri, Suvaansh and Singh, Kunal Pratap and Mottaghi, Roozbeh and Choi, Jonghyun},
  title     = {Agent with the Big Picture: Perceiving Surroundings for Interactive Instruction Following},
  booktitle = {Embodied AI Workshop @ CVPR 2021},
  year      = {2021},
}
```