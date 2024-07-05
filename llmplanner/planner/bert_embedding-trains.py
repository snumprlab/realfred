import os, json
import torch
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
import numpy as np

device = 'cuda'
model = SentenceTransformer('bert-base-nli-mean-tokens').to(device)
train_dict = pickle.load(open('all_examples/train_all_emb.p', 'rb'))

for k in tqdm(train_dict.keys()):
    text = str(k)
    output = model.encode(text)
    train_dict[k] = output

pickle.dump(train_dict, open('all_examples/train_all_emb.p', 'wb'))