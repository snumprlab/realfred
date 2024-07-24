import os
import torch
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def main(destination = 'all_exmaples'):
    device = 'cuda'
    model = SentenceTransformer('bert-base-nli-mean-tokens').to(device)
    try:
        os.makedirs(destination, exist_ok=True)
        command = f'cp all_exmaples/train_all_emb.p {destination}/train_all_emb.p'
        os.system(command)
    except:
        pass
    train_dict = pickle.load(open(f'{destination}/train_all_emb.p', 'rb'))

    for k in tqdm(train_dict.keys()):
        text = str(k)
        output = model.encode(text)
        train_dict[k] = output

    pickle.dump(train_dict, open(f'{destination}/train_all_emb.p', 'wb'))


if __name__ == "__main__":
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--dn', help='desination', default='all_examples', type=str)
    args = parser.parse_args()


    main(args.dn)