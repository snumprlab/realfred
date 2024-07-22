import os, json
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


import numpy as np
from tqdm import tqdm

def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    return normalized_vectors

def calculate_cosine_similarity(embedding_vector, dictionary_vectors):
    similarities = np.dot(embedding_vector, dictionary_vectors.T)
    return similarities

def find_most_similar_keys(embedding_vector, embedding_dictionary, k=5):
    dictionary_vectors = np.array(list(embedding_dictionary.values()))
    similarities = calculate_cosine_similarity(embedding_vector, dictionary_vectors)
    most_similar_indices = np.argsort(similarities)[-k:][::-1]
    most_similar_keys = []
    for idx in most_similar_indices:        
        most_similar_keys.append(list(embedding_dictionary.keys())[idx])
    return most_similar_keys

def main(destination = 'all_exmaples'):
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # load dictionary with embeddings
    train_dict = pickle.load(open(f'{destination}/train_all_emb.p', 'rb'))
    sps = ['valid_seen', 'valid_unseen', 'tests_seen', 'tests_unseen']
    for sp in sps:
        jsons = json.load(open(f'{sp}_langs.json'))

        retrieved_keys = defaultdict(dict)

        for task, anns in tqdm(jsons.items()):
            for r in anns.keys():
                text = anns[r]
                output = model.encode(text)

                k = 9  
                similar_keys = find_most_similar_keys(output, train_dict, k)
                retrieved_keys[task][r] = similar_keys

        with open(f'{destination}/ReALFRED-{sp}_retrieved_keys.json', 'w') as f:
            json.dump(retrieved_keys, f, indent=4)

if __name__ == "__main":
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--dn', help='desination', default='all_examples', type=str)
    args = parser.parse_args()


    main(args.dn)