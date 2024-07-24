import os
import sys
import json
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import torch
import os
from PIL import Image
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
from tqdm import tqdm


# img   : PIL.Image
# rtype : str
def randomize(img, rtype):
    if rtype == 'rgb':
        return img
    elif rtype == 'color_swap':
        return Image.fromarray(np.asarray(img)[:,:,np.random.permutation(3)])
    else:
        return img

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--data', help='data folder', default='data/Re_json_feat1.1.0')
    parser.add_argument('--batch', help='batch size', default=128, type=int)
    parser.add_argument('--gpu', help='use gpu', action='store_true', default=True)
    parser.add_argument('--visual_model', default='resnet18', help='model type: maskrcnn or resnet18', choices=['resnet18', 'maskrcnn'])
    parser.add_argument('--filename', help='filename of feat', default='feat_conv_panoramic.pt')
    parser.add_argument('--img_folder', help='folder containing raw images', default='raw_images_panoramic')
    parser.add_argument('--randomization', help='type of randomization', choices=['rgb', 'color_swap', 'auto_aug', 'randaug'], default='rgb')
    parser.add_argument('--skip_existing', help='skip folders that already have feats', action='store_true')


    # parser
    args = parser.parse_args()

    # load resnet model
    autoaug = args.randomization == 'auto_aug'
    randaug = args.randomization == 'randaug'
    extractor = Resnet(args, eval=True, autoaug=autoaug, randaug = randaug)
    
    
    for s in ['train']:#, 'valid_seen', 'valid_unseen']:
        for ep in tqdm(os.listdir(os.path.join(args.data, s))):
            for trial in os.listdir(os.path.join(args.data, s, ep)):
                pp = os.path.join(os.path.join(args.data, s, ep), trial)
                if not os.path.isfile(os.path.join(pp, 'augmented_traj_data.json')):
                    continue # Panoramic image generation in progress...
                if args.skip_existing and os.path.isfile(os.path.join(pp, args.filename)):
                    continue

                img_root = os.path.join(args.data, s, ep, trial, args.img_folder)

                ds = ['left', 'up', 'front', 'down', 'right']

                imgs = {}
                for i, d in enumerate(ds):
                    imgs[d] = [randomize(Image.open(os.path.join(img_root, p)).crop((i*300,0,(i+1)*300,300)), args.randomization) for p in sorted(os.listdir(img_root))]

                feats = {}
                for i, d in enumerate(ds):
                    feats[d] = extractor.featurize(imgs[d], batch=args.batch)
                feat = torch.stack([feats[d] for d in ds], dim=0)

                torch.save(feat.cpu(), os.path.join(args.data, s, ep, trial, args.filename))
