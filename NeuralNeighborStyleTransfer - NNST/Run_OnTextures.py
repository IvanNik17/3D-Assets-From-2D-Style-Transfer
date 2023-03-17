# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 08:53:59 2022

@author: IvanTower
"""


# Core Imports
import time
import argparse
import random

# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utils import misc as misc
from utils.misc import load_path_for_pytorch
from utils.stylize import produce_stylization

import os
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
     '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Fix Random Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Define command line parser and get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--content_path'   , type=str, default="./inputs", required=True)
parser.add_argument('--style_path'     , type=str, default=None, required=True)
parser.add_argument('--output_path'    , type=str, default="./outputs", required=True)
parser.add_argument('--high_res'       , action='store_true'                  )
parser.add_argument('--cpu'            , action='store_true'                  )
parser.add_argument('--no_flip'        , action='store_true'                  )
parser.add_argument('--content_loss'   , action='store_true'                  )
parser.add_argument('--dont_colorize'  , action='store_true'                  )
parser.add_argument('--alpha'          , type=float, default=0.75             )
args = parser.parse_args()




inputDir = args.content_path
outputDir = args.output_path



# Interpret command line arguments
# content_path = args.content_path
style_path = args.style_path
# output_path = args.output_path
max_scls = 4
sz = 512
if args.high_res:
    max_scls = 5
    sz = 1024
flip_aug = (not args.no_flip)
content_loss = args.content_loss
misc.USE_GPU = (not args.cpu)
content_weight = 1. - args.alpha

# Error checking for arguments
# error checking for paths deferred to imageio
assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

# Define feature extractor
cnn = misc.to_device(Vgg16Pretrained())
phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

# Load images
style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)


for path, subdirs, files in os.walk( os.path.join(".", inputDir), topdown=True):
            
    files.sort(key=natural_keys)
    
    curr_output_dir = os.path.join(outputDir, path.split("\\")[-1])
    
    if not os.path.isdir(curr_output_dir):
            os.makedirs(curr_output_dir)

    for curr_file in files:
        
        
        # if os.path.isfile(os.path.join(curr_output_dir, f'output_{curr_file}')):
        #     print(f"file {curr_file} exists -  skip")
            
        #     # print(style_path.split("/")[-1].split(".")[0])
            
        # else:
            
            
        
        print(f"running {path} {curr_file}")
        
        curr_path = os.path.join(path, curr_file)
        content_im_orig = misc.to_device(load_path_for_pytorch(curr_path, target_size=sz)).unsqueeze(0)
        
        
        # Run Style Transfer
        torch.cuda.synchronize()
        start_time = time.time()
        output = produce_stylization(content_im_orig, style_im_orig, phi,
                                    max_iter=200,
                                    lr=2e-3,
                                    content_weight=content_weight,
                                    max_scls=max_scls,
                                    flip_aug=flip_aug,
                                    content_loss=content_loss,
                                    dont_colorize=args.dont_colorize)
        torch.cuda.synchronize()
        print('Done! total time: {}'.format(time.time() - start_time))
        
        # Convert from pyTorch to numpy, clip to valid range
        new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)
        
        # Save stylized output
        save_im = (new_im_out * 255).astype(np.uint8)
        # imwrite(os.path.join(curr_output_dir, f'output_{curr_file}'), save_im)
        # imwrite(os.path.join(curr_output_dir, f'output_{curr_file}_{style_path.split("/")[-1].split(".")[0]}.jpg'), save_im)

        imwrite(os.path.join(curr_output_dir, f'{curr_file}'), save_im)




# Free gpu memory in case something else needs it later
if misc.USE_GPU:
    torch.cuda.empty_cache()
