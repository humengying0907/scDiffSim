import os
import sys
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import random
import numpy as np
import anndata as ad
import pandas as pd
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.train_util import TrainLoop

from data_loader import *
from configs import model_and_diffusion_defaults

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(save_dir, prefix=""):
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # Configure logger to output to both stdout and a CSV file
    logger.configure(dir=log_dir, format_strs=["stdout", "csv"], log_suffix=prefix)


def train_diffusion(data,**args):
    setup_seed(1234)
    dist_util.setup_dist()

    model_and_diffusion_defaults_settings = model_and_diffusion_defaults()
    model_and_diffusion_defaults_settings['input_dim'] = args['n_latent']   # set this to the same value as n_latent in vae!
    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults_settings
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args['schedule_sampler'], diffusion)
    setup_logger(save_dir=args['save_dir'], prefix=args['diffusion_prefix'])

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args['batch_size'],
        microbatch=args['microbatch'],
        lr=args['lr'],
        ema_rate=args['ema_rate'],
        log_interval=args['log_interval'],
        save_percentages=args['save_percentages'],
        resume_checkpoint='',
        use_fp16=bool(args['use_fp16']),
        fp16_scale_growth=args['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args['weight_decay'],
        lr_anneal_steps=args['lr_anneal_steps'],
        save_dir=args['save_dir'],
        prefix = args['diffusion_prefix'],
    ).run_loop()


def main():
    parser = argparse.ArgumentParser(description="Parser for diffusion training")

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--n_latent', type=int, default=128, help="number of latent dimensions in vae and diffusion model")
    parser.add_argument('--colname_of_cellType', type=str, default='cell_type',help = 'colname indicating the cell types of single cells from the adata object')
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory where the trained diffusion model will be saved')

    # diffusion training parameters
    parser.add_argument('--lr_anneal_steps', type=int, default=500000, help="Number of steps for learning rate annealing")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size for diffusion model")
    parser.add_argument('--schedule_sampler', type=str, default='uniform', help="Schedule sampler for diffusion")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for diffusion model")
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument('--microbatch', type=int, default=-1, help="Microbatch size; -1 disables microbatching")
    parser.add_argument('--ema_rate', type=str, default='0.9999', help="Exponential moving average rate")
    parser.add_argument('--log_interval', type=int, default=500, help="Log interval for diffusion model training")
    parser.add_argument('--save_percentages', type=int, nargs='+', help="Percentages of training steps to save checkpoints. If specified, intermediate checkpoints of diffusion models can be randomly selected to guide future simulations")
    parser.add_argument('--use_fp16', type=int, choices=[0, 1], default=0, help="Enable (1) or disable (0) fp16 precision training")
    parser.add_argument('--fp16_scale_growth', type=float, default=1e-3, help="Growth factor for fp16 scaling")


    args = parser.parse_args()
    
    if not os.path.isdir(args.simu_obj_dir):
        raise ValueError(f"The provided simu_obj_dir '{args.simu_obj_dir}' is not a valid directory.")

    if not args.save_percentages:
        args.save_percentages = []
        
    if not os.path.exists(os.path.join(args.simu_obj_dir, "scvi_model.pt")):
        raise ValueError('A trained scVI model required for diffusion training was not found. Please train the scVI model first before diffusion training')
    
    args = vars(args)
    adata, _, classes = prepare_adata_for_augmentation(args['data_path'],
                                                       args['colname_of_cellType'],
                                                       'sample', # this parameter is not required when using the entire adata for scvi training and will be assigned a non-meaningful value
                                                       None)

    data = load_data_embedding(adata,classes,
                              args['simu_obj_dir'],
                              'scvi_',
                              args['batch_size'])


    args['save_dir'] = args['simu_obj_dir']
    args['diffusion_prefix'] = 'diffusion_'

    train_diffusion(data,**args)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
