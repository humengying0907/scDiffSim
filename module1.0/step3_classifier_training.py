import os
import sys
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

from simple_scvi._mymodel import MyModel

import anndata as ad
import scanpy as sc
import blobfile as bf
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_classifier_and_diffusion
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
import torch
import torch.nn as nn
import numpy as np

from data_loader import *
from configs import classifier_and_diffusion_defaults, classifier_requirements_defaults

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
        
def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)

def save_model(mp_trainer, opt, step, save_dir, model_prefix=""):
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        model_file = os.path.join(save_dir, f"{model_prefix}model{step:06d}.pt")
        torch.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            model_file,
        )
        if opt is not None:  # Only save the optimizer state if it's provided
            opt_file = os.path.join(save_dir, f"{model_prefix}opt{step:06d}.pt")
            torch.save(opt.state_dict(), opt_file)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(save_dir, prefix=""):
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    # Configure logger to output to both stdout and a CSV file
    logger.configure(dir=log_dir, format_strs=["stdout", "csv"], log_suffix=prefix)


def train_classifier(data, num_classes,save_dir,classifier_prefix,**args):

    setup_seed(1234)
    val_data = None  # not considering validation dataset for now

    dist_util.setup_dist()

    classifier_and_diffusion_default_settings = classifier_and_diffusion_defaults()
    classifier_and_diffusion_default_settings['num_class'] = num_classes

    model, diffusion = create_classifier_and_diffusion(
        **classifier_and_diffusion_default_settings
    )
    
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(
        args['schedule_sampler'], diffusion
    )
        
    resume_step = 0
    
    # resume_checkpoint is set to '' therefore will skip everything related
    dist_util.sync_params(model.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=bool(args['classifier_use_fp16']), initial_lg_loss_scale=16.0
    )
    setup_logger(save_dir=save_dir, prefix=classifier_prefix)

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=True,
    )

    opt = AdamW(mp_trainer.master_params, lr=args['lr'], weight_decay=args['weight_decay'])

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())

        batch = batch.to(dist_util.dev())
        # Noisy cells
        if bool(args['noised']):
            t, _ = schedule_sampler.sample(
                batch.shape[0], dist_util.dev(), start_guide_time=args['start_guide_time']
            )
            batch = diffusion.q_sample(batch, t)
        else:
            t = torch.zeros(batch.shape[0], dtype=torch.long, device=dist_util.dev())

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args['microbatch'], batch, labels, t)
        ):
            logits = model(sub_batch, sub_t)
            loss = F.cross_entropy(logits, sub_labels, reduction="none")

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()
            losses[f"{prefix}_acc@1"] = compute_top_k(
                logits, sub_labels, k=1, reduction="none"
            )

            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))
    
    save_percentages = args['save_percentages']
    save_steps = [(p / 100) * args['iterations'] for p in save_percentages]
    save_optimizer = bool(args['save_optimizer'])

    print('start classifier training \n')
    
    for step in range(args['iterations'] - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args['batch_size'] * dist.get_world_size(),
        )
        if bool(args['anneal_lr']):
            set_annealed_lr(opt, args['lr'], (step + resume_step) / args['iterations'])
            
        forward_backward_log(data)
        mp_trainer.optimize(opt)
        
        if val_data is not None and not step % args['eval_interval']:
            with torch.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(val_data, prefix="val")
                    model.train()
        if not step % args['log_interval']:
            logger.dumpkvs()  # Same logging behavior as original code

        if (
            step
            and dist.get_rank() == 0
            and step in save_steps
        ):
            logger.log("saving model...")
            save_model(
                mp_trainer, 
                opt if save_optimizer else None,  # Save optimizer only if specified
                step + resume_step, 
                save_dir, 
                model_prefix=classifier_prefix
            )
    logger.dumpkvs()
            
    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(
            mp_trainer, 
            opt if save_optimizer else None,  # Save optimizer only if specified
            step + resume_step, 
            save_dir, 
            model_prefix=classifier_prefix)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="Parser for classifier training")

    parser.add_argument('--data_path', type=str)
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory where the trained classifiers will be saved')
    parser.add_argument('--heterogeneity_classifier', type=int, choices=[0, 1], default=1, help="Enable (1) or disable (0) heterogeneity classifier training. If enabled, a heterogeneity classifier will be trained for each cell type to distinguish their source of variations, such as origins, sample IDs, or cell states.") 
    
    parser.add_argument('--colname_of_cellType', type=str, default='cell_type',help = 'colname indicating the cell types of single cells from the adata object')
    parser.add_argument('--colname_of_cellType_heterogeneity', type=str, default='sample',help = 'Column name specifying the sources of variation (e.g., sample ID, cell states) within each cell type for single cells in the adata object. When the heterogeneity classifier is enabled, a separate classifier will be trained for each cell type to differentiate between these identifiers')

    parser.add_argument('--iterations', type=int, default=10000, help="Number of iterations for training")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for data loader")
    parser.add_argument('--classifier_use_fp16', type=int, choices=[0, 1], default=0, help="Enable (1) or disable (0) fp16 precision training")
    parser.add_argument('--noised', type=int, choices=[0, 1], default=1, help="Enable (1) or disable (0) noisy classifier")   
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument('--anneal_lr', type=int, choices=[0, 1], default=0, help="Enable (1) or disable (0) learning rate annealing")
    parser.add_argument('--microbatch', type=int, default=-1, help="Microbatch size")
    parser.add_argument('--schedule_sampler', type=str, default='uniform', help="Schedule sampler type")
    parser.add_argument('--log_interval', type=int, default=100, help="Logging interval")
    parser.add_argument('--eval_interval', type=int, default=100, help="Evaluation interval")
    parser.add_argument('--start_guide_time', type=int, default=500, help="Start guide time for training")
    parser.add_argument('--save_percentages', type=int, nargs='+', help="Percentages of training steps to save checkpoints. If specified, intermediate checkpoints of classifiers can be randomly selected to guide future simulations")
    parser.add_argument('--save_optimizer', type=int, choices=[0, 1], default=0, help="Save (1) or not save (0) optimizer state")

    args = parser.parse_args()

    if not os.path.isdir(args.simu_obj_dir):
        raise ValueError(f"The provided simu_obj_dir '{args.simu_obj_dir}' is not a valid directory.")

    if not os.path.exists(os.path.join(args.simu_obj_dir, "scvi_model.pt")):
        raise ValueError('A trained scVI model required for classifier training was not found. Please train the scVI model first before proceeding')

    if not args.save_percentages:
        args.save_percentages = []
        
    classifier_requirements = classifier_requirements_defaults()

    args = vars(args)

    print('start classifier training for cell type classification')
    adata, num_classes, classes = prepare_adata_for_augmentation(args['data_path'],
                                                                 args['colname_of_cellType'],
                                                                 'sample', # this parameter is not required here and will be assigned a non-meaningful value     
                                                                 None,
                                                                 classifier_requirements['min_nCells_per_class'])

    data = load_data_embedding(adata,classes,
                              args['simu_obj_dir'],
                              'scvi_',
                              args['batch_size'])
    
    if num_classes < classifier_requirements['min_nClasses'] or adata.shape[0] < classifier_requirements['min_nCells']:
        raise ValueError(
            f"Training conditions not satisfied for cellType classifier: "
            f"Number of classes ({num_classes}) must be greater than {classifier_requirements['min_nClasses']}, and "
            f"number of cells ({adata.shape[0]}) must be greater than {classifier_requirements['min_nCells']}."
        )

    train_classifier(data, num_classes, args['simu_obj_dir'],'celltype_classifier_',**args)
    
    if bool(args['heterogeneity_classifier']):
        ori_adata = ad.read_h5ad(args['data_path'], backed='r')
        cell_types = np.unique(ori_adata.obs[args['colname_of_cellType']])
        heterogeneity_dir = os.path.join(args['simu_obj_dir'], "heterogeneity_classifiers")

        if not os.path.isdir(heterogeneity_dir):
            os.makedirs(heterogeneity_dir, exist_ok=True)

        for ct in cell_types:
            # load adata for selected cell type only
            adata, num_classes, classes = prepare_adata_for_augmentation(args['data_path'],
                                                                         args['colname_of_cellType'],
                                                                         args['colname_of_cellType_heterogeneity'],
                                                                         ct,
                                                                         classifier_requirements['min_nCells_per_class'])

            # no heter_classifier will be trained if the classifier conditions not satisfied
            if num_classes < classifier_requirements_defaults()['min_nClasses'] or adata.shape[0] < classifier_requirements['min_nCells']:
                continue

            print(f'Start heterogeneity classifier training for {ct}')
            data = load_data_embedding(adata,classes,
                                      args['simu_obj_dir'],
                                      'scvi_',
                                      args['batch_size'])
            
            train_classifier(data, num_classes,heterogeneity_dir,f"{ct}_heterogeneity_classifier_",**args)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()