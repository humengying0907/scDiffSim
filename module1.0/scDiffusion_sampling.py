import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier
)

def multi_condi_sampling(ct,
                         diffusion_path,
                         celltype_classifier_path,
                         heterogeneity_classifier_path,
                         celltype_classifier_map,
                         heterogeneity_classifier_map,
                         num_cells,
                         n_latent,
                         **sampling_args):        

    celltype_classifier_num_classes = len(celltype_classifier_map)
    heterogeneity_classifier_num_classes = len(heterogeneity_classifier_map)
    ct_id = celltype_classifier_map[ct]
    
    dist_util.setup_dist()
    
    model_and_diffusion_defaults_settings = model_and_diffusion_defaults()
    model_and_diffusion_defaults_settings['input_dim'] = n_latent
    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults_settings
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(diffusion_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    classifier1 = create_classifier(model_and_diffusion_defaults_settings['input_dim'],
                              model_and_diffusion_defaults_settings['hidden_dim'],
                              celltype_classifier_num_classes) 
    classifier1.load_state_dict(
    dist_util.load_state_dict(celltype_classifier_path, map_location="cpu")
    )
    classifier1.to(dist_util.dev())
    classifier1.eval()
    
    classifier2 = create_classifier(model_and_diffusion_defaults_settings['input_dim'],
                              model_and_diffusion_defaults_settings['hidden_dim'],
                              heterogeneity_classifier_num_classes) 
    classifier2.load_state_dict(
    dist_util.load_state_dict(heterogeneity_classifier_path, map_location="cpu")
    )
    classifier2.to(dist_util.dev())
    classifier2.eval()
    
    def cond_fn_multi(x, t, y=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits1 = classifier1(x_in, t)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            selected1 = log_probs1[range(len(logits1)), y1.view(-1)]
    
            logits2 = classifier2(x_in, t)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected2 = log_probs2[range(len(logits2)), y2.view(-1)]
            
            grad1 = torch.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * 2
            grad2 = torch.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * 2
            
            return grad1+grad2
                
    def model_fn(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        return model(x, t) 
        
    logger.log("sampling...")
    all_cell = []

    for j in range(heterogeneity_classifier_num_classes):        
        model_kwargs = {}
        classes1 = (ct_id)*torch.ones((num_cells,), device=dist_util.dev(), dtype=torch.long)
        classes2 = (j)*torch.ones((num_cells,), device=dist_util.dev(), dtype=torch.long)
        classes = torch.stack((classes1,classes2), dim=1)
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not bool(sampling_args['use_ddim']) else diffusion.ddim_sample_loop
        )
        sample, traj = sample_fn(
            model_fn,
            (num_cells, model_and_diffusion_defaults()['input_dim']),
            clip_denoised=sampling_args['clip_denoised'],
            model_kwargs=model_kwargs,
            cond_fn=cond_fn_multi,
            device=dist_util.dev(),
            noise = None,
            start_time=diffusion.betas.shape[0],
            start_guide_steps=sampling_args['start_guide_steps'],
        )
    
        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample) 
        
        all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])
        dist.barrier()
        
    arr = np.concatenate(all_cell, axis=0)
    return arr
        
def condi_sampling(ct,
                   diffusion_path, 
                   classifier_path,
                   celltype_classifier_map,
                   num_cells,
                   n_latent,
                   **sampling_args):
    
    ct_id = celltype_classifier_map[ct]
    dist_util.setup_dist()

    model_and_diffusion_defaults_settings = model_and_diffusion_defaults()
    model_and_diffusion_defaults_settings['input_dim'] = n_latent
    model, diffusion = create_model_and_diffusion(
        **model_and_diffusion_defaults_settings
    )
    
    model.load_state_dict(
        dist_util.load_state_dict(diffusion_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    classifier = create_classifier(model_and_diffusion_defaults_settings['input_dim'],
                                  model_and_diffusion_defaults_settings['hidden_dim'],
                                  len(celltype_classifier_map))
    
    classifier.load_state_dict(
        dist_util.load_state_dict(classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    classifier.eval()
    
    def cond_fn_ori(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * sampling_args['classifier_scale']
            return grad
        
    def model_fn(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        
        if bool(sampling_args['class_cond']):
            return model(x, t, y)
        else:
            return model(x, t)
        
    # logger.log("sampling...")
    all_cell = []
    model_kwargs = {}

    classes = (ct_id)*torch.ones((num_cells,), device=dist_util.dev(), dtype=torch.long)
    model_kwargs["y"] = classes
    sample_fn = (
        diffusion.p_sample_loop if not bool(sampling_args['use_ddim']) else diffusion.ddim_sample_loop
    )
    sample, traj = sample_fn(
        model_fn,
        (num_cells, 
         model_and_diffusion_defaults()['input_dim']),
        clip_denoised=bool(sampling_args['clip_denoised']),
        model_kwargs=model_kwargs,
        cond_fn=cond_fn_ori,
        device=dist_util.dev(),
        noise = None,
    )
    gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, sample) 
    
    all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])            
    arr = np.concatenate(all_cell, axis=0)
    dist.barrier()
    # logger.log(f"sampling complete for {ct}")

    return arr