def umap_defaults():
    """
    Defaults for umap plots
    """
    return dict(
        target_sum = 1e4,
        n_top_genes = 3000,
        n_pcs = 30,
        n_neighbors = 20,
        umap_min_dist = 0.3,
        color_palette = 'tab20',
        dpi = 200,
        umap_width = 8,
        umap_height = 6
    )

def subplots_adjust_defaults():
    return dict(left = 0.05,
               right = 0.95,
               wspace = 0.5,
               hspace = 0.4)

def classifier_requirements_defaults():
    """
    Defines the default conditions required for training a classifier.

    - For a cell type classifier, an error will be raised if these conditions are not met.
    - For a cell type heterogeneity classifier, no classifier will be created for cell types that fail to meet these conditions.
    """
    return dict(
        min_nClasses = 3,
        min_nCells_per_class = 10,
        min_nCells = 500
    )

def SCANVI_training_defaults():
    """
    Defaults for SCANVI training
    """
    return dict(
        n_hidden=128,
        n_latent=10,
        n_layers=1,
        dropout_rate=0.1,
        dispersion='gene',
        gene_likelihood='zinb',
        linear_classifier=False,
    )

def sampling_defaults():
    """
    Default settings for the sampling process
    """
    return dict(
        classifier_scale=2.0,
        class_cond=0,  # 0: Disabled, 1: Enabled
        use_ddim=0,    # 0: Disabled, 1: Enabled
        clip_denoised=1,  # 0: Disabled, 1: Enabled
        start_guide_steps=500,  # Time to use classifier guidance (for multi-condition sampling)
    )


target_sum = 1e6 # pseudobulk defaults
nCells_for_libsize_calc = 200 # sample size used to calculate the latent library size for the original data
pseudobulk_nCells_dist = 'beta' # choices are 'beta' or 'uniform'
nGenes_for_DE_analysis = 5000

# default setting copied from scDiffusion
# https://github.com/EperLuo/scDiffusion/blob/main/guided_diffusion/script_util.py 
def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        class_cond=False,
    )
    
def model_and_diffusion_defaults():
    res = dict(
        input_dim = 128,
        hidden_dim = [512,512,256,128],
        dropout = 0.0
    )
    res.update(diffusion_defaults())
    return res

def classifier_and_diffusion_defaults():
    res = dict(
        input_dim = 128,
        hidden_dim = [512,512,256,128],
        classifier_use_fp16=False,
        dropout = 0.1,
    )
    res.update(diffusion_defaults())
    return res



    