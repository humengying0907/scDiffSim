import os
import sys
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

import anndata as ad
from simple_scvi._mymodel import MyModel
from data_loader import *
import matplotlib.pyplot as plt

def vae_training(adata,
                 n_hidden = 512,
                 n_latent = 128, # make sure this is equal to input_dim for diffusion model!
                 n_layers = 2,
                 dropout_rate = 0.1,
                 max_epochs = 260,
                 save_dir = './',
                 vae_prefix = '' 
                ):
    '''
    train a VAE model for the selected cell type:
    the learnt latent z will be pass to diffusion and classifier
    the learnt decoder structure will be used for data augmentation (for this cell type specifically)
    '''
    
    convergence_plots_dir = os.path.join(save_dir, "convergence_plots")
    os.makedirs(convergence_plots_dir, exist_ok=True)
    
    # use all genes for vae training (genes being used should be the same across cell types)
    MyModel.setup_anndata(adata, batch_key=None)
    vae = MyModel(adata, n_hidden=n_hidden, n_latent=n_latent, 
                  n_layers=n_layers, dropout_rate=dropout_rate)
    
    vae.train(max_epochs = max_epochs)     
    vae.save(dir_path = save_dir, prefix = vae_prefix, overwrite=True)
    print(f'SCVI model saved to {save_dir}/{vae_prefix}model.pt')

    # Plot and save the convergence plot
    plt.figure(figsize=(8, 6))  # Set the figure size (width x height in inches)
    plt.plot(vae.history["elbo_train"], label="Train", linewidth=2)
    plt.title('scVI model elbo train', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True)  
    convergence_plot_path = os.path.join(convergence_plots_dir, f'{vae_prefix}training_elbo.pdf')
    plt.savefig(convergence_plot_path, format='pdf', bbox_inches='tight')  # Use bbox_inches to avoid clipping
    plt.close()
    print(f'Convergence plot saved to {convergence_plot_path}')

def main():
    parser = argparse.ArgumentParser(description="Parser for scVI training")

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--n_latent', type=int, default=128, help="number of latent dimensions in vae, please ensure that this value is consistent between scVI training and diffusion training")
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory where the trained scVI model will be saved')

    # parameters with default values
    parser.add_argument('--colname_of_cellType', type=str, default='cell_type',help = 'colname indicating the cell types of single cells from the adata object')
    parser.add_argument('--n_hidden', type=int, default=512, help = 'number of nodes per hidden layer')
    parser.add_argument('--n_layers', type=int, default=2, help = 'number of hidden layers used for encoder and decoder NNs')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--max_epochs', type=int, default=260)

    args = parser.parse_args()
    adata, num_classes, _ = prepare_adata_for_augmentation(args.data_path,
                                                           args.colname_of_cellType,
                                                           'sample', # this parameter is not required here and will be assigned a non-meaningful value
                                                           None)
    
    if not os.path.isdir(args.simu_obj_dir):
        os.makedirs(args.simu_obj_dir, exist_ok=True) 

    vae_training(adata,
                 args.n_hidden,
                 args.n_latent,
                 args.n_layers,
                 args.dropout_rate,
                 args.max_epochs,
                 args.simu_obj_dir,
                 'scvi_')

if __name__ == "__main__":
    main()


