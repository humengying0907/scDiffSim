import os
import sys
import warnings
import argparse

warnings.simplefilter(action='ignore', category=FutureWarning)

import scvi
from data_loader import *
from configs import SCANVI_training_defaults
import matplotlib.pyplot as plt

def SCANVI_training(adata,
                    colname_of_labels,
                    save_dir = './',
                    SCANVI_prefix = 'celltypeAnno_SCANVI_',
                    max_epochs = 260):

    SCANVI_params = SCANVI_training_defaults()
    
    scvi.model.SCANVI.setup_anndata(adata, labels_key = colname_of_labels, unlabeled_category = 'Unknown')
    scanvi_model = scvi.model.SCANVI(adata, 
                                     n_hidden=SCANVI_params['n_hidden'], 
                                     n_latent=SCANVI_params['n_latent'], 
                                     n_layers=SCANVI_params['n_layers'], 
                                     dropout_rate=SCANVI_params['dropout_rate'], 
                                     dispersion=SCANVI_params['dispersion'], 
                                     gene_likelihood=SCANVI_params['gene_likelihood'], 
                                     linear_classifier=SCANVI_params['linear_classifier'])
    
    scanvi_model.train(max_epochs=max_epochs)
    scanvi_model.save(dir_path = save_dir, prefix = SCANVI_prefix, overwrite=True)

    convergence_plots_dir = os.path.join(save_dir, "convergence_plots")
    os.makedirs(convergence_plots_dir, exist_ok=True)

    # Plot and save the convergence plot
    plt.figure(figsize=(8, 6))  # Set the figure size (width x height in inches)
    plt.plot(scanvi_model.history["elbo_train"], label="Train", linewidth=2)
    plt.title('SCANVI elbo train', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True)  
    convergence_plot_path = os.path.join(convergence_plots_dir, f'{SCANVI_prefix}_training_elbo.pdf')
    plt.savefig(convergence_plot_path, format='pdf', bbox_inches='tight')  # Use bbox_inches to avoid clipping
    plt.close()
    print(f'Convergence plot saved to {convergence_plot_path}')

def main():
    parser = argparse.ArgumentParser(description="Parser for SCANVI training")

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--colname_of_labels', type=str, default='cell_type',help = 'column name indicating cell annotations from the adata object. A SCANVI model will be trained to classify these annotations and subsequently used to label and filter new simulated single cells')
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory where the trained SCANVI model will be saved')
    parser.add_argument('--max_epochs', type=int, default=260)

    args = parser.parse_args()
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Input `adata` file not found at the specified path: {args.data_path}")
    
    adata = prepare_ori_adata(args.data_path)
    SCANVI_training(adata, args.colname_of_labels, args.simu_obj_dir, 'celltypeAnno_SCANVI_',args.max_epochs)

if __name__ == "__main__":
    main()



    
    