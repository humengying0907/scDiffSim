import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import random
import itertools
import glob
import math
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from natsort import natsorted
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from configs import umap_defaults, subplots_adjust_defaults
from utils import get_top_variable_genes

def plot_umap_all_ct(concatenate_adata,
                     prob_cut = 0.8,
                     save_dir = './',
                     export = True):
    
    umap_params = umap_defaults()
    adjust_params = subplots_adjust_defaults()

    if export:
        export_dir = os.path.join(save_dir, "results_visualization")
        os.makedirs(export_dir, exist_ok=True)

    simulated_indices = concatenate_adata.obs[concatenate_adata.obs['source'] == 'simulated'].index
    ori_indices = concatenate_adata.obs[concatenate_adata.obs['source'] == 'original'].index

    # Select only a subset of simulated cells in UMAP to avoid overcrowding
    n_cells = min(len(ori_indices), len(simulated_indices))

    sampled_simulated_indices = np.random.choice(simulated_indices, size=n_cells, replace=False)
    final_indices = np.concatenate([ori_indices, sampled_simulated_indices])

    subset = concatenate_adata[final_indices].copy()        
    sc.pp.normalize_total(subset, target_sum = umap_params['target_sum'])
    sc.pp.log1p(subset)
    sc.pp.highly_variable_genes(subset, n_top_genes = umap_params['n_top_genes'], subset = True)
    sc.tl.pca(subset) 
    sc.pp.neighbors(subset, n_pcs=umap_params['n_pcs'], n_neighbors=umap_params['n_neighbors'])
    
    umap_pca = sc.tl.umap(subset, min_dist=umap_params['umap_min_dist'],copy = True)

    # assign the same color palette
    adata_orig = umap_pca[umap_pca.obs['source'] == 'original', :].copy()
    adata_simu = umap_pca[umap_pca.obs['source'] == 'simulated', :].copy()
    
    adata_orig.obs['umap_color'] = adata_orig.obs['cell_type'].astype(str)
    adata_simu.obs['umap_color'] = adata_simu.obs['cell_type'].astype(str)
    
    categories = adata_orig.obs['umap_color'].unique()
    palette = sns.color_palette(umap_params['color_palette'], len(categories))
    color_mapping = dict(zip(categories, palette))

    if min(adata_simu.obs['prob']) < 1:
        adata_simu_filtered = adata_simu[adata_simu.obs['prob'] >= prob_cut].copy()
        fig, axs = plt.subplots(1, 3, figsize=(umap_params['umap_width'] * 3, 
                                               umap_params['umap_height'])) 
        plt.subplots_adjust(**adjust_params)
        
        sc.pl.umap(adata_orig, color='umap_color', ax=axs[0],show = False, palette = color_mapping,title = 'original cells')
        sc.pl.umap(adata_simu, color='umap_color', ax=axs[1],show = False,palette = color_mapping,title = 'simulated cells (raw)')
        sc.pl.umap(adata_simu_filtered, color='umap_color', ax=axs[2],show = False,
                   palette = color_mapping,title = "simulated cells (filtered)")
        
    else:
        fig, axs = plt.subplots(1, 2, figsize=(umap_params['umap_width'] * 2, 
                                               umap_params['umap_height'])) 
        plt.subplots_adjust(**adjust_params)
        
        sc.pl.umap(adata_orig, color='umap_color', ax=axs[0],show = False, palette = color_mapping,title = 'original')
        sc.pl.umap(adata_simu, color='umap_color', ax=axs[1],show = False,palette = color_mapping,title = 'simulated')
        
    if export:
        plt.savefig(os.path.join(export_dir,'all_umap.pdf'), dpi=umap_params['dpi'], bbox_inches='tight')
    else:
        plt.show()


def plot_umap_single_ct(ct,
                        concatenate_adata,
                        prob_cut = 0.8,
                        save_dir = './',
                        export = True):
    
    print(f"Plotting UMAP plot for {ct} ...")

    umap_params = umap_defaults()
    adjust_params = subplots_adjust_defaults()

    if export:
        export_dir = os.path.join(save_dir, "results_visualization")
        os.makedirs(export_dir, exist_ok=True)
    
    if min(concatenate_adata.obs['prob']) < 1:
        print(f"Filtering has already been applied for {ct}; only plotting cells with prob >= {prob_cut}")
        concatenate_adata = concatenate_adata[concatenate_adata.obs['prob'] >= prob_cut].copy()

        if (concatenate_adata.obs['source'] == 'simulated').sum() == 0:
            raise ValueError(
                f"Filter applied but none of the simulated cells passed the prob cut threshold for {ct}. No UMAP plot will be generated for this cell type."
            )

        title1b =  f"original + simulated {ct} (filtered)"
    else:
        title1b =  f"original + simulated {ct}"

    simulated_indices = concatenate_adata.obs[
        (concatenate_adata.obs['source'] == 'simulated') & (concatenate_adata.obs['cell_type'] == ct)
    ].index
    
    ori_indices = concatenate_adata.obs[concatenate_adata.obs['source'] == 'original'].index
    
    # Select only a subset of simulated cells with a size comparable to the original 
    n_cells = min(
        sum((concatenate_adata.obs['source'] == 'original') & (concatenate_adata.obs['cell_type'] == ct)),
        len(simulated_indices)
    )
    
    sampled_simulated_indices = np.random.choice(simulated_indices, size=n_cells, replace=False)
    final_indices = np.concatenate([ori_indices, sampled_simulated_indices])
    
    subset = concatenate_adata[final_indices].copy()        
    sc.pp.normalize_total(subset, target_sum = umap_params['target_sum'])
    sc.pp.log1p(subset)
    sc.pp.highly_variable_genes(subset, n_top_genes = umap_params['n_top_genes'], subset = True)
    sc.tl.pca(subset) 
    sc.pp.neighbors(subset, n_pcs=umap_params['n_pcs'], n_neighbors=umap_params['n_neighbors'])
    
    umap_pca = sc.tl.umap(subset, min_dist=umap_params['umap_min_dist'],copy = True)


    umap_pca.obs['source_color'] = np.nan
    umap_pca.obs.loc[umap_pca.obs['cell_type'] != ct, 'source_color'] = 'other_ct'
    umap_pca.obs.loc[umap_pca.obs['cell_type'] == ct, 'source_color'] = umap_pca.obs['source'].astype(str)

    umap_pca.obs['source_color'] = pd.Categorical(
        umap_pca.obs['source_color'],
        categories=['original', 'simulated', 'other_ct'],
        ordered=True
    )

    # check if heterogeneity classifier is enabled for this ct
    search_pattern = os.path.join(save_dir,'heterogeneity_classifiers', f"{ct}_heterogeneity_classifier_*.pt")
    heter_classifiers = natsorted(glob.glob(search_pattern))

    # cell type color palette (the same as umap for all ct)
    categories = umap_pca.obs['cell_type'].unique()
    palette = sns.color_palette(umap_params['color_palette'], len(categories))
    ct_color_mapping = dict(zip(categories, palette))

    adata_orig = umap_pca[umap_pca.obs['source'] == 'original', :].copy()

    if heter_classifiers:
        fig, axs = plt.subplots(2, 2, figsize=(umap_params['umap_width'] * 2, 
                                               umap_params['umap_height'] * 2)) 
        plt.subplots_adjust(**adjust_params)
        sc.pl.umap(adata_orig, color='cell_type',  ax=axs[0,0],show = False, palette = ct_color_mapping, title = 'original')   
        sc.pl.umap(umap_pca, color='source_color',
                    palette={
                        'other_ct': '#cacacb',
                        'original': '#1f77b4',
                        'simulated': '#ff7f0e'
                    },
                   ax=axs[0,1],show = False, title = title1b)
        
        adata_orig.obs['umap_color'] = np.nan
        adata_orig.obs.loc[adata_orig.obs['cell_type'] != ct, 'umap_color'] = 'other_ct'
        adata_orig.obs['pseudobulkID'] = adata_orig.obs['pseudobulkID'].str.replace('^orig_', '', regex=True)
        adata_orig.obs.loc[adata_orig.obs['cell_type'] == ct, 'umap_color'] = adata_orig.obs['pseudobulkID']
        adata_orig.obs['umap_color'] = adata_orig.obs['umap_color'].astype(str)
        
        categories = adata_orig.obs['umap_color'].unique()
        palette = sns.color_palette(umap_params['color_palette'], len(categories))
        ct_heter_color_mapping = dict(zip(categories, palette))
        
        sc.pl.umap(adata_orig, color='umap_color', ax=axs[1,0],show = False, palette = ct_heter_color_mapping,title = f"{ct} heterogeneity in original adata" )

        adata_right = umap_pca[~((umap_pca.obs['source'] == 'original') & (umap_pca.obs['cell_type'] == ct))].copy()
        
        adata_right.obs['umap_color'] = np.nan
        adata_right.obs.loc[adata_right.obs['cell_type'] != ct, 'umap_color'] = 'other_ct'
        adata_right.obs['pseudobulkID'] = adata_right.obs['pseudobulkID'].str.replace(r'^simulated_|_[^_]*$', '', regex=True)
        
        adata_right.obs.loc[adata_right.obs['cell_type'] == ct, 'umap_color'] = adata_right.obs['pseudobulkID']
        adata_right.obs['umap_color'] = adata_right.obs['umap_color'].astype(str)
        
        sc.pl.umap(adata_right, color='umap_color', ax=axs[1,1],show = False, palette=ct_heter_color_mapping, title=f"{ct} heterogeneity in simulated adata")
        
    else:
        fig, axs = plt.subplots(1, 2, figsize=(umap_params['umap_width'] * 2, 
                                               umap_params['umap_height'])) 
        plt.subplots_adjust(**adjust_params)
        sc.pl.umap(adata_orig, color='cell_type',  ax=axs[0],show = False, palette = ct_color_mapping, title = 'original')
        sc.pl.umap(umap_pca, color='source_color',
                    palette={
                        'other_ct': '#cacacb',
                        'original': '#1f77b4',
                        'simulated': '#ff7f0e'
                    },
                   ax=axs[1],show = False, title = title1b)
    if export:
        plt.savefig(os.path.join(export_dir,f'{ct}_umap.pdf'), dpi=umap_params['dpi'], bbox_inches='tight')
    else:
        plt.show()


def plot_pca_pseudobulk_grid(expr_data_dict, 
                             cluster_dict, 
                             var_genes_dict=None, 
                             top_n=1000, 
                             save_dir='./', 
                             export=True):

    # Calculate the grid dimensions
    num_plots = len(expr_data_dict)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    # Create a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
    axes = axes.flatten()  # Flatten the grid to easily iterate over subplots

    # Iterate through each cell type and plot PCA
    for i, (cell_type, expr_data) in enumerate(expr_data_dict.items()):

        search_pattern = os.path.join(save_dir,'heterogeneity_classifiers', f"{cell_type}_heterogeneity_classifier_*.pt")
        heter_classifiers = natsorted(glob.glob(search_pattern))

        if heter_classifiers:
            title = f"{cell_type} \n (heter_classifier enabled)"
        else:
            title = cell_type
        
        try:
            ax = axes[i]
            cluster = cluster_dict[cell_type]

            if len(cluster) != expr_data.shape[1]:
                raise ValueError(f"The length of 'cluster' for cell type '{cell_type}' must match the number of samples in 'expr_data'.")

            # Filter or select variable genes
            if var_genes_dict and cell_type in var_genes_dict:
                var_genes = var_genes_dict[cell_type]
                expr_data = expr_data.loc[var_genes]
                if expr_data.empty:
                    raise ValueError(f"The provided 'var_genes' list does not match any genes in 'expr_data' for cell type '{cell_type}'.")
            else:
                var_genes = get_top_variable_genes(expr_data, top_n)
                expr_data = expr_data.loc[var_genes]

            if expr_data.max().max() > 50:
                expr_data = np.log2(expr_data + 1)

            # Perform PCA
            pca = PCA(n_components=2)
            pca_coords = pca.fit_transform(expr_data.T)

            # Create a DataFrame for PCA results
            proj = pd.DataFrame(pca_coords, columns=[f"Dim.{i+1}" for i in range(2)])
            proj["cluster"] = cluster

            # Plot on the subplot
            sns.scatterplot(
                ax=ax,
                x="Dim.1",
                y="Dim.2",
                hue="cluster",
                data=proj,
                palette="tab10",
                s=50
            )
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Dim.1")
            ax.set_ylabel("Dim.2")
            ax.legend(title="Cluster", loc="best", fontsize=10)
        except Exception as e:
            print(f"Error processing cell type '{cell_type}': {e}")

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Save the figure if export is True
    if export:
        export_dir = os.path.join(save_dir, "results_visualization")
        os.makedirs(export_dir, exist_ok=True)
        plt.savefig(os.path.join(export_dir, "pseudobulks_pca.pdf"), dpi=300)

    plt.show()

    