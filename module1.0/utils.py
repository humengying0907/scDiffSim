import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import glob
from natsort import natsorted
import random
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy.stats import norm, uniform, beta, ttest_ind
from scipy.sparse import csr_matrix, vstack
from scvi.distributions import ZeroInflatedNegativeBinomial

import matplotlib.pyplot as plt

from configs import nGenes_for_DE_analysis


def build_sampling_config(cell_type, 
                          max_batch,
                          avail_diffusion_paths,
                          avail_celltype_classifier_paths,
                          avail_heter_classifiers = None, 
                          nCells_per_ct = 500
                         ):
    
    if isinstance(cell_type, list) and len(cell_type) > 1:
        
        # use the same sampling config for all cell_types
        rows = []
        rows.append({"batch": 0, 
                     "diffusion_model": avail_diffusion_paths[-1], 
                     "classifier_model": avail_celltype_classifier_paths[-1]})
        if max_batch > 1:
            for i in range(1, max_batch):      
                new_row = {
                    "batch": i,
                    "diffusion_model": random.choice(avail_diffusion_paths),
                    "classifier_model": random.choice(avail_celltype_classifier_paths)}
                rows.append(new_row) 
            
        configure_df = pd.DataFrame(rows) 

    elif (isinstance(cell_type, list) and len(cell_type) == 1) or isinstance(cell_type, str):        
        # specify sampling config for this cell_type only
        rows = []
        rows.append({"batch": 0, 
                     "cell_type": cell_type,
                     "diffusion_model": avail_diffusion_paths[-1], 
                     "celltype_classifier": avail_celltype_classifier_paths[-1], 
                     "heter_classifier": avail_heter_classifiers[-1] if avail_heter_classifiers else None, 
                     "nCells_per_ct": nCells_per_ct,
                     "heterogeneity": bool(avail_heter_classifiers)
                    })    

        if max_batch > 1:
            for i in range(1, max_batch):      
                new_row = {
                    "batch": i,
                    "cell_type": cell_type,
                    "diffusion_model": random.choice(avail_diffusion_paths),
                    "celltype_classifier": random.choice(avail_celltype_classifier_paths), 
                    "heter_classifier": random.choice(avail_heter_classifiers) if avail_heter_classifiers else None, 
                    "nCells_per_ct": nCells_per_ct,
                    "heterogeneity": bool(avail_heter_classifiers)
                }
                rows.append(new_row) 
        
        configure_df = pd.DataFrame(rows) 
        configure_df['heter_classifier'] = configure_df['heter_classifier'].fillna('<NA>')

    else:
        raise ValueError("Invalid cell_type input")

    return configure_df

def sample_beta(low=50, high=100):
    # Draw from a Beta(1,2) distribution on [0,1]
    # Beta(1,2) is high near 0 and low near 1.
    X = beta.rvs(1, 2, size=1)
    
    # Linearly transform X from [0,1] to [low, high]
    Y = low + (high - low) * X
    return Y

def aggregate_by_group(Expr, 
                       identifier_labels,  # A list or 1D NumPy array containing labels used for grouping. 
                       gene_names=None, 
                       min_nCells_per_pseudobulk=10,
                       pseudobulk_subsampling=False, 
                       pseudobulk_nCells_low = 10,
                       pseudobulk_nCells_high = 500,
                       pseudobulk_nCells_dist='uniform'):
    
    # Validate input dimensions
    if Expr.shape[1] != len(identifier_labels):
        raise ValueError("Number of columns in Expr must equal the length of identifier_labels")
        
    if isinstance(identifier_labels, list):
        identifier_labels = np.array(identifier_labels)  
    elif isinstance(identifier_labels, np.ndarray) and identifier_labels.ndim != 1:
        raise ValueError("identifier_labels must be a 1D array if it's a NumPy array.")

    # Get unique labels and filter by min_nCells_per_pseudobulk
    labels, counts = np.unique(identifier_labels, return_counts=True)
    valid_labels = [label for label, count in zip(labels, counts) if count >= min_nCells_per_pseudobulk]

    if len(valid_labels) == 0:
        print(f"No identifiers with more than {min_nCells_per_pseudobulk} cells to aggregate. Returning an empty DataFrame.")
        return pd.DataFrame(), []

    # Group indices by valid labels
    groups = {}
    nCells_per_pseudobulk = []
    
    if pseudobulk_subsampling:
        for label in valid_labels:
            indices = np.where(identifier_labels == label)[0]

            if pseudobulk_nCells_dist == 'uniform':
                sample_size = random.randint(max(min_nCells_per_pseudobulk,pseudobulk_nCells_low),
                                             max(min_nCells_per_pseudobulk,pseudobulk_nCells_high))
                
            elif pseudobulk_nCells_dist == 'beta':
                n = sample_beta(max(min_nCells_per_pseudobulk, pseudobulk_nCells_low),
                                max(min_nCells_per_pseudobulk, pseudobulk_nCells_high))[0]  
                sample_size = int(np.floor(n))        
            else:
                raise ValueError(f"Invalid nCells_dist value: {nCells_dist}. "
                 f"Allowed values are 'uniform' or 'beta'.")

            sample_size = min(len(indices),sample_size)

            if sample_size < min_nCells_per_pseudobulk:
                continue
                
            groups[label] = np.random.choice(indices, size=sample_size, replace=False)
            nCells_per_pseudobulk.append(sample_size)

        if len(groups) == 0:
            print(f"No identifiers with more than {min_nCells_per_pseudobulk} cells to aggregate under current subsampling settings. Returning an empty DataFrame. Please consider adjusting the subsampling configs")
            return pd.DataFrame(), []

    else:
        groups = {label: np.where(identifier_labels == label)[0] for label in valid_labels}
        nCells_per_pseudobulk.extend([len(groups[label]) for label in valid_labels])

    # Aggregate expression values for each group
    C = np.column_stack([np.mean(Expr[:, indices], axis=1) for indices in groups.values()])

    # Create DataFrame for aggregated results
    df_C = pd.DataFrame(C, columns=valid_labels)
    if gene_names is not None:
        df_C.index = gene_names

    return df_C, nCells_per_pseudobulk

def latent2expr(simulated_latent_array,vae,library_size=None,fixed_libsize = 8):
    n_samples = simulated_latent_array.shape[0]

    if isinstance(simulated_latent_array, torch.Tensor) and simulated_latent_array.is_cuda:
        z = simulated_latent_array  
    else:
        z = torch.tensor(simulated_latent_array).cuda()
        
    if library_size is None: 
        library_size = torch.ones(n_samples, 1, dtype=torch.float32) * fixed_libsize

    decoding_batch_size = 128  # Adjust to fit GPU memory, defalut to 128

    cell_gen_expr_csr = []  

    # Loop over batches
    for i in range(0, n_samples, decoding_batch_size):
        batch_end = min(i + decoding_batch_size, n_samples)
        z_batch = z[i:batch_end, :]
        library_size_batch = library_size[i:batch_end]  
    
        with torch.no_grad():
            generative_outputs = vae.module.generative(z_batch, library_size_batch)
            px_r_batch = generative_outputs["px_r"]  
            px_rate_batch = generative_outputs["px_rate"] 
            px_dropout_batch = generative_outputs["px_dropout"] 
    
            dist_batch = ZeroInflatedNegativeBinomial(
                mu=px_rate_batch,
                theta=px_r_batch,
                zi_logits=px_dropout_batch
            )
    
            sampled_batch = dist_batch.sample().cpu().numpy()  # Convert to numpy for sparse storage
            sampled_csr = csr_matrix(sampled_batch)  # Convert to sparse format
            cell_gen_expr_csr.append(sampled_csr)  # Append the sparse matrix
    
    simulated_expr = vstack(cell_gen_expr_csr)
    return simulated_expr


def SCANVI_query_prob(query_adata,
                      scanvi_model,
                      colname_of_labels = 'cell_type', # column name indicating label keys for scanvi_model
                      SCANVI_max_epochs = 50,
                      save_dir = './',
                      SCANVI_prefix = 'celltypeAnno_SCANVI_'):
    
    query_adata.obs[colname_of_labels] = 'Unknown' 
    scvi.model.SCANVI.prepare_query_anndata(query_adata, reference_model = scanvi_model)
    scanvi_query = scvi.model.SCANVI.load_query_data(query_adata, scanvi_model)
    scanvi_query.train(max_epochs=SCANVI_max_epochs)

    convergence_plots_dir = os.path.join(save_dir, "convergence_plots")
    os.makedirs(convergence_plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))  # Set the figure size (width x height in inches)
    plt.plot(scanvi_query.history["elbo_train"], label="query", linewidth=2)
    plt.title('SCANVI elbo query', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True)  
    convergence_plot_path = os.path.join(convergence_plots_dir, f'{SCANVI_prefix}_query_elbo.pdf')
    plt.savefig(convergence_plot_path, format='pdf', bbox_inches='tight')  
    plt.close()
    print(f'Convergence plot saved to {convergence_plot_path}')
    
    probs = scanvi_query.predict(soft = True)
    return probs

def pseudobulk_nCells_by_ct(obs,colname_of_cellType = 'cell_type',colname_of_cellType_heterogeneity = 'sample'):
    
    results = []
    for cell_type, filtered_obs in obs.groupby(colname_of_cellType):
        group_sizes = filtered_obs.groupby(colname_of_cellType_heterogeneity).size()
        min_ncells = group_sizes.min()
        max_ncells = group_sizes.max()
        results.append({'cell_type': cell_type, 'min_nCells': min_ncells, 'max_nCells': max_ncells})

    result_df = pd.DataFrame(results)

    return result_df


def get_top_variable_genes(expression_data, top_n=1000):
    """
    Identifies the top variable genes from a gene expression dataset.

    Parameters:
    - expression_data (pd.DataFrame): Gene expression data (genes x samples).
    - top_n (int): Number of top variable genes to select.

    Returns:
    - pd.Index: Indices (gene names) of the top variable genes.
    """
    log_transformed = np.log2(expression_data + 1)
    var = log_transformed.var(axis=1)
    sorted_var = var.sort_values(ascending=False)
    top_var_genes = sorted_var.index[:top_n]

    return top_var_genes


def load_pseudobulk_data(pseudobulk_dir):
    """
    Load pseudobulk data from files in a specified folder and prefix 'ct_' to orig column names.

    Parameters:
    - pseudobulk_dir (str): Path to the folder containing pseudobulk files.

    Returns:
    - combined_expr (pd.DataFrame): Combined gene expression DataFrame.
    - metadata (pd.DataFrame): Metadata indicating cell type for each sample.
    """
    data_frames = []
    cell_types = []

    # Get all pseudobulk files in the folder
    for file in os.listdir(pseudobulk_dir):
        if file.endswith("_pseudobulk.csv"):
            cell_type = file.split('_pseudobulk.csv')[0]  # Extract cell type from filename
            file_path = os.path.join(pseudobulk_dir, file)
            df = pd.read_csv(file_path, index_col=0)

            # Prefix 'ct_' to 'orig' column names to ensure uniqueness
            df.columns = [f"{cell_type}_{col}" if col.startswith("orig") else col for col in df.columns]
            
            data_frames.append(df)
            cell_types.extend([cell_type] * df.shape[1])  # Repeat cell type for each column (sample)

    # Combine data frames
    combined_expr = pd.concat(data_frames, axis=1)

    # Create metadata with updated column names
    metadata = pd.DataFrame({"cell_type": cell_types}, index=combined_expr.columns)
    
    return combined_expr, metadata

def pseudobulk_de_analysis(expr_data, metadata, top_n=500, var_genes_n=5000):
    """
    Perform DE analysis on combined pseudobulk profiles for different cell types to identify marker genes.

    Parameters:
    - expr_data (pd.DataFrame): Combined gene expression DataFrame (genes x samples).
    - metadata (pd.DataFrame): Metadata indicating cell type for each sample.
    - top_n (int): Number of top DE genes to identify.

    Returns:
    - marker_genes (dict): Dictionary of top DE genes for each cell type.
    """
    marker_genes = {}
    cell_types = metadata["cell_type"].unique()

    # selecting only top var_genes_n genes for DE analysis
    top_var_genes = get_top_variable_genes(expr_data, top_n=var_genes_n)
    expr_data = expr_data.loc[top_var_genes]

    if expr_data.max().max() > 50:
        expr_data = np.log2(expr_data + 1)
        
    for ct in cell_types:

        print(f"Performing DE analysis for cell type: {ct}...")

        group1 = expr_data.loc[:, metadata["cell_type"] == ct]
        group2 = expr_data.loc[:, metadata["cell_type"] != ct]

        p_values = []
        for gene in expr_data.index:
            t_stat, p_val = ttest_ind(group1.loc[gene], group2.loc[gene], equal_var=False)
            p_values.append((gene, p_val))
        
        sorted_genes = sorted(p_values, key=lambda x: x[1])[:top_n]
        marker_genes[ct] = [gene for gene, _ in sorted_genes]

    return marker_genes


def prep_pca_genes(pca_genes_choice, pseudobulk_dir=None, top_n = 1000):

    if pca_genes_choice == 'marker_genes':
        # Load pseudobulk data and perform differential expression analysis
        combined_expr, metadata = load_pseudobulk_data(pseudobulk_dir)
        marker_genes = pseudobulk_de_analysis(combined_expr, metadata, top_n, nGenes_for_DE_analysis)

    elif pca_genes_choice == 'top_var_genes':
        marker_genes = {}

        # Check if the pseudobulk directory exists and contains the required files
        if not pseudobulk_dir or not os.path.exists(pseudobulk_dir):
            raise ValueError("The specified pseudobulk directory does not exist.")

        pseudobulk_files = [file for file in os.listdir(pseudobulk_dir) if file.endswith("_pseudobulk.csv")]
        if len(pseudobulk_files) == 0:
            raise ValueError("No pseudobulk files available for visualization.")
        
        for file in pseudobulk_files:
            ct = file.split('_pseudobulk.csv')[0] 
            file_path = os.path.join(pseudobulk_dir, file)
            
            df = pd.read_csv(file_path, index_col=0) 
            marker_genes[ct] = get_top_variable_genes(df, top_n)  
    else:
        # Handle invalid choice
        raise ValueError("Invalid pca_genes_choice. Must be 'marker_genes' or 'top_var_genes'.")

    return marker_genes













