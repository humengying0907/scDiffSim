import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import pandas as pd

from visualization import plot_pca_pseudobulk_grid
from utils import *

def main():
    parser = argparse.ArgumentParser(description="Parser for sampling")
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory containing necessary models for sampling')
    parser.add_argument('--pseudobulk_pca_genes_file', type=str, default = 'pseudobulk_pca_genes.csv',help = '') 
    parser.add_argument('--pca_genes_choice', type=str, 
                        default = 'marker_genes',
                        choices = ['marker_genes','top_var_genes'],help = '')
    parser.add_argument('--top_n', type=int, default=500, help="Number of top DE/variable genes for pca")

    args = parser.parse_args()
    if not os.path.isdir(args.simu_obj_dir):
        raise ValueError(f"The provided simu_obj_dir '{args.simu_obj_dir}' is not a valid directory.")

    pseudobulk_dir = os.path.join(args.simu_obj_dir,'simulation_res')
    if not os.path.isdir(pseudobulk_dir):
        raise ValueError('Unable to find the directory that store pseudobulk results')

    pseudobulk_files = [file for file in os.listdir(pseudobulk_dir) if file.endswith("_pseudobulk.csv")]
    if len(pseudobulk_files) == 0:
        raise ValueError("No pseudobulk files avaialble for visualization.")

    cell_types = [file.split('_pseudobulk.csv')[0] for file in pseudobulk_files]

    os.makedirs(os.path.join(args.simu_obj_dir, 'pseudobulk_configs'), exist_ok=True)
    pca_gene_file = os.path.join(args.simu_obj_dir, 'pseudobulk_configs', args.pseudobulk_pca_genes_file)
    
    if os.path.exists(pca_gene_file):
        pca_genes_df = pd.read_csv(pca_gene_file)        
        required_columns = {'cell_type', 'pca_genes'}
        
        if (not required_columns.issubset(pca_genes_df.columns) 
            or not set(cell_types).issubset(pca_genes_df['cell_type'].unique())):
            
            print("The PCA gene file format is incorrect or missing cell types. It will be overwritten.")
            pca_genes = prep_pca_genes(args.pca_genes_choice, 
                                       pseudobulk_dir, 
                                       args.top_n)
            
            rows = [
                {"cell_type": ct, "pca_genes": ",".join(genes)}
                for ct, genes in pca_genes.items()
            ]
            pca_genes_df = pd.DataFrame(rows)
            pca_genes_df.to_csv(pca_gene_file, index=False)
 

        else:
            print('Using provided pseudobulk_pca_genes_file to guide pca plotting')
            pca_genes = {
                row["cell_type"]: row["pca_genes"].split(",")
                for _, row in pca_genes_df.iterrows()
            }
    else:
        pca_genes = prep_pca_genes(args.pca_genes_choice, 
                                   pseudobulk_dir, 
                                   args.top_n)
        rows = [
            {"cell_type": ct, "pca_genes": ",".join(genes)}
            for ct, genes in pca_genes.items()
        ]
        pca_genes_df = pd.DataFrame(rows)
        pca_genes_df.to_csv(pca_gene_file, index=False)


     # Prepare data for grid plotting
    expr_data_dict = {}
    cluster_dict = {}

    for ct in cell_types:
        try:
            pseudobulk_file = os.path.join(pseudobulk_dir, f'{ct}_pseudobulk.csv')
            if not os.path.exists(pseudobulk_file):
                raise FileNotFoundError(f"Pseudobulk file for '{ct}' not found.")

            expr_data = pd.read_csv(pseudobulk_file, index_col=0)
            cluster = [
                "orig" if col.startswith("orig_") else "simulated" if col.startswith("simulated_") else "unknown"
                for col in expr_data.columns
            ]

            expr_data_dict[ct] = expr_data
            cluster_dict[ct] = cluster
        except Exception as e:
            print(f"Error processing cell type '{ct}': {e}")

    # Plot PCA grid
    plot_pca_pseudobulk_grid(
        expr_data_dict=expr_data_dict,
        cluster_dict=cluster_dict,
        var_genes_dict=pca_genes,
        top_n=args.top_n, # since var_genes_dict is provided, this parameter is not applicable
        save_dir=args.simu_obj_dir,
        export=True
    )

if __name__ == "__main__":
    main()








    