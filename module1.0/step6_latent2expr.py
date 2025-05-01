import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import random
import itertools
import scvi
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from natsort import natsorted
from scipy.sparse import csr_matrix, vstack

from data_loader import *
from utils import *
from visualization import *
from configs import target_sum, nCells_for_libsize_calc, pseudobulk_nCells_dist, classifier_requirements_defaults


def main():
    parser = argparse.ArgumentParser(description="Parser for sampling")
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory containing necessary models for sampling')
    parser.add_argument('--latent_libsize', type=str, default = 'fixed', choices=['fixed', 'sampled'], help = "Method to determine the latent library size for the scVI generative model: 'fixed' uses a predefined size, while 'sampled' calculates and samples from the original AnnData. Default is 'fixed'")
    parser.add_argument('--fixed_latent_libsize', type=float, default=9.0, help='latent libsize (in log space) pass to scVI generative model. Used only when `latent_libsize` is set to `fixed`')

    parser.add_argument('--export_adata', type=int, choices=[0, 1], default=0, help="Boolean parameter indicating whether to export the AnnData object for the concatenated original and simulated single-cell data. If enabled, a `concatenated_adata.h5ad` file will be created in the `simulation_res` subfolder. Default is 0") 
    parser.add_argument('--generate_pseudobulk', type=int, choices=[0, 1], default=1, help="Boolean parameter indicating whether to create pseudobulk profiles for the simulated single cells. If enabled, a pseudobulk CSV file will be generated for each cell type in the `simulation_res` subfolder. Default is 1.")
    parser.add_argument('--plot_umap', type=int, choices=[0, 1], default=1, help="Enable (1) or disable (0) umap visualization for the simulated single cells. If enabled, umaps will be created in the `umap` folder in the `simulation_res` subfolder")
 
    parser.add_argument('--pseudobulk_subsampling', type=int, choices=[0, 1], default=0, help='Enable (1) or disable (0) subsampling of simulated cells when aggregating them to create pseudobulk profiles. When enabled, the number of cells to aggregate (nCells) for each cell type in each pseudobulk profile is determined based on the pseudobulk nCells from the original AnnData object')
    parser.add_argument('--pseudobulk_nCells_guide_file', type=str, default = 'pseudobulk_nCells_guide.csv',help = 'The file name inside pseudobulk_configs subfolder in simu_obj_dir which specifies the parameters that determine the number of cells to be aggregated during pseudobulk calculation. If the file is not available, it will be automatically generated using pseudobulk_nCells derived from the original adata.')    
    parser.add_argument('--min_nCells_per_pseudobulk', type=int, default=10, help="minimum number of single cells to aggreate to make the pseudobulk profile. Pseudobulk profiles will not be generated for pseudobulkIDs with fewer cells than this value.")
    parser.add_argument('--include_orig_pseudobulk', type=int, choices=[0, 1], default=1, help="Boolean parameter (0 or 1) indicating whether to create a pseudobulk profile for the original `adata` object. When enabled, `colname_of_cellType_heterogeneity` parameter is required")
    
    parser.add_argument('--filter', type=int, choices=[0, 1], default=1, help="Enable (1) or disable (0) cell filtering based on predicted cell_type probabilities. If enabled, a pretrained SCANVI model is required for cell type prediction")
    parser.add_argument('--prob_cut', type=float, default=0.8, help='When filter = TRUE, only cells with a predicted cell-type probability above this threshold will be retained for pseudobulk calculation and visualization')
    
    parser.add_argument('--SCANVI_max_epochs', type=int, default=20, help="training steps for SCANVI query model")
    parser.add_argument('--colname_of_cellType', type=str, default='cell_type',help = 'column name indicating the cell types of single cells from the adata object')
    parser.add_argument('--colname_of_cellType_heterogeneity', type=str, default='sample',help = 'Column name specifying the sources of variation (e.g., sample ID, cell states) within each cell type for single cells in the adata object. Required when `include_orig_pseudobulk` is enabled')

    args = parser.parse_args()
    if not os.path.isdir(args.simu_obj_dir):
        raise ValueError(f"The provided simu_obj_dir '{args.simu_obj_dir}' is not a valid directory.")
        
    if args.export_adata == 0 and args.generate_pseudobulk == 0 and args.plot_umap == 0:
        raise ValueError("No outputs option selected: At least one of --export_adata, --generate_pseudobulk, or --plot_umap must be enabled (set to 1).")
        
    args = vars(args)

    # load the trained vae model
    ori_adata = prepare_ori_adata(args['data_path'])
    vae = MyModel.load(dir_path = args['simu_obj_dir'], adata=ori_adata,prefix = 'scvi_')

    # load simulated res
    simulated_file_path = os.path.join(args['simu_obj_dir'], "simulation_res",'simulated_latent.npz')
    simulated_res = np.load(simulated_file_path)
    simulated_latent = simulated_res['simulated_latent']
    simulated_cell_types = simulated_res['simulated_cell_types']
    simulated_pseudobulkID = simulated_res['pseudobulkID']
    
    n_samples = simulated_latent.shape[0]
    
    if args['latent_libsize'] == 'sampled':
       # calculate latent library sizes from the original AnnData and sample from them for the scVI generative model
        random_indices = np.random.choice(ori_adata.n_obs, size=min(nCells_for_libsize_calc, ori_adata.n_obs), replace=False)
        adata_subset = ori_adata[random_indices, :].copy()
        myinput = torch.tensor(adata_subset.X.toarray())  
        with torch.no_grad():
            outputs = vae.module.inference(myinput)
        sampled_indices = torch.multinomial(outputs['library'].flatten(), num_samples=n_samples, replacement=True)
        library_size = outputs['library'].flatten()[sampled_indices].unsqueeze(1)
        simulated_expr = latent2expr(simulated_latent,vae,library_size)
        
    elif args['latent_libsize'] == 'fixed':
        simulated_expr = latent2expr(simulated_latent,vae,library_size=None,fixed_libsize = args['fixed_latent_libsize'])
        
    else:
        raise ValueError("Invalid input for 'latent_libsize': it must be either 'sampled' or 'fixed'.")

    colname_of_cellType = args['colname_of_cellType']
    colname_of_cellType_heterogeneity = args['colname_of_cellType_heterogeneity']

    # build adata for simulated single cells
    query_adata = ad.AnnData(simulated_expr)
    query_adata.var_names = ori_adata.var_names
    query_adata.obs['cell_type'] = list(simulated_cell_types) 
    query_adata.obs['pseudobulkID'] = ['simulated_' + str(id_) for id_ in simulated_pseudobulkID]
    query_adata.obs['prob'] = 1.0 

    # calculate cell type probabilities using the trained SCANVI model
    # note: currently, filtering is supported only based on predicted cell type probabilities. 
    # Further filtering within each cell type, such as by heterogeneity probabilities, is not yet implemented
    if bool(args['filter']):
        scanvi_model = scvi.model.SCANVI.load(dir_path = args['simu_obj_dir'],
                                              adata = ori_adata,
                                              prefix = 'celltypeAnno_SCANVI_')
        
        probs = SCANVI_query_prob(query_adata,
                                  scanvi_model,
                                  colname_of_cellType,
                                  args['SCANVI_max_epochs'],
                                  args['simu_obj_dir'],
                                  'celltypeAnno_SCANVI_')
        print(isinstance(probs, np.ndarray))
        probs.to_csv(os.path.join(args['simu_obj_dir'],'simulation_res','celltype_probs.csv'),index = False)
        
        query_adata.obs['cell_type'] = list(simulated_cell_types) # after SCANVI_query_prob(), cell_type will be set to 'Unknown', therefore should reverse to simulated_cell_types

        probs = pd.read_csv(os.path.join(args['simu_obj_dir'],'simulation_res','celltype_probs.csv'))
        
        for ct in probs.columns:  
            ct_indices = query_adata.obs[query_adata.obs['cell_type'] == ct].index.astype(int)
            query_adata.obs.loc[ct_indices.astype(str), 'prob'] = probs.loc[ct_indices, ct].values
            
        # query_adata.obs.to_csv(os.path.join(args['simu_obj_dir'],'simulation_res','simulated_obs.csv'),index = False)

    # Modify the .obs attribute in ori_adata to prepare for concatenation with simulated_adata
    ori_adata.obs = pd.DataFrame({
        'cell_type': ori_adata.obs[colname_of_cellType],
        'pseudobulkID': (
            'orig_' + ori_adata.obs[colname_of_cellType_heterogeneity].astype(str) if colname_of_cellType_heterogeneity in ori_adata.obs.columns else 'orig'),
        'prob' : [1] * ori_adata.n_obs}, index=ori_adata.obs.index)

    concatenate_adata = ori_adata.concatenate(
    query_adata, 
    join='outer',  
    batch_key='source',  
    batch_categories=['original', 'simulated'])
    
    if bool(args['export_adata']):
        print(f"Warning: Attempting to save an AnnData object with shape {adata.shape}. This process may take a long time.")
        concatenate_adata.write(os.path.join(args['simu_obj_dir'], "simulation_res",'concatenate_adata.h5ad'))

    if bool(args['generate_pseudobulk']) or bool(args['plot_umap']):
        celltype_classifier_map, _ = get_classifier_map(args['data_path'],
                                                     args['colname_of_cellType'],
                                                     'sample', # this parameter is not required and will pass a non-meaningful value
                                                     None,
                                                     classifier_requirements_defaults()['min_nCells_per_class'])
        cell_types = list(celltype_classifier_map.keys())

    if bool(args['generate_pseudobulk']):

        os.makedirs(os.path.join(args['simu_obj_dir'], "pseudobulk_configs"), exist_ok=True)
        
        # create pseudobulk for each cell type
        adata = concatenate_adata.copy()

        # cpm normalize before aggregation
        sc.pp.normalize_total(adata, target_sum = target_sum)
        scExpr = adata.X.T

        # initialize pseudobulk_subsampling guiding file if not provided (only required when pseudobulk_subsampling = True)
        if bool(args['pseudobulk_subsampling']):
            subsampling_guide_path = os.path.join(args['simu_obj_dir'],'pseudobulk_configs',args['pseudobulk_nCells_guide_file'])
            
            if not os.path.exists(subsampling_guide_path):
                subsampling_guide = pseudobulk_nCells_by_ct(adata.obs[adata.obs['source'] == 'original'],'cell_type','pseudobulkID')
                subsampling_guide['dist'] = pseudobulk_nCells_dist
                
                subsampling_guide.to_csv(subsampling_guide_path,index = False)
            else:
                subsampling_guide = pd.read_csv(subsampling_guide_path)
                required_columns = {'cell_type', 'min_nCells', 'max_nCells','dist'}
                
                if (
                    not required_columns.issubset(subsampling_guide.columns) 
                    or not subsampling_guide.get('dist').isin(['beta', 'uniform']).any()
                    or not set(cell_types).issubset(subsampling_guide['cell_type'])
                ):
                    print("Required columns for the subsampling guide are either missing in the provided pseudobulk_nCells_guide_file or the format is incorrect. A new guide will be created using nCells from the original data and will overwrite the existing one.")
                    
                    subsampling_guide = pseudobulk_nCells_by_ct(
                        adata.obs[adata.obs['source'] == 'original'], 'cell_type', 'pseudobulkID'
                    )
                    subsampling_guide['dist'] = pseudobulk_nCells_dist
                    subsampling_guide.to_csv(subsampling_guide_path, index = False)
                else:
                    print('Using provided pseudobulk_nCells_guide_file to guide nCells sampling for pseudobulk calculation')

        # Initialize lists to store nCells_per_pseuodbulk
        all_pseudobulk_ids = []
        all_cell_types = []
        all_nCells_per_pseudobulk = []

        for ct in cell_types:
            print(ct)
            
            ct_indices_simulated = (adata.obs['cell_type'] == ct) & (adata.obs['source'] == 'simulated') & (adata.obs['prob'] >= args['prob_cut'])
            # Initialize variables to handle cases where an error occurs
            pseudobulk_ct_simulated = None
            nCells_per_pseudobulk = []
            
            # Perform pseudobulk aggregation with or without subsampling
            try:
                if bool(args['pseudobulk_subsampling']):
                    pseudobulk_ct_simulated, nCells_per_pseudobulk = aggregate_by_group(
                        scExpr[:, ct_indices_simulated],
                        adata.obs.loc[ct_indices_simulated, 'pseudobulkID'].to_numpy(),
                        adata.var_names,
                        args['min_nCells_per_pseudobulk'],
                        True,
                        subsampling_guide.loc[subsampling_guide['cell_type'] == ct, 'min_nCells'].values[0],
                        subsampling_guide.loc[subsampling_guide['cell_type'] == ct, 'max_nCells'].values[0],
                        subsampling_guide.loc[subsampling_guide['cell_type'] == ct, 'dist'].values[0]
                    )
                else:
                    pseudobulk_ct_simulated, nCells_per_pseudobulk = aggregate_by_group(
                        scExpr[:, ct_indices_simulated],
                        adata.obs.loc[ct_indices_simulated, 'pseudobulkID'].to_numpy(),
                        adata.var_names,
                        args['min_nCells_per_pseudobulk'],
                        pseudobulk_subsampling=False
                    )
            except ValueError as e:
                print(f"Error during pseudobulk aggregation for {ct}: {e}")
                    
            # Store results if no error occurred
            if pseudobulk_ct_simulated is not None:
                all_pseudobulk_ids.extend(pseudobulk_ct_simulated.columns)
                all_cell_types.extend([ct] * len(nCells_per_pseudobulk))
                all_nCells_per_pseudobulk.extend(nCells_per_pseudobulk)
            
            if bool(args['include_orig_pseudobulk']):
                ct_indices_orig = (adata.obs['cell_type'] == ct) & (adata.obs['source'] == 'original')
                
                try:
                    pseudobulk_ct_orig, nCells_per_pseudobulk = aggregate_by_group(
                        scExpr[:, ct_indices_orig],
                        adata.obs.loc[ct_indices_orig, 'pseudobulkID'].to_numpy(),
                        adata.var_names,
                        args['min_nCells_per_pseudobulk'],
                        pseudobulk_subsampling=False
                    )
                    
                    all_pseudobulk_ids.extend(pseudobulk_ct_orig.columns)
                    all_cell_types.extend([ct] * len(nCells_per_pseudobulk))
                    all_nCells_per_pseudobulk.extend(nCells_per_pseudobulk)
        
                    # Combine original and simulated pseudobulk data
                    if pseudobulk_ct_simulated is not None:
                        pseudobulk_ct = pd.concat([pseudobulk_ct_orig, pseudobulk_ct_simulated], axis=1)
                    else:
                        pseudobulk_ct = pseudobulk_ct_orig
        
                    pseudobulk_ct.to_csv(
                        os.path.join(args['simu_obj_dir'], "simulation_res", f"{ct}_pseudobulk.csv"), index=True
                    )
                    
                except ValueError as e:
                    print(f"Error during original pseudobulk aggregation for {ct}: {e}")
                    
                    if pseudobulk_ct_simulated is not None and not pseudobulk_ct_simulated.empty:
                        pseudobulk_ct_simulated.to_csv(os.path.join(args['simu_obj_dir'], "simulation_res", f"{ct}_pseudobulk.csv"), index=True)
                    else:
                        print("Simulated pseudobulk DataFrame is empty. Skipping saving to CSV.")
            else:
                if pseudobulk_ct_simulated is not None and not pseudobulk_ct_simulated.empty:
                    pseudobulk_ct_simulated.to_csv(os.path.join(args['simu_obj_dir'], "simulation_res", f"{ct}_pseudobulk.csv"), index=True)
                else:
                    print("Simulated pseudobulk DataFrame is empty. Skipping saving to CSV.")

        nCells_df = pd.DataFrame({
            "cell_type": all_cell_types,
            "pseudobulkID": all_pseudobulk_ids,
            "nCells_per_pseudobulk": all_nCells_per_pseudobulk
        })
        
        nCells_df.to_csv(os.path.join(args['simu_obj_dir'],'pseudobulk_configs', "pseudobulk_nCells.csv"), index=False)

    if bool(args['plot_umap']):
        plot_umap_all_ct(concatenate_adata,args['prob_cut'],args['simu_obj_dir'],True)

        for ct in cell_types:
            try:
                plot_umap_single_ct(ct,concatenate_adata,args['prob_cut'],args['simu_obj_dir'],True)
            except ValueError as e:
                print(e)
                continue
                
if __name__ == "__main__":
    main()



        

            