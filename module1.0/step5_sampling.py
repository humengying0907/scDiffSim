import os
import sys
import warnings
import argparse
import random
import glob
from natsort import natsorted
from math import ceil

warnings.simplefilter(action='ignore', category=FutureWarning)

from data_loader import *
from scDiffusion_sampling import *
from utils import build_sampling_config
from configs import sampling_defaults, classifier_requirements_defaults

def main():
    parser = argparse.ArgumentParser(description="Parser for sampling")
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--simu_obj_dir", '-d', type=str, help = 'directory containing necessary models for sampling')
    parser.add_argument('--celltype_heterogeneity', type=int, choices=[0, 1], default=1, help="Enable (1) or disable (0) multi-condition sampling. When enabled, for cell types with a trained heterogeneity classifier, the simulated single cells will reflect the variation defined during the heterogeneity classifier training, such as sampleID, source of origin, or cell states.") 

    parser.add_argument("--nCells_mode", type=str,choices = ['auto','fixed'], default = 'auto', help = 'Method to determine number of cells per batch for each cell type. When set to `auto`, `num_cells_per_batch` is determined as follows: for cell types without a heterogeneity classifier, it is set to half the size of the cell type; for cell types with a heterogeneity classifier, it is set to the maximum number of cells per heterogeneity level')
    parser.add_argument('--num_cells_per_batch', type=int, default=500, help="Number of simulated single cells per batch. In pseudobulk calculation, single cells within the same batch are aggregated to create a pseudobulk profile for a single sample. Applicable only when `nCells_per_ct` is set to `fixed`.")
    parser.add_argument('--num_pseudobulk', type=int, default=50, help = 'Number of pseudobulk profiles to generate. For cell types with a heterogeneity classifier, `num_pseudobulk` / `n_heter` batches are used during simulation, as each batch generates `n_heter` profiles. For cell types without a heterogeneity classifier, the simulation uses `num_pseudobulk` batches directly')
    
    parser.add_argument('--n_latent', type=int, default=128, help="number of latent dimensions in vae and diffusion model")
    parser.add_argument('--colname_of_cellType', type=str, default='cell_type',help = 'Column name indicating the cell types labels from the adata object')
    parser.add_argument('--colname_of_cellType_heterogeneity', type=str, default='sample',help = 'Column name indicating the sources of variation (e.g., sample ID, cell states) within each cell type for single cells in the adata object. Required when `celltype_heterogeneity` is enabled')

    args = parser.parse_args()

    if not os.path.isdir(args.simu_obj_dir):
        raise ValueError(f"The provided simu_obj_dir '{args.simu_obj_dir}' is not a valid directory.")

    celltype_classifiers = natsorted(glob.glob(os.path.join(args.simu_obj_dir, "celltype_classifier_*.pt")))
    if not celltype_classifiers:
        raise ValueError('A trained celltype classifier is required.')

    diffusion_models = natsorted(glob.glob(os.path.join(args.simu_obj_dir, "diffusion_*.pt")))
    if not diffusion_models:
        raise ValueError('A trained diffusion model is required.')

    n_latent = args.n_latent
    
    args = vars(args)
    sampling_args = sampling_defaults()
    
    export_dir = os.path.join(args['simu_obj_dir'], "simulation_res")
    os.makedirs(export_dir, exist_ok=True)
    
    celltype_classifier_map, ct_counts_dict = get_classifier_map(args['data_path'],
                                                 args['colname_of_cellType'],
                                                 'sample', # this parameter is not required here and will pass a non-meaningful value
                                                 None,
                                                 classifier_requirements_defaults()['min_nCells_per_class'])
    
    celltype_classifier_num_classes = len(celltype_classifier_map)
    cell_types = list(celltype_classifier_map.keys())

    celltype_labels_all = []
    cell_gen_all = []
    pseudobulkID_all = []


    if bool(args['celltype_heterogeneity']):
        heterogeneity_dir = os.path.join(args['simu_obj_dir'], 'heterogeneity_classifiers')
        if not os.path.isdir(heterogeneity_dir):
            raise ValueError(f"When `celltype_heterogeneity` is enabled, heterogeneity classifiers are required.")
        
        configure_df_all = pd.DataFrame(columns=["batch", "cell_type", "diffusion_model",
                                             "celltype_classifier","heter_classifier",
                                             "nCells_per_ct","heterogeneity"])
        
        for ct in cell_types:
            print(ct)
            
            search_pattern = os.path.join(heterogeneity_dir, f"{ct}_heterogeneity_classifier_*.pt")
            heter_classifiers = natsorted(glob.glob(search_pattern))
            
            if len(heter_classifiers) == 0:
                print(f"No heterogeneity classifier is available for {ct}. Multi-batch simulations will be conducted conditioned on cell-type classifier only")

                if args['nCells_mode'] == 'auto':
                    num_cells = ct_counts_dict[ct]//2
                elif args['nCells_mode'] == 'fixed':
                    num_cells = args['num_cells_per_batch']
                else:
                    raise ValueError(f"Invalid value for nCells_mode: {args['nCells_mode']}. Expected 'auto' or 'fixed'.")

                cell_gen_ct = []                
                max_batch = args['num_pseudobulk']    
                configure_df = build_sampling_config(ct,
                                                     max_batch,
                                                     diffusion_models,
                                                     celltype_classifiers,
                                                     heter_classifiers,
                                                     num_cells)
                                    
                for batch in range(max_batch):
                    arr = condi_sampling(ct,
                                         configure_df.at[batch, "diffusion_model"],
                                         configure_df.at[batch, "celltype_classifier"],
                                         celltype_classifier_map,
                                         num_cells,
                                         n_latent,
                                         **sampling_args)
                    dist.destroy_process_group()

                    cell_gen_ct.append(arr) 

                cell_gen_ct = np.vstack(cell_gen_ct)

                # update simulation res
                configure_df_all = pd.concat([configure_df_all,configure_df],ignore_index=True)
                pseudobulkID_all += list(np.repeat(range(max_batch),num_cells))
                cell_gen_all.append(cell_gen_ct)
                celltype_labels_all.extend([ct] * (num_cells *args['num_pseudobulk']))

            else:
                heter_classifier_map, cs_counts_dict = get_classifier_map(args['data_path'],
                                             args['colname_of_cellType'],
                                             args['colname_of_cellType_heterogeneity'],
                                             ct,
                                             classifier_requirements_defaults()['min_nCells_per_class'])
                
                max_batch = ceil(args['num_pseudobulk']/len(heter_classifier_map))
                
                if args['nCells_mode'] == 'auto':
                    num_cells = max(cs_counts_dict.values())
                elif args['nCells_mode'] == 'fixed':
                    num_cells = args['num_cells_per_batch']
                else:
                    raise ValueError(f"Invalid value for nCells_mode: {args['nCells_mode']}. Expected 'auto' or 'fixed'.")
                
                configure_df = build_sampling_config(ct,
                                                     max_batch,
                                                     diffusion_models,
                                                     celltype_classifiers,
                                                     heter_classifiers,
                                                     num_cells)
                cell_gen_ct = []
                for batch in range(max_batch):                    
                    arr = multi_condi_sampling(ct,
                                               configure_df.at[batch, "diffusion_model"],
                                               configure_df.at[batch, "celltype_classifier"],
                                               configure_df.at[batch, "heter_classifier"],
                                               celltype_classifier_map,
                                               heter_classifier_map,
                                               num_cells,
                                               n_latent,
                                               **sampling_args)
                    dist.destroy_process_group()

                    cell_gen_ct.append(arr) 
                    
                cell_gen_ct = np.vstack(cell_gen_ct)
                
                pseudobulkID_raw = list(np.repeat([f"{item}_{i}" for i in range(max_batch) for item in heter_classifier_map.keys()], num_cells))
                
                selected_unique_pseuodbulkID = random.sample(list(np.unique(pseudobulkID_raw)), args['num_pseudobulk'])
                indices = [i for i, value in enumerate(pseudobulkID_raw) if value in selected_unique_pseuodbulkID]
                
                # update simulation res
                configure_df_all = pd.concat([configure_df_all,configure_df],ignore_index=True)
                pseudobulkID_all += [pseudobulkID_raw[i] for i in indices]
                cell_gen_all.append(cell_gen_ct[indices])
                celltype_labels_all.extend([ct] * (num_cells *args['num_pseudobulk']))

    else:
        
        max_batch = args['num_pseudobulk']        
        configure_df_all = build_sampling_config(cell_types,
                                             max_batch,
                                             diffusion_models,
                                             celltype_classifiers)
        
        for ct in cell_types:
            if args['nCells_mode'] == 'auto':
                num_cells = ct_counts_dict[ct]//2
            elif args['nCells_mode'] == 'fixed':
                num_cells = args['num_cells_per_batch']
            else:
                raise ValueError(f"Invalid value for nCells_mode: {args['nCells_mode']}. Expected 'auto' or 'fixed'.")
                
            for batch in range(max_batch):
                diffusion_path = configure_df_all.at[batch, "diffusion_model"]
                classifier_path = configure_df_all.at[batch, "classifier_model"]
                arr = condi_sampling(ct,
                                     diffusion_path,
                                     classifier_path,
                                     celltype_classifier_map,
                                     num_cells,
                                     n_latent,
                                     **sampling_args)
                dist.destroy_process_group()

                cell_gen_all.append(arr) 
                
            pseudobulkID_all += list(np.repeat(range(max_batch),num_cells))
            celltype_labels_all.extend([ct] * (num_cells *args['num_pseudobulk']))

    # merge config rows if they share the same paramters
    config_export = (
        configure_df_all.groupby([col for col in configure_df_all.columns if col != 'batch'], as_index=False)
        .agg({'batch': lambda x: ','.join(map(str, sorted(x)))})
    )

    
    if not bool(args['celltype_heterogeneity']):
        if args['nCells_mode'] == 'auto':
            nCells_per_ct_dict = {key: value // 2 for key, value in ct_counts_dict.items()}
        elif args['nCells_mode'] == 'fixed':
            nCells_per_ct_dict = {ct: args['num_cells_per_batch'] for ct in cell_types}

        expanded_rows = []
        for _, row in config_export.iterrows():
            for ct in cell_types:
                new_row = row.copy()  
                new_row["ct"] = ct    
                new_row["nCells_per_ct"] = nCells_per_ct_dict[ct]  
                expanded_rows.append(new_row)    
        config_export = pd.DataFrame(expanded_rows, index=[idx for idx in range(len(expanded_rows))])
    
    config_export.to_csv(
        os.path.join(args['simu_obj_dir'], "simulation_res", "simulation_config.csv"),
        index=False
    )
    file_path = os.path.join(export_dir, "simulated_latent.npz")

    np.savez(file_path, 
             simulated_latent = np.vstack(cell_gen_all),
             simulated_cell_types = celltype_labels_all,
             pseudobulkID = pseudobulkID_all)     

if __name__ == "__main__":
    main()

    




