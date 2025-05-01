import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import anndata as ad
import scanpy as sc
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import scipy.sparse

from simple_scvi._mymodel import MyModel

class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        out_dict = {}
        if self.class_name is not None:
            out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        return arr, out_dict

def get_classifier_map(data_path,
                       colname_of_cellType = 'cell_type',
                       colname_of_cellType_heterogeneity= 'sample',                       
                       selected_cell_type = None, # set to None to include all cell types
                       min_nCells_per_class = 10 # keep this consistent with prepare_adata_for_augmentation()
):
    
    ori_adata = ad.read_h5ad(data_path, backed='r')

    label_encoder = LabelEncoder()
    if selected_cell_type is None:

        label_counts = ori_adata.obs[colname_of_cellType].value_counts()        
        valid_labels = label_counts[label_counts >= min_nCells_per_class].index  
        filtered_obs = ori_adata.obs[ori_adata.obs[colname_of_cellType].isin(valid_labels)]
        label_encoder.fit(ori_adata.obs[colname_of_cellType])
        label_counts_final = filtered_obs[colname_of_cellType].value_counts().to_dict()      
        
    elif isinstance(selected_cell_type, str):
        filtered_obs = ori_adata.obs[ori_adata.obs[colname_of_cellType] == selected_cell_type]
        label_counts = filtered_obs[colname_of_cellType_heterogeneity].value_counts()
        valid_labels = label_counts[label_counts >= min_nCells_per_class].index        
        filtered_obs = filtered_obs[filtered_obs[colname_of_cellType_heterogeneity].isin(valid_labels)]        
        label_encoder.fit(filtered_obs[colname_of_cellType_heterogeneity])
        label_counts_final = filtered_obs[colname_of_cellType_heterogeneity].value_counts().to_dict()     

    else:
        print("Invalid input: 'selected_cell_type' must be a string or None.")

    label_to_integer = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return label_to_integer, label_counts_final
    
def prepare_adata_for_augmentation(data_path,
                                   colname_of_cellType='cell_type',
                                   colname_of_cellType_heterogeneity='sample',
                                   selected_cell_type='Malignant', # set to None to include all cell types,
                                   min_nCells_per_class = 10 # only return adata with nCells_per_class greater than this thresholod
                                  ):

    # Load the data
    ori_adata = sc.read_h5ad(data_path)  
    sc.pp.filter_genes(ori_adata, min_cells=10)  

    if selected_cell_type is None:
        adata = ori_adata.copy()

    elif isinstance(selected_cell_type, str):
        print(f"Filtering adata to include the selected cell type: '{selected_cell_type}'.")
        adata = ori_adata[ori_adata.obs[colname_of_cellType] == selected_cell_type, :].copy()

    elif isinstance(selected_cell_type, list) and len(selected_cell_type) == 1:
        print(f"Filtering adata to include the selected cell type: '{selected_cell_type[0]}'.")
        adata = ori_adata[ori_adata.obs[colname_of_cellType] == selected_cell_type[0], :].copy()

    elif isinstance(selected_cell_type, list):
        print(f"Filtering adata to include the selected cell types: {', '.join(selected_cell_type)}.")
        print("The cell type identifier will serve as the class label for conditional augmentation.")
        adata = ori_adata[ori_adata.obs[colname_of_cellType].isin(selected_cell_type), :].copy()

    else:
        raise ValueError("Invalid input: 'selected_cell_type' must be a string, list, or None.")
        
    print('adata shape:', adata.shape, '\n')
    
    if isinstance(adata.X, scipy.sparse.csc_matrix):
        adata.X = adata.X.tocsr()
        
    label_encoder = LabelEncoder()
    if selected_cell_type is None or (isinstance(selected_cell_type, list) and len(selected_cell_type) > 1):
        # Using `cell_type` as the class label
        label_counts = adata.obs[colname_of_cellType].value_counts()        
        valid_labels = label_counts[label_counts >= min_nCells_per_class].index        
        adata = adata[adata.obs[colname_of_cellType].isin(valid_labels)].copy()

        num_classes = adata.obs[colname_of_cellType].nunique()
        label_encoder.fit(adata.obs[colname_of_cellType])
        classes = label_encoder.transform(adata.obs[colname_of_cellType])
        
    else:
        # Using `colname_of_cellType_heterogeneity` as the class label
        label_counts = adata.obs[colname_of_cellType_heterogeneity].value_counts()        
        valid_labels = label_counts[label_counts >= min_nCells_per_class].index
        adata = adata[adata.obs[colname_of_cellType_heterogeneity].isin(valid_labels)].copy()
        
        num_classes = adata.obs[colname_of_cellType_heterogeneity].nunique()
        label_encoder.fit(adata.obs[colname_of_cellType_heterogeneity])
        classes = label_encoder.transform(adata.obs[colname_of_cellType_heterogeneity])
        
    return adata, num_classes, classes 

# pending: check if we need to use orig_adata in MyModel.load
def load_data_embedding(adata,
                        classes,
                        vae_dir = './',
                        vae_prefix = '',
                        batch_size = 12):

    '''
    create a latent embedding dataloader (for diffusin and classifier training),
    using the vae model trained on the selected cell type, project gene expression to latent space and export as a dataloader
    '''
    print('load trained vae model')
    vae = MyModel.load(dir_path = vae_dir, adata=adata,prefix = vae_prefix)

    print('project gene expression to latent space')
    myinput = torch.tensor(adata.X.toarray())  
    with torch.no_grad():
        outputs = vae.module.inference(myinput)
    
    dataset = CellDataset(outputs['z'].cpu().detach().numpy(),classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    while True:
        yield from loader 

def prepare_ori_adata(data_path):
    '''
    load original scRNA data (for SCANVI training and visualization)
    '''
    adata = sc.read_h5ad(data_path)  
    sc.pp.filter_genes(adata, min_cells=10)  
    
    if isinstance(adata.X, scipy.sparse.csc_matrix):
        adata.X = adata.X.tocsr()
        
    return adata