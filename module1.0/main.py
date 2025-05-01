import argparse

def generate_commands(output_file, module_path, data_path, simu_obj_dir, num_pseudobulk, 
                      colname_of_cellType, colname_of_cellType_heterogeneity, pseudobulk_subsampling, 
                      min_nCells_per_pseudobulk):
    """
    Generate command skeleton for bulk simulation tasks
    """
    
    if not module_path.endswith('/'):
        module_path += '/'
    
    # Commands with all arguments passed to each script
    commands = [
        f"python {module_path}step1_vae_training.py --data_path {data_path} -d {simu_obj_dir} --colname_of_cellType {colname_of_cellType}",
        f"python {module_path}step2_diffusion_training.py --data_path {data_path} -d {simu_obj_dir} --colname_of_cellType {colname_of_cellType}",
        f"python {module_path}step3_classifier_training.py --data_path {data_path} -d {simu_obj_dir} --colname_of_cellType {colname_of_cellType} --colname_of_cellType_heterogeneity {colname_of_cellType_heterogeneity}",
        f"python {module_path}step4_SCANVI_training.py --data_path {data_path} -d {simu_obj_dir} --colname_of_labels {colname_of_cellType}",
        f"python {module_path}step5_sampling.py --data_path {data_path} -d {simu_obj_dir} --num_pseudobulk {num_pseudobulk} --colname_of_cellType {colname_of_cellType} --colname_of_cellType_heterogeneity {colname_of_cellType_heterogeneity}",
        f"python {module_path}step6_latent2expr.py --data_path {data_path} -d {simu_obj_dir} --colname_of_cellType {colname_of_cellType} --colname_of_cellType_heterogeneity {colname_of_cellType_heterogeneity} --pseudobulk_subsampling {pseudobulk_subsampling} --min_nCells_per_pseudobulk {min_nCells_per_pseudobulk}",
        f"python {module_path}step7_pseudobulk_visualization.py -d {simu_obj_dir}"
    ]

    # Write commands to the specified output file
    with open(output_file, "w") as f:
        for command in commands:
            f.write(command + "\n")

    print(f"Commands written to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate a text file of Python commands.")
    parser.add_argument("--output_file", type=str, default="commands.txt",
                        help="The output text file to save the commands.")
    parser.add_argument("--module_path", type=str, required=True,
                        help="Path to the bulkSimu_scDiffusion module.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the data file.")
    parser.add_argument("--simu_obj_dir",'-d', type=str, required=True,
                        help="Name of the simulation object directory.")
    parser.add_argument("--num_pseudobulk", type=int, default=50,
                        help="Number of pseudobulk profiles to generate.")
    parser.add_argument("--colname_of_cellType", type=str, default="cell_type",
                        help="Column name indicating the cell types in the adata object.")
    parser.add_argument("--colname_of_cellType_heterogeneity", type=str, default="sample",
                        help="Column name indicating sources of variation in the adata object.")
    parser.add_argument("--pseudobulk_subsampling", type=int, choices=[0, 1], default=1,
                        help="Enable (1) or disable (0) subsampling of simulated cells.")
    parser.add_argument("--min_nCells_per_pseudobulk", type=int, default=10,
                        help="Minimum number of single cells to aggregate for a pseudobulk profile.")

    args = parser.parse_args()

    generate_commands(
        args.output_file, args.module_path, args.data_path, args.simu_obj_dir,
        args.num_pseudobulk, args.colname_of_cellType,
        args.colname_of_cellType_heterogeneity, args.pseudobulk_subsampling,
        args.min_nCells_per_pseudobulk
    )

if __name__ == "__main__":
    main()
