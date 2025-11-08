import os
import sys
import glob
import torch
import dgl
from rdkit import Chem
import numpy as np
from scipy.spatial import cKDTree
import datetime
from utils import protein_to_residue_graph, ligand_to_atom_graph, create_homogeneous_interaction_graph

def find_all_directories(base_path):
    all_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, d))]
    return all_dirs

def get_required_files(directory):
    """Obtain mol2 and pdb files"""
    base_name = os.path.basename(directory)
    
    mol2_files = [
        os.path.join(directory, f"{base_name}_b_ligand_minimized.mol2"),
        os.path.join(directory, f"{base_name}_b_ligand.mol2")
    ]
    
    pdb_files = [
        os.path.join(directory, f"{base_name}_pocket_5.pdb"),
        os.path.join(directory, f"{base_name}_receptor.pdb")
    ]
    
    mol2_file = next((f for f in mol2_files if os.path.exists(f)), None)
    pdb_file = next((f for f in pdb_files if os.path.exists(f)), None)
    
    return mol2_file, pdb_file

def generate_interaction_graph(pdb_file, mol2_file, complex_name):
    """Generate the interaction diagram of a single complex"""
    try:
        ligand_mol = Chem.MolFromMol2File(mol2_file, sanitize=False)
        if ligand_mol is None:
            print(f"Unable to read the mol2 file: {mol2_file}")
            return None
        
        interaction_graph = create_homogeneous_interaction_graph(
            protein_pdb=pdb_file,
            ligand_mol=ligand_mol,
            protein_cutoff=30.0, 
            ligand_cutoff=15.0,
            inter_cutoff=12.0
        )
        
        if interaction_graph is not None:
            interaction_graph.complex_name = complex_name
            if 'x' not in interaction_graph.ndata:
                interaction_graph.ndata['x'] = torch.zeros(interaction_graph.number_of_nodes(), 3)
            if 'atom_feat' not in interaction_graph.ndata:
                interaction_graph.ndata['atom_feat'] = torch.zeros(interaction_graph.number_of_nodes(), 21)
            if 'res_feat' not in interaction_graph.ndata:
                interaction_graph.ndata['res_feat'] = torch.zeros(interaction_graph.number_of_nodes(), 20)
            if 'e' not in interaction_graph.edata:
                interaction_graph.edata['e'] = torch.zeros(interaction_graph.number_of_edges(), 8)
        
        return interaction_graph
        
    except Exception as e:
        print(f"Error occurred while generating the interaction diagram {complex_name}: {e}")
        return None

def save_interaction_graph(graph, output_path):
    """Error occurred while generating the interaction diagram {complex_name}: {e}"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        graph_list = [graph]
        dgl.save_graphs(output_path, graph_list)
            
    except Exception as e:
        print(f"Error occurred while saving the interaction diagram: {e}")
        return False

def process_all_complexes(input_base_path, output_base_path):
    """process all complexes"""
    all_dirs = find_all_directories(input_base_path)
    print(f"Found {len(all_dirs)} complexes")
    
    os.makedirs(output_base_path, exist_ok=True)
    print(f"Interaction graph Index: {output_base_path}")
    
    successful_graphs = 0
    failed_graphs = 0
    
    for input_directory in all_dirs:
        complex_name = os.path.basename(input_directory)
        print(f"\nProcessing complex: {complex_name}")
        
        mol2_file, pdb_file = get_required_files(input_directory)
        
        if not mol2_file or not pdb_file:
            print(f"Skip {complex_name}: Missing necessary files.")
            failed_graphs += 1
            continue
        
        graph = generate_interaction_graph(pdb_file, mol2_file, complex_name)
        
        if graph is not None:
            output_directory = os.path.join(output_base_path, complex_name)
            os.makedirs(output_directory, exist_ok=True)
            
            # 保存交互图到新目录
            output_file = os.path.join(output_directory, "interaction_graphs.bin")
            if save_interaction_graph(graph, output_file):
                successful_graphs += 1   
        else:
            failed_graphs += 1
    
    return successful_graphs, failed_graphs

def main():
    input_base_path = "./dataset/TargetNP/TargetNP-4811"
    
    output_base_path = "./data/interaction_graph"
    
    print("Starting to generate the complex interaction diagram...")
    print("=" * 60)
    print(f"Interaction Diagram Index: {output_base_path}")
    print("=" * 60)
    
    successful, failed = process_all_complexes(input_base_path, output_base_path)

if __name__ == "__main__":
    successful_graphs = main()