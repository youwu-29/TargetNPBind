import pandas as pd
import numpy as np
import os
import json, pickle
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit import Chem

max_seq_len = 3000
seq_dict = None
INTERACTION_GRAPHS_ROOT = "./data/interaction_graph"
USE_INTERACTION_GRAPHS = True 

def generate_pharmacophore_data(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Unable to generate the molecule from SMILES: {smiles}.")
        return {}
    
    # Assign a unique mapping number to each atom
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)  
    
    mol_h = Chem.AddHs(mol)
    pharmacophore_groups = detect_pharmacophore_groups(mol_h)
    
    mapped_groups = {}
    for frag_id, atom_indices in pharmacophore_groups.items():
        mapped_indices = []
        for idx in atom_indices:
            atom = mol_h.GetAtomWithIdx(idx)
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                for orig_atom in mol.GetAtoms():
                    if orig_atom.GetAtomMapNum() == map_num:
                        mapped_indices.append(orig_atom.GetIdx())
                        break
        
        if mapped_indices:
            mapped_groups[frag_id] = mapped_indices
    
    return mapped_groups

def detect_pharmacophore_groups(mol_h):
    """Detecting pharmacophores on hydrogenated molecules (with the mapping number retained)"""
    pharmacophore_smarts = {
        'HBD_Group': '[O,N;!H0][H]',
        'HBA_Group': '[C,c](=[O])[O,N;H0]',
        'HYD_Alkyl_Chain': '[C;H2,H1][C;H2,H1]',
        'ARO_Ring_System': 'a1aaaaa1',
        'CARBOXYLIC_ACID': 'C(=O)[O;H1][H]',
        'HYDROXYL': '[O;H1][H]',
        'AMINO_GROUP': '[N;H2,H1][H]',
    }
    
    pharmacophore_groups = {}
    fragment_id = 1
    
    for feat_name, smarts in pharmacophore_smarts.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol_h.GetSubstructMatches(pattern)
            for match in matches:
                heavy_atoms = []
                for atom_idx in match:
                    atom = mol_h.GetAtomWithIdx(atom_idx)
                    if atom.GetAtomicNum() > 1:
                        heavy_atoms.append(atom_idx)
                
                if heavy_atoms:
                    pharmacophore_groups[fragment_id] = heavy_atoms
                    fragment_id += 1
    
    return pharmacophore_groups

def generate_atomic_pharmacophore(mol):
    pharmacophore_groups = {}
    fragment_id = 1

    for atom in mol.GetAtoms():
        pharmacophore_groups[fragment_id] = [atom.GetIdx()]
        fragment_id += 1
    
    return pharmacophore_groups

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom, pharmacophore_groups):
    """Atomic features, including basic features and pharmacophore identification bits"""
    base_features = [
        one_of_k_encoding_unk(atom.GetSymbol(),
                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li',
                               'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']),
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        [atom.GetIsAromatic()]
    ]

    features = [item for sublist in base_features for item in sublist]
    all_pharma_atoms = set()
    for atom_indices in pharmacophore_groups.values():
        all_pharma_atoms.update(atom_indices)
    
    is_pharmacophore = [1] if atom.GetIdx() in all_pharma_atoms else [0]
    features += is_pharmacophore

    features = np.array(features, dtype=np.float32)

    base_len = len(features) - 1

    base_part = features[:base_len]
    pharma_flag = features[-1]
    
    if sum(base_part) > 0:
        base_part = base_part / sum(base_part)
    
    return np.concatenate([base_part, [pharma_flag]])

def smile_to_graph(smile, pharmacophore_groups=None):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, [], [], [], {}
    
    if pharmacophore_groups is not None:
        pharmacophore_groups = pharmacophore_groups
    else:
        print("No pharmacophore groups.")
        pharmacophore_groups = generate_pharmacophore_data(smile)
    
    c_size = mol.GetNumAtoms()

    features = []
    pharma_flags = []
    
    all_pharma_atoms = set()
    for atom_indices in pharmacophore_groups.values():
        all_pharma_atoms.update(atom_indices)
    
    for atom in mol.GetAtoms():
        feature = atom_features(atom, pharmacophore_groups)
        features.append(feature)
        is_pharma = 1 if atom.GetIdx() in all_pharma_atoms else 0
        pharma_flags.append(is_pharma)
        
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index, pharma_flags, pharmacophore_groups

def seq_cat(prot):
    """Convert the protein sequence into a numerical representation"""
    global seq_dict
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

def load_data():
    complexes = {}
    with open('./dataset/TargetNP-4811/INDEX_Target_NP.txt', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            complex_id = parts[0]
            affinity = float(parts[1])
            smiles = parts[3]
            sequence = ' '.join(parts[4:])
            complexes[complex_id] = (smiles, sequence, affinity)
    
    def read_split_file(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    train_val_ids = read_split_file('./dataset/TargetNP-4811/train_protein.txt')
    core_test_ids = read_split_file('./dataset/TargetNP-4811/test_protein.txt') 
    
    return complexes, train_val_ids, core_test_ids

def generate_datasets():

    def read_split_file(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
        
    train_val_ids = read_split_file('./dataset/TargetNP-4811/train_protein.txt')
    core_test_ids = read_split_file('./dataset/TargetNP-4811/test_protein.txt')
    
    all_ids = set(train_val_ids) | set(core_test_ids)
    print(f"Number of training set IDs: {len(train_val_ids)}")
    print(f"Number of independent test set IDs: {len(core_test_ids)}")
    
    complexes = {}
    with open('./dataset/TargetNP-4811/INDEX_Target_NP.txt', 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            complex_id = parts[0]
            
            if complex_id in all_ids:
                try:
                    affinity = float(parts[1])
                    smiles = parts[3]
                    sequence = ' '.join(parts[4:])
                    complexes[complex_id] = (smiles, sequence, affinity)
                except Exception as e:
                    print(f"Parsing {complex_id} failed: {str(e)}")
    
    print(f"Successfully loaded {len(complexes)} complexes")
    
    train_val_data = []
    core_test_data = []
    
    for cid, (smiles, seq, affinity) in complexes.items():
        pharmacophore_groups = generate_pharmacophore_data(smiles)

        pharmacophore_fragments = []
        for frag_id, atom_indices in pharmacophore_groups.items():
            pharmacophore_fragments.append({
                'fragment_id': frag_id,
                'atom_indices': atom_indices
            })
        
        data_item = {
            'complex_id': cid,
            'compound_iso_smiles': smiles,
            'target_sequence': seq,
            'affinity': affinity,
            'pharmacophore_fragments': pharmacophore_fragments  
        }
        
        if cid in train_val_ids:
            train_val_data.append(data_item)
        if cid in core_test_ids:
            core_test_data.append(data_item)

    df_train_val = pd.DataFrame(train_val_data)
    df_core_test = pd.DataFrame(core_test_data)
    df_all = pd.DataFrame(train_val_data + core_test_data)
    
    os.makedirs('data', exist_ok=True)
    df_train_val.to_csv('data/train.csv', index=False)
    df_core_test.to_csv('data/test.csv', index=False)
    
    print(f"Size of the training set: {len(df_train_val)}")
    print(f"Core test set size: {len(df_core_test)}")
    
    return df_train_val, df_core_test, df_all


if __name__ == "__main__":
    USE_INTERACTION_GRAPHS = True
    print(f"Interaction graph usage status: {'Enabled' if USE_INTERACTION_GRAPHS else 'Disabled'}")
    
    df_train_val, df_core_test, df_all = generate_datasets()
    
    all_smiles = set()
    all_smiles.update(df_train_val['compound_iso_smiles'])
    all_smiles.update(df_core_test['compound_iso_smiles'])
    
    smile_pharmacophore_groups = {}
    for _, row in df_all.iterrows():
        smile = row['compound_iso_smiles']
        
        fragments_list = row['pharmacophore_fragments']
        pharmacophore_groups = {}
        for frag in fragments_list:
            frag_id = frag['fragment_id']
            atom_indices = frag['atom_indices']
            pharmacophore_groups[frag_id] = atom_indices
        
        if smile not in smile_pharmacophore_groups:
            smile_pharmacophore_groups[smile] = pharmacophore_groups
    
    smile_graph = {}
    for smile in all_smiles:
        pharmacophore_groups = smile_pharmacophore_groups.get(smile, {})
        g = smile_to_graph(smile, pharmacophore_groups=pharmacophore_groups)
        smile_graph[smile] = g
    
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
    print(f"max_seq_len={max_seq_len}")
    
    datasets = [
        ('train', df_train_val),
        ('test', df_core_test)
    ]
    
    for dataset_name, df in datasets:
        root_path = 'data/processed'
        os.makedirs(root_path, exist_ok=True)
        processed_file = os.path.join(root_path, f'{dataset_name}.pt')
        
        if not os.path.isfile(processed_file):
            drugs = df['compound_iso_smiles'].values
            prots = df['target_sequence'].values
            Y = df['affinity'].values
            
            XT = [seq_cat(t) for t in prots]
            
            print(f'Preparing {dataset_name}.pt in pytorch format!')
            dataset = TestbedDataset(
                root=root_path,
                dataset=dataset_name,
                xd=drugs,
                xt=XT,
                y=Y,
                smile_graph=smile_graph,
                interaction_graphs_root=INTERACTION_GRAPHS_ROOT,
                df=df,
                protein_seqs=prots.tolist()
            )
            print(f'{processed_file} has been created')
        else:
            print(f'{processed_file} already exists')

