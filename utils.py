import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import dgl
from rdkit import Chem
from rdkit.Chem import rdchem
import torch.nn as nn
from scipy.spatial import cKDTree
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist

USE_INTERACTION_GRAPHS = True

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='data/processed', dataset='train',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None, smile_graph=None,
                 interaction_graphs_root=None, df=None, protein_seqs=None):

        self.root = root
        self.df = df
        self.interaction_graphs_root = interaction_graphs_root
        self.protein_seqs = protein_seqs 

        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']


    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]

            c_size, features, edge_index, pharma_flags, pharmacophore_groups = smile_graph[smiles]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Unable to generate molecules: {smiles}")
                continue

            complex_id = self.df.iloc[i]['complex_id']  
            interaction_graphs = self._load_interaction_graphs(complex_id)

            if self.protein_seqs is not None:
                raw_protein_seq = self.protein_seqs[i]
            else:
                print("raw_protein_seq = \"\"")
                raw_protein_seq = ""

            #valid
            valid_pharmacophore_groups = {}
            for frag_id, atom_indices in pharmacophore_groups.items():
                valid_atom_indices = []
                for idx in atom_indices:
                    if idx < mol.GetNumAtoms():
                        valid_atom_indices.append(idx)
                    else:
                        print(f"Atomic index {idx} is out of range")
                if valid_atom_indices:
                    valid_pharmacophore_groups[frag_id] = valid_atom_indices

            #dicttolist
            pharmacophore_groups_list = []
            for frag_id, atom_indices in valid_pharmacophore_groups.items():
                pharmacophore_groups_list.append((frag_id, atom_indices))

            data = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels]),
                target=torch.LongTensor([target]),
                interaction_graphs=interaction_graphs,
                protein_seq=raw_protein_seq,
                pharma_flags=torch.Tensor(pharma_flags),
                pharmacophore_groups=pharmacophore_groups_list  # list
            )
            data.c_size = torch.LongTensor([c_size])
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def _load_interaction_graphs(self, complex_id):
        if not USE_INTERACTION_GRAPHS:
            return []

        if not self.interaction_graphs_root:
            return []

        graph_dir = os.path.join(self.interaction_graphs_root, complex_id)
        graph_file = os.path.join(graph_dir, 'interaction_graphs.bin')

        if os.path.exists(graph_file):
            try:
                graphs, _ = dgl.load_graphs(graph_file)
                return graphs
            except Exception as e:
                return []
        else:
            print(f"The interaction diagram file does not exist: {graph_file}")
            return []


class UnifiedEdgeProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_prot = nn.Sequential(
            nn.Linear(8, 16),  
            nn.ReLU(),
            nn.LayerNorm(16)
        )

        self.proj_lig = nn.Sequential(
            nn.Linear(17, 16),  
            nn.ReLU(),
            nn.LayerNorm(16)
        )

        self.proj_inter = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.LayerNorm(16)
        )

        self.type_embed = nn.Embedding(3, 4)  # 3种边类型，嵌入为4维

    def forward(self, graph):
        edge_types = graph.edata['edge_type']

        feats = torch.zeros(graph.number_of_edges(), 16, device=edge_types.device)

        # Process in batches for different types
        prot_mask = edge_types == 0
        if prot_mask.any():
            prot_feats = graph.edata['prot_edge_feat'][prot_mask]
            feats[prot_mask] = self.proj_prot(prot_feats)

        lig_mask = edge_types == 1
        if lig_mask.any():
            lig_feats = graph.edata['lig_edge_feat'][lig_mask]
            feats[lig_mask] = self.proj_lig(lig_feats)

        inter_mask = edge_types == 2
        if inter_mask.any():
            inter_feats = graph.edata['inter_dist'][inter_mask]
            feats[inter_mask] = self.proj_inter(inter_feats)

        type_embeds = self.type_embed(edge_types)

        return torch.cat([feats, type_embeds], dim=1)  # [num_edges, 20]

def create_homogeneous_interaction_graph(
        protein_pdb,
        ligand_mol,
        protein_cutoff=30.0,
        ligand_cutoff=15.0,
        inter_cutoff=12.0,
        verbose=True
):
    # Construct the protein residue map
    protein_g = protein_to_residue_graph(protein_pdb)

    # Construct the ligand atom map
    ligand_g = ligand_to_atom_graph(ligand_mol)

    g = dgl.DGLGraph()

    num_protein_nodes = protein_g.number_of_nodes()
    num_ligand_nodes = ligand_g.number_of_nodes()
    total_nodes = num_protein_nodes + num_ligand_nodes
    g.add_nodes(total_nodes)

    node_types = torch.zeros(total_nodes, dtype=torch.long)
    node_types[:num_protein_nodes] = 0
    node_types[num_protein_nodes:] = 1
    g.ndata['type'] = node_types

    g.ndata['res_feat'] = torch.zeros(total_nodes, protein_g.ndata['h'].shape[1])
    g.ndata['res_feat'][:num_protein_nodes] = protein_g.ndata['h']

    g.ndata['atom_feat'] = torch.zeros(total_nodes, ligand_g.ndata['h'].shape[1])
    g.ndata['atom_feat'][num_protein_nodes:] = ligand_g.ndata['h']

    g.ndata['x'] = torch.zeros(total_nodes, 3)
    g.ndata['x'][:num_protein_nodes] = protein_g.ndata['x']
    g.ndata['x'][num_protein_nodes:] = ligand_g.ndata['x']

    src, dst = protein_g.edges()
    g.add_edges(src, dst)

    protein_edge_feat = protein_g.edata.get('e', torch.zeros(protein_g.number_of_edges(), 1))
    g.edata['prot_edge_feat'] = torch.zeros(g.number_of_edges(), protein_edge_feat.shape[1])
    g.edata['prot_edge_feat'][:protein_g.number_of_edges()] = protein_edge_feat

    src, dst = ligand_g.edges()
    src = src + num_protein_nodes
    dst = dst + num_protein_nodes
    g.add_edges(src, dst)

    ligand_edge_feat = ligand_g.edata.get('e', torch.zeros(ligand_g.number_of_edges(), 1))
    ligand_edge_ids = torch.arange(protein_g.number_of_edges(), g.number_of_edges())
    g.edata['lig_edge_feat'] = torch.zeros(g.number_of_edges(), ligand_edge_feat.shape[1])
    g.edata['lig_edge_feat'][ligand_edge_ids] = ligand_edge_feat

    #Optimize using KDTree
    prot_coords = g.ndata['x'][:num_protein_nodes].numpy()
    lig_coords = g.ndata['x'][num_protein_nodes:].numpy()

    kd_tree = cKDTree(lig_coords)
    neighbors = kd_tree.query_ball_point(prot_coords, r=inter_cutoff)

    inter_src = []
    inter_dst = []
    inter_dists = []

    for i, lig_indices in enumerate(neighbors):
        if lig_indices:
            # Calculate the actual distance
            dists = np.linalg.norm(lig_coords[lig_indices] - prot_coords[i], axis=1)

            valid_mask = dists < inter_cutoff
            valid_lig_indices = np.array(lig_indices)[valid_mask]
            valid_dists = dists[valid_mask]

            inter_src.extend([i] * len(valid_lig_indices))
            inter_dst.extend(valid_lig_indices)
            inter_dists.extend(valid_dists)

    num_inter = len(inter_src)

    if num_inter > 0:
        protein_indices = torch.tensor(inter_src, dtype=torch.long)
        ligand_indices = torch.tensor(inter_dst, dtype=torch.long) + num_protein_nodes

        start_edge_id = g.number_of_edges()
        g.add_edges(protein_indices, ligand_indices)
        end_edge_id1 = g.number_of_edges()

        g.add_edges(ligand_indices, protein_indices)
        end_edge_id2 = g.number_of_edges()

        edge_types = torch.zeros(g.number_of_edges(), dtype=torch.long)
        edge_types[:protein_g.number_of_edges()] = 0  
        edge_types[protein_g.number_of_edges():ligand_edge_ids[-1] + 1] = 1  
        edge_types[start_edge_id:end_edge_id1] = 2  # Positive interaction
        edge_types[end_edge_id1:end_edge_id2] = 2  # Reverse interaction
        g.edata['edge_type'] = edge_types

        dists = torch.tensor(inter_dists, dtype=torch.float32)
        inter_edge_feat = torch.zeros(g.number_of_edges(), 1)
        inter_edge_feat[start_edge_id:end_edge_id1] = dists.unsqueeze(1)
        inter_edge_feat[end_edge_id1:end_edge_id2] = dists.unsqueeze(1)
        g.edata['inter_dist'] = inter_edge_feat
    else:
        edge_types = torch.zeros(g.number_of_edges(), dtype=torch.long)
        edge_types[:protein_g.number_of_edges()] = 0  
        edge_types[protein_g.number_of_edges():] = 1 
        g.edata['edge_type'] = edge_types
        g.edata['inter_dist'] = torch.zeros(g.number_of_edges(), 1)

    projector = UnifiedEdgeProjector()
    g.edata['unified_feat'] = projector(g)

    return g


HYDROPHOBICITY = {
    'ALA': 1.8, 'VAL': 4.2, 'ILE': 4.5, 'LEU': 3.8, 'PHE': 2.8,
    'CYS': 2.5, 'MET': 1.9, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5,
    'UNK': 0.0
}

VOLUME = {
    'ALA': 88.6, 'ARG': 173.4, 'ASN': 114.1, 'ASP': 111.1, 'CYS': 108.5,
    'GLN': 143.8, 'GLU': 138.4, 'GLY': 60.1, 'HIS': 153.2, 'ILE': 166.7,
    'LEU': 166.7, 'LYS': 168.6, 'MET': 162.9, 'PHE': 189.9, 'PRO': 112.7,
    'SER': 89.0, 'THR': 116.1, 'TRP': 227.8, 'TYR': 193.6, 'VAL': 140.0,
    'UNK': 100.0
}

CHARGE = {
    'ARG': 1, 'LYS': 1, 'HIS': 0.5,  
    'ASP': -1, 'GLU': -1,
    'ALA': 0, 'ASN': 0, 'CYS': 0, 'GLN': 0, 'GLY': 0, 'ILE': 0,
    'LEU': 0, 'MET': 0, 'PHE': 0, 'PRO': 0, 'SER': 0, 'THR': 0,
    'TRP': 0, 'TYR': 0, 'VAL': 0, 'UNK': 0
}


def protein_to_residue_graph(pdb_file_path, cutoff=10.0, verbose=True):
    """
    Constructing residue-level protein diagrams from PDB files
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file_path)
    model = structure[0]

    residue_features = []
    residue_positions = []
    residue_ids = []
    residue_secondary = [] 
    residue_names = [] 
    residue_full_info = []  

    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                   'THR', 'TRP', 'TYR', 'VAL', 'UNK']
    ss_types = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'C']  

    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != ' ':
                continue

            res_id = residue.get_id()
            res_name = residue.get_resname().strip().upper() 
            chain_id = chain.id
            
            try:
                ca_atom = residue['CA']
                ca_pos = ca_atom.get_coord()
            except KeyError:
                atoms = list(residue.get_atoms())
                if atoms:
                    coords = np.array([atom.get_coord() for atom in atoms])
                    ca_pos = np.mean(coords, axis=0)
                else:
                    continue 

            ss_type = 'C'
            if 'SSE' in residue.xtra:
                ss_type = residue.xtra['SSE']
            elif hasattr(residue, 'secondary_structure'):
                ss_type = residue.secondary_structure

            if res_name in amino_acids:
                res_feature = [1 if res_name == aa else 0 for aa in amino_acids]
            else:
                res_feature = [0] * (len(amino_acids) - 1) + [1] 

            residue_features.append(res_feature)
            residue_positions.append(ca_pos)
            residue_ids.append((chain_id, res_id[1]))
            residue_secondary.append(ss_type) 
            residue_names.append(res_name)

    num_residues = len(residue_features)
    if num_residues == 0:
        print(f"Warning: The protein {pdb_file_path} does not have valid residues")
        g = dgl.DGLGraph()
        g.add_nodes(1)
        g.ndata['h'] = torch.zeros(1, len(amino_acids)).float()
        g.ndata['x'] = torch.zeros(1, 3).float()
        return g

    residue_positions = np.array(residue_positions)

    g = dgl.DGLGraph()
    g.add_nodes(num_residues)

    g.ndata['h'] = torch.tensor(residue_features).float()
    g.ndata['x'] = torch.tensor(residue_positions).float()

    dist_matrix = np.linalg.norm(
        residue_positions[:, None] - residue_positions[None, :], axis=-1
    )

    # Add adjacent edges of the sequence
    seq_edges = []
    for i in range(num_residues - 1):
        if residue_ids[i][0] == residue_ids[i + 1][0] and \
                residue_ids[i][1] + 1 == residue_ids[i + 1][1]:
            seq_edges.append((i, i + 1))
            seq_edges.append((i + 1, i))

    # Using KDTree to Accelerate Spatial Neighboring Edge Calculation
    residue_positions_np = np.array(residue_positions)
    kd_tree = cKDTree(residue_positions_np)
    pairs = kd_tree.query_pairs(cutoff, output_type='ndarray')

    spatial_edges = []
    if len(pairs) > 0:
        for i, j in pairs:
            spatial_edges.append((int(i), int(j)))
            spatial_edges.append((int(j), int(i)))

    all_edges = seq_edges + spatial_edges

    num_edges = len(all_edges)

    if all_edges:
        src, dst = zip(*all_edges)
        g.add_edges(src, dst)
        # Distance feature
        edge_dists = dist_matrix[list(src), list(dst)].reshape(-1, 1)

        #Direction vector characteristics
        dir_vectors = residue_positions[list(dst)] - residue_positions[list(src)]
        norms = np.linalg.norm(dir_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1 
        unit_vectors = dir_vectors / norms

        #Secondary structure similarity feature
        ss_similarity = []
        for s, d in zip(src, dst):
            similarity = 1.0 if residue_secondary[s] == residue_secondary[d] else 0.0
            ss_similarity.append([similarity])
        ss_similarity = np.array(ss_similarity)

        # Sequence distance feature
        residue_seq_index = {}

        chain_groups = {}
        for i, (chain_id, res_id) in enumerate(residue_ids):
            if chain_id not in chain_groups:
                chain_groups[chain_id] = []
            chain_groups[chain_id].append((i, res_id))

        for chain_id, residues in chain_groups.items():
            residues.sort(key=lambda x: x[1])

            for seq_idx, (res_idx, res_id) in enumerate(residues):
                residue_seq_index[res_idx] = seq_idx

        seq_dists = []
        for s, d in zip(src, dst):
            if residue_ids[s][0] == residue_ids[d][0]: 
                seq_idx_s = residue_seq_index.get(s)
                seq_idx_d = residue_seq_index.get(d)

                if seq_idx_s is not None and seq_idx_d is not None:
                    seq_dist = abs(seq_idx_s - seq_idx_d)
                else:
                    seq_dist = abs(residue_ids[s][1] - residue_ids[d][1])
            else:
                seq_dist = 100 

            seq_dists.append([seq_dist])
        seq_dists = np.array(seq_dists)

        # Interactions of amino acid types
        aa_interaction = []
        for s, d in zip(src, dst):
            res_s = residue_names[s]
            res_d = residue_names[d]

            hydro_diff = abs(HYDROPHOBICITY[res_s] - HYDROPHOBICITY[res_d])
            hydro_complement = 1.0 - min(hydro_diff / 5.0, 1.0) 

            vol_sim = min(VOLUME[res_s], VOLUME[res_d]) / max(VOLUME[res_s], VOLUME[res_d])

            charge_complement = 0
            if (CHARGE[res_s] > 0 and CHARGE[res_d] < 0) or (CHARGE[res_s] < 0 and CHARGE[res_d] > 0):
                charge_complement = 1.0
            elif CHARGE[res_s] == 0 and CHARGE[res_d] == 0:
                charge_complement = 0.5
            else:
                charge_complement = 0.0

            hydrophobic = {'ALA', 'VAL', 'ILE', 'LEU', 'PHE', 'TRP', 'TYR', 'MET', 'CYS'}
            polar = {'SER', 'THR', 'ASN', 'GLN'}
            charged = {'ARG', 'LYS', 'HIS', 'ASP', 'GLU'}

            if res_s in hydrophobic and res_d in hydrophobic:
                class_sim = 1.0
            elif res_s in polar and res_d in polar:
                class_sim = 0.8
            elif res_s in charged and res_d in charged:
                class_sim = 0.6
            else:
                class_sim = 0.3

            interaction_score = (hydro_complement * 0.4 +
                                 vol_sim * 0.2 +
                                 charge_complement * 0.3 +
                                 class_sim * 0.1)

            aa_interaction.append([interaction_score])
        aa_interaction = np.array(aa_interaction)

        # angle feature
        angle_features = []
        for s, d in zip(src, dst):
            predecessors = [i for i in range(num_residues)
                            if (i, s) in all_edges and i != d]
            successors = [i for i in range(num_residues)
                          if (s, i) in all_edges and i != d]

            angles = []
            for p in predecessors:
                vec1 = residue_positions[s] - residue_positions[p]
                vec2 = residue_positions[d] - residue_positions[s]
                angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6))
                angles.append(angle)

            for s in successors:
                vec1 = residue_positions[s] - residue_positions[d]
                vec2 = residue_positions[s] - residue_positions[s]
                angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6))
                angles.append(angle)

            avg_angle = np.mean(angles) if angles else 0.0
            angle_features.append([avg_angle])
        angle_features = np.array(angle_features)

        edge_feats = np.concatenate([
            edge_dists,  # 1
            unit_vectors,  # 3
            ss_similarity,  # 1
            seq_dists,  # 1
            aa_interaction,  # 1
            angle_features  # 1
        ], axis=1)  #8
        g.edata['e'] = torch.tensor(edge_feats).float()

    return g

def ligand_to_atom_graph(mol):
    """Atomic-level graph construction"""
    g = dgl.DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    if mol.GetConformers():
        geom = mol.GetConformer().GetPositions()
    else:
        geom = np.zeros((num_atoms, 3))

    atom_symbols = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Other']
    node_feats = []
    for atom in mol.GetAtoms():
        atom_type = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols)
        degree = one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        implicit_valence = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        hybridization = one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                rdchem.HybridizationType.SP,
                rdchem.HybridizationType.SP2,
                rdchem.HybridizationType.SP3,
                rdchem.HybridizationType.SP3D,
                rdchem.HybridizationType.SP3D2
            ]
        )
        aromatic = [int(atom.GetIsAromatic())]
        features = atom_type + degree + implicit_valence + hybridization + aromatic
        node_feats.append(features)

    g.ndata['h'] = torch.tensor(node_feats).float()
    g.ndata['x'] = torch.tensor(geom).float()  # 3D坐标

    edge_index = []
    edge_attr = []
    bond_count = 0  

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_count += 1

        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx()
            if mol.GetBondBetweenAtoms(j, k) and i != k:
                vector1 = geom[j] - geom[i]
                vector2 = geom[k] - geom[i]
                angles_ijk.append(angle(vector1, vector2))
                areas_ijk.append(area_triangle(vector1, vector2))
                dists_ik.append(np.linalg.norm(geom[i] - geom[k]))

        angles_ijk = np.array(angles_ijk) if angles_ijk else np.array([0.])
        areas_ijk = np.array(areas_ijk) if areas_ijk else np.array([0.])
        dists_ik = np.array(dists_ik) if dists_ik else np.array([0.])

        dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
        dist_ij2 = cal_dist(geom[i], geom[j], ord=2)

        geom_feats = [
            angles_ijk.max() * 0.1,
            angles_ijk.sum() * 0.01,
            angles_ijk.mean() * 0.1,
            areas_ijk.max() * 0.1,
            areas_ijk.sum() * 0.01,
            areas_ijk.mean() * 0.1,
            dists_ik.max() * 0.1,
            dists_ik.sum() * 0.01,
            dists_ik.mean() * 0.1,
            dist_ij1 * 0.1,
            dist_ij2 * 0.1,
        ]

        bond_type = bond.GetBondType()
        bond_feats = [
            int(bond_type == rdchem.BondType.SINGLE),
            int(bond_type == rdchem.BondType.DOUBLE),
            int(bond_type == rdchem.BondType.TRIPLE),
            int(bond_type == rdchem.BondType.AROMATIC),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]

        total_feats = bond_feats + geom_feats

        edge_index.append((i, j))
        edge_index.append((j, i))
        edge_attr.append(total_feats)
        edge_attr.append(total_feats) 

    if edge_index:
        src, dst = zip(*edge_index)
        g.add_edges(src, dst)

        # Ensure that the number of features matches the number of edges.
        if len(edge_attr) == len(edge_index):
            g.edata['e'] = torch.tensor(edge_attr).float()
        else:
            print(f"The number of edge features ({len(edge_attr)}) does not match the number of edges ({len(edge_index)})!")

            edge_attr_fixed = [feat for feat in edge_attr[:len(edge_index)]]
            g.edata['e'] = torch.tensor(edge_attr_fixed).float()
    else:
        g.edata['e'] = torch.zeros(0, 17).float() 

    return g

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


def angle(vector1, vector2):
    cos_angle = vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8)
    angle = np.arccos(cos_angle)
    return angle

def area_triangle(vector1, vector2):
    trianglearea = 0.5 * np.linalg.norm( \
        np.cross(vector1, vector2))
    return trianglearea

def cal_dist(vertex1, vertex2, ord=2):
    return np.linalg.norm(vertex1 - vertex2, ord=ord)        
        
        
        

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def mae(y, f):
    return np.mean(np.abs(y - f))
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

