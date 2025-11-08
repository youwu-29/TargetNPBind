import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_max_pool as gmp, global_mean_pool as gap
from .interaction_model import InteractionGraphProcessor
import dgl
from torch_geometric.nn import GCNConv
import random
import math
import numpy as np
    
    
class FeatureDecoupler(nn.Module):
    def __init__(self, base_dim, aug_dim, dropout_rate=0.2, case_data=None):
        super().__init__()
        
        self.dropout_rate = dropout_rate
        self.base_expander = nn.Linear(base_dim, aug_dim)
        self.gate = nn.Sequential(
            nn.Linear(aug_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.LayerNorm(64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.step_counter = 0
        self.print_interval = 1  
        self.case_data = case_data  

    def forward(self, base_feat, aug_feat):
        base_expanded = self.base_expander(base_feat)  # [batch, aug_dim]
        base_expanded = F.dropout(base_expanded, p=self.dropout_rate, training=self.training)  

        combined = torch.cat([base_expanded, aug_feat], dim=1)
        gate_val = self.gate(combined)

        if self.training:
            self.step_counter += 1
            if self.step_counter % self.print_interval == 0:
                with torch.no_grad():
                    batch_mean = gate_val.mean().item()
                    batch_min = gate_val.min().item()
                    batch_max = gate_val.max().item()
                    print(
                        f"[Gate Value] Steps: {self.step_counter} | Mean: {batch_mean:.4f} | Range: [{batch_min:.4f}, {batch_max:.4f}]")

        fused_feat = gate_val * base_expanded + (1 - gate_val) * aug_feat
        fused_feat = F.dropout(fused_feat, p=0.1, training=self.training)  
        return fused_feat

class FragmentAtomAttention(nn.Module):
    def __init__(self, in_channels, case_data=None):
        super().__init__()
        self.case_data = case_data
        self.frag_query = nn.Linear(in_channels, in_channels)
        self.atom_key = nn.Linear(in_channels, in_channels)
        self.atom_value = nn.Linear(in_channels, in_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.frag_query.weight)
        nn.init.zeros_(self.frag_query.bias)
        nn.init.xavier_uniform_(self.atom_key.weight)
        nn.init.zeros_(self.atom_key.bias)
        nn.init.xavier_uniform_(self.atom_value.weight)
        nn.init.zeros_(self.atom_value.bias)
    
    def forward(self, x, pharmacophore_groups, batch=None):
        updated_x = x.clone()
        all_updated_fragments = [] 
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        unique_batches = torch.unique(batch)
        frag_feat_list = []
        
        for b in unique_batches:
            b_idx = b.item()
            mol_atom_mask = (batch == b)
            mol_atom_indices = mol_atom_mask.nonzero(as_tuple=True)[0]
            
            if pharmacophore_groups is not None and b_idx in pharmacophore_groups:
                batch_groups = pharmacophore_groups[b_idx]
            else:
                batch_groups = {}
            
            fragment_features = []
            frag_id_to_index = {} 
            
            for idx, frag_id in enumerate(batch_groups.keys()):
                frag_id_to_index[frag_id] = idx
            
            for frag_id, atom_indices in batch_groups.items():
                frag_feat = torch.mean(x[atom_indices], dim=0)
                fragment_features.append(frag_feat)
            
            if fragment_features:
                fragment_features = torch.stack(fragment_features)  
            else:
                fragment_features = torch.empty(0, x.size(1), device=x.device)
            
            # Fragment → Atomic Attention
            if fragment_features.numel() > 0:  
                Q = self.frag_query(fragment_features)  # [F, d]
                K = self.atom_key(x[mol_atom_indices])  # [n_mol, d]
                V = self.atom_value(x[mol_atom_indices])  # [n_mol, d]
                
                attn_scores = torch.mm(Q, K.t())  # [F, n_mol]
                attn_weights = F.softmax(attn_scores, dim=1)  # [F, n_mol]
                
                attn_weights_t = attn_weights.t()  # [n_mol, F]
                attn_weights_expanded = attn_weights_t.unsqueeze(-1)  # [n_mol, F, 1]
                V_expanded = V.unsqueeze(1)  # [n_mol, 1, d]
                
                weighted_V = attn_weights_expanded * V_expanded  # [n_mol, F, d]
                atom_updates = torch.sum(weighted_V, dim=1)  # [n_mol, d]
                updated_x[mol_atom_indices] += atom_updates
                
                updated_fragments = []
                for frag_id, atom_indices in batch_groups.items():
                    internal_atoms = updated_x[atom_indices]
                    internal_mean = torch.mean(internal_atoms, dim=0)

                    frag_idx = frag_id_to_index[frag_id]
                    frag_feat = fragment_features[frag_idx]
                    
                    updated_fragments.append(internal_mean)
                
                if updated_fragments:
                    updated_fragments = torch.stack(updated_fragments)  # [F, d]
                    frag_feat_list.append(updated_fragments)
                    
        if frag_feat_list:
            all_updated_fragments = torch.cat(frag_feat_list, dim=0)
        else:
            all_updated_fragments = torch.empty(0, x.size(1), device=x.device)
        
        return updated_x, all_updated_fragments


class PharmaEnhancedSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max', case_data=None):
        super().__init__()
        self.case_data = case_data
        self.sage_main = SAGEConv(in_channels, out_channels, aggr=aggr)
        self.pharma_attn = FragmentAtomAttention(out_channels, case_data=case_data)
        self.res_gate = nn.Sequential(
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, pharmacophore_groups, batch=None):
        main_out = self.sage_main(x, edge_index)
        
        if pharmacophore_groups:  
            enhanced_features, _ = self.pharma_attn(
                main_out, 
                pharmacophore_groups,
                batch=batch
            )
            
            gate = self.res_gate(enhanced_features)  # [n, 1]
            output = gate * enhanced_features + (1 - gate) * main_out
            return output
        else:
            return main_out

class TargetNPBind(nn.Module):
    def __init__(self, n_output=1, num_features_xd=79, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(TargetNPBind, self).__init__()

        self.n_output = n_output
        total_atom_features = num_features_xd
        
        self.sage1 = PharmaEnhancedSAGE(
            in_channels=num_features_xd,
            out_channels=256,
            aggr='max',
        )
        
        self.sage2 = SAGEConv(256, 512, aggr='max')
        self.sage3 = SAGEConv(512, output_dim, aggr='max')

        self.fc_g1 = nn.Linear(output_dim * 2, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(
            in_channels=embed_dim,  
            out_channels=n_filters,  
            kernel_size=8,
            padding=4 
        )
        self.conv_xt_2 = nn.Conv1d(
            in_channels=n_filters,  
            out_channels=n_filters,  
            kernel_size=8,
            padding=4  
        )
        self.conv_xt_3 = nn.Conv1d(
            in_channels=n_filters,  
            out_channels=n_filters,  
            kernel_size=8,
            padding=4  
        )
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        conv_out_features = 375  

        self.fc1_xt = nn.Linear(n_filters * conv_out_features, output_dim)
        self.cross_attn = CrossAttentionModule(input_dim=output_dim)

        self.decoupler = FeatureDecoupler(
            base_dim=256,  # xc
            aug_dim=256,  # interaction_rep
            case_data=self.case_data
        )
        
        self.ligand_dropout = nn.Dropout(0.2)
        self.dropout_conv = nn.Dropout(0.2) 
        self.dropout_fc = nn.Dropout(0.2)    

        self.fusion_dropout = nn.Dropout(0.1)
        self.pred_dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, n_output)

        self.interaction_processor = InteractionGraphProcessor(config=None, case_data=self.case_data)

        self.interaction_rep = None  
        self.xc = None  
           
    def forward(self, data):
        if not self.training:
            self.clear_case_data()
        if hasattr(self.interaction_processor.fusion_module, 'batch_common_feats'):
            self.interaction_processor.fusion_module.batch_common_feats = None

        device = next(self.parameters()).device

        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        data = data.to(device)
        
        pharmacophore_dict = {}
        if hasattr(data, 'pharmacophore_groups'):
            try:
                pharmacophore_groups_list = data.pharmacophore_groups
                
                for batch_idx, groups in enumerate(pharmacophore_groups_list):
                    batch_dict = {}
                    for group in groups:
                        if isinstance(group, list) and len(group) == 2:
                            frag_id = group[0]
                            atom_indices = group[1]
                            batch_dict[frag_id] = atom_indices
                    pharmacophore_dict[batch_idx] = batch_dict
                    
            except Exception as e:
                print(f"\nFailed to process the information of drug effect groupings: {str(e)}")
                import traceback
                traceback.print_exc()
                raise RuntimeError("Unable to process the information on drug effect groupings")
        else:
            print("Warning: No information on drug efficacy groups is available.")

        batch = data.batch 
        x = self.sage1(x, edge_index, pharmacophore_groups=pharmacophore_dict, batch=batch)
        #x = self.ligand_dropout(x)
        x = self.sage2(x, edge_index)
        #x = self.ligand_dropout(x)
        x = self.sage3(x, edge_index)

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.fc_g1(x))
        #x = self.ligand_dropout(x)
        x = self.fc_g2(x)
        
        embedded_xt = self.embedding_xt(target)  # (batch_size, seq_len, embed_dim)
        embedded_xt = embedded_xt.permute(0, 2, 1)  # (batch_size, embed_dim, seq_len)
        
        conv1 = F.relu(self.conv_xt_1(embedded_xt))  # (batch_size, n_filters, seq_len)
        #conv1 = self.dropout_conv(conv1)
        pool1 = self.pool(conv1)  # (batch_size, n_filters, seq_len/2)

        conv2 = F.relu(self.conv_xt_2(pool1))  # (batch_size, n_filters, seq_len/2)
        #conv2 = self.dropout_conv(conv2)
        pool2 = self.pool(conv2)  # (batch_size, n_filters, seq_len/4)

        conv3 = F.relu(self.conv_xt_3(pool2))  # (batch_size, n_filters, seq_len/4)
        #conv3 = self.dropout_conv(conv3)
        pool3 = self.pool(conv3)  # (batch_size, n_filters, seq_len/8)

        conv_xt = pool3.view(pool3.size(0), -1)  # (batch_size, n_filters * (seq_len/8))
        #conv_xt = self.dropout_fc(conv_xt)

        xt_conv = self.fc1_xt(conv_xt)
        xc = torch.cat([x, xt_conv], dim=1) 
        self.xc = xc

        device = next(self.parameters()).device
        interaction_reps = []
        contrast_losses = [] 

        if not hasattr(data, 'interaction_graphs'):
            has_interaction = False
        else:
            has_interaction = True

        if hasattr(data, 'interaction_graphs'):
            ig_data = data.interaction_graphs

            if isinstance(ig_data, list):
                non_empty = sum(1 for g in ig_data if g is not None and len(g) > 0)
        else:
            print("The input data does not have the "interaction_graphs" attribute.")

        if not has_interaction or len(data.interaction_graphs) == 0:
            print("The "interaction_graphs" does not exist or is empty.")
            batch_size = xc.size(0) if xc.dim() > 0 and xc.size(0) > 0 else 1
            print(f"Create a zero tensor.")
            device = xc.device if torch.is_tensor(xc) else torch.device('cpu')
            interaction_rep = torch.zeros(batch_size, 256, device=device)
            contrast_loss = torch.tensor(0.0, device=device) 
        else:
            for sample_graphs in data.interaction_graphs:
                sample_graphs = [g.to(device) for g in sample_graphs]

                rep, sample_contrast_loss = self.interaction_processor(sample_graphs)
                interaction_reps.append(rep)
                contrast_losses.append(sample_contrast_loss)

            interaction_rep = torch.stack(interaction_reps, dim=0)
            contrast_loss = torch.mean(torch.stack(contrast_losses))

            if interaction_rep.dim() == 3:
                interaction_rep = interaction_rep.squeeze(1)  

        self.interaction_rep = interaction_rep  
        combined = self.decoupler(xc, interaction_rep)  # [batch, 256]
        
        combined = self.fusion_dropout(combined)
        self.final_features = combined  
        #combined = self.pred_dropout(combined)
        pred_head_input = F.relu(self.fc1(combined))
        pred_head_input = self.dropout_layer(pred_head_input)
        pred_head_input = F.relu(self.fc2(pred_head_input))
        pred_head_input = self.dropout_layer(pred_head_input)
        out = self.out(pred_head_input)

        return out

