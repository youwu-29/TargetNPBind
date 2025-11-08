import torch.nn as nn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl import mean_nodes
import numpy as np


class EGNNConv(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, dropout_rate=0.2):
        super().__init__()
        self.edge_feat_size = edge_feat_size
        self.dropout_rate = dropout_rate

        self.phi_e = nn.Sequential(
            nn.Linear(in_size * 2 + 1 + edge_feat_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

        self.phi_h = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            nn.SiLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, feat, coords, edge_index, edge_attr=None):
        row, col = edge_index
        num_nodes = feat.size(0) 
        dtype = feat.dtype
        coords = coords.to(dtype)
        rel_coords = coords[row] - coords[col]
        dist_sq = torch.sum(rel_coords ** 2, dim=1, keepdim=True).to(dtype)

        msg_input = [feat[row], feat[col], dist_sq]
        if self.edge_feat_size > 0 and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            edge_attr = edge_attr.to(dtype)
            msg_input.append(edge_attr)

        msg_input = torch.cat(msg_input, dim=1)
        messages = self.phi_e(msg_input)
        coord_weights = self.phi_x(messages)

        coord_update = rel_coords * coord_weights
        coord_agg = torch.zeros_like(coords, dtype=dtype)
        coord_agg.index_add_(0, row, coord_update.to(dtype))
        deg = torch.zeros(coords.size(0), device=coords.device, dtype=torch.float32)
        deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg = deg.clamp(min=1).unsqueeze(-1)  
        deg = deg.to(dtype)  

        new_coords = coords + coord_agg / deg
        agg_msgs = torch.zeros(num_nodes, messages.size(1), dtype=dtype, device=feat.device)
        agg_msgs.index_add_(0, row, messages)

        node_input = torch.cat([feat, agg_msgs], dim=1)
        new_feat = self.phi_h(node_input)
            
        return new_feat, new_coords

class GeometricGNNBase(nn.Module):
    def __init__(self, in_feats, out_feats, gnn_type="egnn", edge_feat_size=0, dropout_rate=0.2):
        super().__init__()
        self.gnn_type = gnn_type
        self.edge_feat_size = edge_feat_size
        self.dropout_rate = dropout_rate

        if gnn_type == "egnn":
            self.gnn = EGNNConv(in_size=in_feats,
                                hidden_size=256,
                                out_size=out_feats,
                                edge_feat_size=edge_feat_size,
                                dropout_rate=dropout_rate,
                                )

        self.res_gate = nn.Sequential(
            nn.Linear(out_feats, 1),
            nn.Sigmoid()
        )
        self.output_norm = nn.LayerNorm(out_feats)

        self.feat_projector = nn.Linear(in_feats, out_feats)
        self.feat_dropout = nn.Dropout(dropout_rate)

    def forward(self, graph, feat, coords=None, edge_weight=None, edge_attr=None):
        if coords is None:
            if 'x' in graph.ndata:
                coords = graph.ndata['x'].detach()
            else:
                coords = torch.zeros(feat.size(0), 3, device=feat.device)

        edge_index = graph.edges()
        if self.gnn_type in ["egnn"]:
            if self.gnn_type == "egnn":
                edge_index = torch.stack(graph.edges(), dim=0)
                new_feat, new_coords = self.gnn(feat, coords, edge_index, edge_attr)

                graph.ndata['x'] = new_coords
            else:
                new_feat = self.gnn(feat, coords, edge_index, edge_attr=edge_attr)
                new_coords = coords 

        projected_feat = self.feat_projector(feat)
        gate = self.res_gate(new_feat)
        output = F.leaky_relu(gate * new_feat + (1 - gate) * projected_feat, 0.2)
        output = self.output_norm(output)
        output =  self.feat_dropout(output)
        
        return output, new_coords


class CommonFeatureFusion(nn.Module):
    def __init__(self, feat_dim, temperature=0.07, dropout_rate=0.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate  

        self.extractor = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.LayerNorm(feat_dim)
        )

        self.batch_common_feats = None

    def forward(self, conf_reps):
        mean_rep = torch.mean(conf_reps, dim=0)
        common_feat = self.extractor(mean_rep)
        common_feat = conf_reps + common_feat
        common_feat = F.dropout(common_feat, p=self.dropout_rate, training=self.training)

        if self.batch_common_feats is None:
            self.batch_common_feats = common_feat.unsqueeze(0).detach()
        else:
            self.batch_common_feats = torch.cat([
                self.batch_common_feats,
                common_feat.unsqueeze(0).detach()
            ], dim=0)

        common_feat = common_feat.squeeze(0) 
        return common_feat

class InteractionGraphProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = 1 
        self.hidden_dim = 256 
        self.dropout_rate = 0.2

        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            in_feats = 53 if i == 0 else self.hidden_dim
            self.gnn_layers.append(
                GeometricGNNBase(
                    in_feats=in_feats,
                    out_feats=self.hidden_dim,
                    gnn_type="egnn",
                    edge_feat_size=20,
                    dropout_rate=self.dropout_rate,
                )
            )
        self.pool = dgl.nn.AvgPooling()
        self.fusion_module = CommonFeatureFusion(
            feat_dim=256,
            temperature=0.07,
            dropout_rate=self.dropout_rate
        )

        self.enhancer = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(128),
            #nn.Dropout(0.2),  
            nn.Linear(128, 256)
        )

        self.dropout = nn.Dropout(self.dropout_rate) 


    def forward(self, interaction_graphs):
        """Process a group of interaction diagrams (select the first conformation only)"""

        if not interaction_graphs:
            device = next(self.parameters()).device
            return torch.zeros(1, 256, device=device), torch.tensor(0.0, device=device)

        graph = interaction_graphs[0]
        device = next(self.parameters()).device
        graph = graph.to(device)

        if 'res_feat' in graph.ndata and 'atom_feat' in graph.ndata:
            res_feat = graph.ndata['res_feat']
            atom_feat = graph.ndata['atom_feat']
            node_feat = torch.cat([res_feat, atom_feat], dim=1)
        else:
            print("Warning: Insufficient node features. Use zero tensor as a substitute.")
            node_feat = torch.zeros(graph.num_nodes(), 53, device=device)

        coords = graph.ndata['x']
        edge_attr = graph.edata['unified_feat']
            
        for i, gnn_layer in enumerate(self.gnn_layers):
            node_feat, updated_coords = gnn_layer(
                graph,
                node_feat,
                coords=coords, 
                edge_attr=edge_attr
            )

            coords = updated_coords
            graph.ndata['x'] = coords

            if i < self.num_layers - 1:
                node_feat = F.leaky_relu(node_feat, 0.1)
                node_feat = self.dropout(node_feat)  

        graph_rep = self.pool(graph, node_feat)  # (1, 256)
        conf_reps = graph_rep.unsqueeze(0)  # (1, 1, 256)
        fused_rep, contrast_loss = self.fusion_module(conf_reps)
        #fused_rep = self.dropout(fused_rep)
        enhanced_rep = self.enhancer(fused_rep)

        return fused_rep, contrast_loss