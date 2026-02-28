import torch.nn as nn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl import mean_nodes
import numpy as np


class EGNNConv(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0, dropout_rate=0.2, case_data=None):
        super().__init__()
        self.edge_feat_size = edge_feat_size
        self.dropout_rate = dropout_rate
        self.case_data = case_data

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),  
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, out_size)
        )
        
    def forward(self, feat, coords, edge_index, edge_attr=None):
        row, col = edge_index
        num_nodes = feat.size(0)
        
        rel_coords = coords[row] - coords[col]
        squared_distance = torch.sum(rel_coords ** 2, dim=1, keepdim=True)
        
        message_inputs = [feat[row], feat[col], squared_distance]
        
        if self.edge_feat_size > 0 and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            message_inputs.append(edge_attr)
        
        message_input = torch.cat(message_inputs, dim=-1)
        
        messages = self.edge_mlp(message_input)
        
        coord_weights = self.coord_mlp(messages)
        
        coord_updates = rel_coords * coord_weights
        
        coord_agg = torch.zeros_like(coords)
        coord_agg.index_add_(0, row, coord_updates)
        
        deg = torch.zeros(coords.size(0), device=coords.device)
        deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg = deg.clamp(min=1).unsqueeze(-1)
        
        new_coords = coords + coord_agg / deg
        
        agg_messages = torch.zeros(num_nodes, messages.size(-1), device=feat.device)
        agg_messages.index_add_(0, row, messages)
        
        node_input = torch.cat([feat, agg_messages], dim=-1)
        new_feat = self.node_mlp(node_input)
        
        if not self.training and self.case_data is not None:
            messages_detached = messages.detach().cpu()
            edge_types_detached = edge_attr[:, 0].detach().cpu() if edge_attr is not None else torch.zeros_like(row)
            atom_coords_detached = new_coords.detach().cpu()
            
            self.case_data['case2']['edge_messages'].append(messages_detached)
            self.case_data['case2']['edge_types'].append(edge_types_detached)
            self.case_data['case2']['atom_coords'].append(atom_coords_detached)
        
        return new_feat, new_coords


class GeometricGNNBase(nn.Module):
    def __init__(self, in_feats, out_feats, gnn_type="egnn", edge_feat_size=0, dropout_rate=0.2, case_data=None):
        super().__init__()
        self.gnn_type = gnn_type
        self.edge_feat_size = edge_feat_size
        self.dropout_rate = dropout_rate
        self.case_data = case_data

        self.gnn = EGNNConv(
            in_size=in_feats,
            hidden_size=256,
            out_size=out_feats,
            edge_feat_size=edge_feat_size,
            dropout_rate=dropout_rate,
            case_data=case_data
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

        edge_index = torch.stack(graph.edges(), dim=0)

        new_feat, new_coords = self.gnn(feat, coords, edge_index, edge_attr)

        graph.ndata['x'] = new_coords

        projected_feat = self.feat_projector(feat)
        gate = self.res_gate(new_feat)
        output = F.leaky_relu(gate * new_feat + (1 - gate) * projected_feat, 0.2)
        output = self.output_norm(output)
        output = self.feat_dropout(output)
        
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
    def __init__(self, config, case_data=None):
        super().__init__()
        self.config = config
        self.case_data = case_data
        
        self.debug_mode = False
        self.hidden_dim = 256
        self.dropout_rate = 0.2
        
        if self.debug_mode:
            self.simple_gnn = dgl.nn.GraphConv(53, self.hidden_dim)  
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.num_layers = 1
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
                        case_data=case_data
                    )
                )
            
            self.pool = dgl.nn.AvgPooling()
            
            self.fusion_strategy = "common"
            if self.fusion_strategy == "common":
                self.fusion_module = CommonFeatureFusion(
                    feat_dim=256,
                    temperature=0.07,
                    print_interval=200,
                    dropout_rate=self.dropout_rate
                )

            self.enhancer = nn.Sequential(
                nn.Linear(256, 128),
                nn.LeakyReLU(0.1),
                nn.LayerNorm(128),
                nn.Linear(128, 256)
            )

        self.dropout = nn.Dropout(self.dropout_rate)
        
        if self.debug_mode:
            self.fusion_module = type('DummyFusionModule', (), {
                'batch_common_feats': None,
                'print_interval': 200,
                'step_counter': 0
            })()

    def forward(self, interaction_graphs):
        
        if not interaction_graphs:
            device = next(self.parameters()).device
            return torch.zeros(1, 256, device=device)

        graph = interaction_graphs[0]
        device = next(self.parameters()).device
        graph = graph.to(device)

        if 'res_feat' in graph.ndata and 'atom_feat' in graph.ndata:
            res_feat = graph.ndata['res_feat']
            atom_feat = graph.ndata['atom_feat']
            node_feat = torch.cat([res_feat, atom_feat], dim=1)
            
            if torch.isnan(node_feat).any() or torch.isinf(node_feat).any():
                node_feat = torch.zeros_like(node_feat)
        else:
            node_feat = torch.zeros(graph.num_nodes(), 53, device=device)

        if self.debug_mode:

            graph_with_self_loop = dgl.add_self_loop(graph)

            node_features = self.simple_gnn(graph_with_self_loop, node_feat)  
            node_features = self.activation(node_features)     
            node_features = self.dropout(node_features)       

            graph.ndata['h'] = node_features
            graph_rep = dgl.mean_nodes(graph, 'h')

            if torch.isnan(graph_rep).any() or torch.isinf(graph_rep).any():
                graph_rep = torch.zeros(1, 256, device=device)

            return graph_rep
            
        else:
            if 'x' in graph.ndata:
                coords = graph.ndata['x']
            else:
                coords = torch.zeros(graph.num_nodes(), 3, device=device)
                graph.ndata['x'] = coords

            if 'unified_feat' in graph.edata:
                edge_attr = graph.edata['unified_feat']
                if edge_attr.size(1) != 20:
                    num_edges = graph.num_edges()
                    edge_attr = torch.zeros(num_edges, 20, device=device)
            else:
                num_edges = graph.num_edges()
                edge_attr = torch.zeros(num_edges, 20, device=device)

            for i, gnn_layer in enumerate(self.gnn_layers):
                
                node_feat, updated_coords = gnn_layer(
                    graph,
                    node_feat,
                    coords=coords,
                    edge_attr=edge_attr
                )
                
                if torch.isnan(node_feat).any():
                    return torch.zeros(1, 256, device=device)
                
                coords = updated_coords
                graph.ndata['x'] = coords
                
                if i < self.num_layers - 1:
                    node_feat = F.leaky_relu(node_feat, 0.1)
                    node_feat = self.dropout(node_feat)

            graph_rep = self.pool(graph, node_feat)

            conf_reps = graph_rep.unsqueeze(0)
            fused_rep, contrast_loss = self.fusion_module(conf_reps)

            enhanced_rep = self.enhancer(fused_rep)
            return fused_rep


