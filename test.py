import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from models.graphsage import GraphSAGENet
from models.graphtransformer import TransformerConvNet
from utils import *
from sklearn.model_selection import KFold
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch_geometric.loader import DataLoader

INTERACTION_GRAPHS_ROOT = "/root/autodl-tmp/NP_complex_interaction_graph"

SEED = 43

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def mae(true, pred):
    return np.mean(np.abs(true - pred))

def predicting(model, device, loader, save_dir=None, phase='test', fold=0):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print(f'Make prediction for {len(loader.dataset)} samples...')
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            
            output = model(data)
            
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            
    
    labels_np = total_labels.numpy().flatten()
    preds_np = total_preds.numpy().flatten()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        csv_file = f'{save_dir}/predictions_{phase}_fold{fold}.csv'
        df = pd.DataFrame({
            'true_affinity': labels_np,
            'pred_affinity': preds_np,
            'error': np.abs(labels_np - preds_np)
        })
        df.to_csv(csv_file, index=False)
        
        weights_file = f'{save_dir}/model_weights_{phase}_fold{fold}.pth'
        torch.save(model.state_dict(), weights_file)

    return labels_np, preds_np

def main():
    model_index = 0
    cuda_index = sys.argv[1] if len(sys.argv) > 1 else "0"
    modeling = [GraphSAGENet][model_index]
    model_st = modeling.__name__
    
    cuda_name = f"cuda:{cuda_index}"
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'Selected model: {model_st} (index: {model_index})')
    
    MODEL_PATH = "/root/autodl-tmp/GraphDTA-master/results/randomgraphtop1-fold5/test_data/model_weights_test_fold5.pth"
    
    model = modeling().to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        print(f'Successfully loaded model from {MODEL_PATH} (strict=False)')
    else:
        print(f'Error: Model file not found at {MODEL_PATH}')
        return
    
    print("Loading datasets...")
    
    TEST_BATCH_SIZE = 64
    
    core_test_data = TestbedDataset(root='data/processed-random-graph-top1', dataset='moltest',
                                    interaction_graphs_root=INTERACTION_GRAPHS_ROOT)
    core_test_loader = DataLoader(core_test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    test_dir = 'results/single_evaluation/randomtest_data'
    os.makedirs(test_dir, exist_ok=True)
    
    G, P = predicting(
        model, device, core_test_loader, 
        save_dir=test_dir,
        phase='test', fold=0
    )
    
    results = {
        'rmse': rmse(G, P),
        'mse': mse(G, P),
        'mae': mae(G, P),
        'ci': ci(G, P),
        'pearson': pearson(G, P),
        'spearman': spearman(G, P)
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f'Model: {model_st} (index: {model_index})')
    print(f'Model path: {MODEL_PATH}')
    print(f'Test samples: {len(core_test_data)}')
    print("-"*60)
    print(f'RMSE: {results["rmse"]:.4f}')
    print(f'MSE: {results["mse"]:.4f}')
    print(f'MAE: {results["mae"]:.4f}')
    print(f'CI: {results["ci"]:.4f}')
    print(f'Pearson: {results["pearson"]:.4f}')
    print(f'Spearman: {results["spearman"]:.4f}')
    print("="*60)
    
    result_file = f'randomgraphtop1-evaluation_{model_st}.csv'
    with open(result_file, 'w') as f:
        f.write("Model,Index,Test_Samples,RMSE,MSE,MAE,CI,Pearson,Spearman,Model_Path\n")
        row = [
            model_st,
            model_index,
            len(core_test_data),
            results['rmse'], 
            results['mse'], 
            results['mae'], 
            results['ci'],
            results['pearson'], 
            results['spearman'],
            MODEL_PATH
        ]
        f.write(','.join(map(str, row)) + '\n')

if __name__ == '__main__':
    main()
