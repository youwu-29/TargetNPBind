import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.targetnpbind import TargetNPBind
from utils import *
from sklearn.model_selection import KFold
import torch.nn.functional as F
import random
from tqdm import tqdm

INTERACTION_GRAPHS_ROOT = "./data/interaction_graph"

SEED = 46

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    total_batches = len(train_loader)
    progress = tqdm(total=total_batches, desc=f'Training Epoch {epoch}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for batch_idx, data in enumerate(train_loader):
        progress.update(1)
        interaction_graphs = data.interaction_graphs
        data = data.to(device)
        data.interaction_graphs = interaction_graphs

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            progress.set_description(f'Training Epoch {epoch} [{batch_idx}/{total_batches}] Loss: {loss.item():.6f}')
    
    progress.close()

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
            
            if (batch_idx + 1) % 10 == 0:
                print(f"epoch {batch_idx+1}/{len(loader)}")
    
    labels_np = total_labels.numpy().flatten()
    preds_np = total_preds.numpy().flatten()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        weights_file = f'{save_dir}/model_weights_{phase}_fold{fold}.pth'
        torch.save(model.state_dict(), weights_file)

    return labels_np, preds_np

cuda_index = sys.argv[1] if len(sys.argv) > 1 else "0"
start_fold = 0
end_fold = 1

cuda_name = f"cuda:{cuda_index}"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 50
PATIENCE = 5  

loss_fn = nn.MSELoss()

results_dir = "./results/train"
os.makedirs(results_dir, exist_ok=True)

print("Loading datasets...")
train_val_data = TestbedDataset(root='data/processed', dataset='train',
                                interaction_graphs_root=INTERACTION_GRAPHS_ROOT)
print(f"Size of the training sets: {len(train_val_data)}")

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

fold_progress = tqdm(total=n_folds, desc='Progress of Cross-Validation')

fold_results = {
    'validation': {
        'rmse': [],
        'mse': [],
        'mae': [],
        'ci': [],
        'pearson': [],
        'spearman': []
    }
}
                  
for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_data)):
    
    if fold < start_fold or fold > end_fold:
        print(f"Skipping fold {fold + 1}")
        fold_progress.update(1)
        continue
    
    fold_progress.set_description(f'Processing fold {fold + 1}/{n_folds}')
    print(f"\n=== Fold {fold + 1}/{n_folds} ===")

    train_set = torch.utils.data.Subset(train_val_data, train_idx)
    val_set = torch.utils.data.Subset(train_val_data, val_idx)

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
    model = TargetNPBind().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float('inf')
    best_epoch = -1
    model_file = f'{results_dir}/model_fold{fold + 1}.pt'
    
    patience_counter = 0
    early_stop = False

    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)

        val_G, val_P = predicting(model, device, val_loader)
        val_mse = mse(val_G, val_P)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            torch.save(model.state_dict(), model_file)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Fold {fold + 1} - No improvement for {patience_counter} epochs (patience: {PATIENCE})')
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered at epoch {epoch} for fold {fold + 1}')
                early_stop = True

        if early_stop:
            print(f'Fold {fold + 1}: Training stopped early after {epoch} epochs')
            break

    model.load_state_dict(torch.load(model_file))
    print(f"Final evaluation on validation set for fold {fold + 1}...")
    G, P = predicting(model, device, val_loader)

    results = {
        'rmse': rmse(G, P),
        'mse': mse(G, P),
        'mae': mae(G, P),
        'ci': ci(G, P),
        'pearson': pearson(G, P),
        'spearman': spearman(G, P)
    }

    print(f"\nFold {fold + 1} VALIDATION Results:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  CI: {results['ci']:.4f}")
    print(f"  Pearson: {results['pearson']:.4f}")
    print(f"  Spearman: {results['spearman']:.4f}")

    for metric in ['rmse', 'mse', 'mae', 'ci', 'pearson', 'spearman']:
        fold_results['validation'][metric].append(results[metric])

    result_file = f'{results_dir}/results_fold{fold + 1}.csv'
    with open(result_file, 'w') as f:
        f.write("Fold,Validation_RMSE,Validation_MSE,Validation_MAE,Validation_CI,Validation_Pearson,Validation_Spearman\n")
        row = [
            fold + 1,
            results['rmse'], results['mse'], results['mae'], results['ci'],
            results['pearson'], results['spearman']
        ]
        f.write(','.join(map(str, row)) + '\n')
    
    fold_progress.update(1)

fold_progress.close()

print("\n=== Final Validation Results Across All Folds ===")
print(f"\nVALIDATION Results:")
for metric in ['rmse', 'mse', 'mae', 'ci', 'pearson', 'spearman']:
    values = fold_results['validation'][metric]
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

final_result_file = f'{results_dir}/final_results.csv'
with open(final_result_file, 'w') as f:
    f.write("Dataset,Metric,Mean,Std\n")
    for metric in ['rmse', 'mse', 'mae', 'ci', 'pearson', 'spearman']:
        values = fold_results['validation'][metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        f.write(f"validation,{metric},{mean_val},{std_val}\n")

print("\nTraining completed!")