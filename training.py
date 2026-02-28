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

INTERACTION_GRAPHS_ROOT = "/root/autodl-tmp/NP_complex_interaction_graph"

SEED = 43

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    # 添加进度条
    total_batches = len(train_loader)
    progress = tqdm(total=total_batches, desc=f'train Epoch {epoch}')
    
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
            progress.set_description(f'train Epoch {epoch} [{batch_idx}/{total_batches}] 损失: {loss.item():.6f}')
    
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
                print(f"batch {batch_idx+1}/{len(loader)}")
    
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

    return labels_np, preds_np

cuda_index = sys.argv[1] if len(sys.argv) > 1 else "0"

modeling = [GraphSAGENet][0]
model_st = modeling.__name__

cuda_name = f"cuda:{cuda_index}"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
print(f'Selected model: {model_st}')

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000
PATIENCE = 10  

loss_fn = nn.MSELoss()

print("Loading datasets...")

core_test_data = TestbedDataset(root='data/processed-pro-graph-top1', dataset='moltest',
                                interaction_graphs_root=INTERACTION_GRAPHS_ROOT)

core_test_loader = DataLoader(core_test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)


train_val_data = TestbedDataset(root='data/processed-pro-graph-top1', dataset='moltrain',
                                interaction_graphs_root=INTERACTION_GRAPHS_ROOT)

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

fold_progress = tqdm(total=n_folds, desc='Progress of cross-validation')

fold_results = {
    'core_test': {
        'rmse': [],
        'mse': [],
        'mae': [],
        'ci': [],
        'pearson': [],
        'spearman': []
    }
}

def mae(true, pred):
    return np.mean(np.abs(true - pred))

start_fold = 0
end_fold = 4
                    
for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_data)):
    
    if fold < start_fold or fold > end_fold:
        print(f"Skip the {fold + 1}th fold.")
        fold_progress.update(1)
        continue
    
    fold_progress.set_description(f'progress fold {fold + 1}/{n_folds}')
    print(f"\n=== Fold {fold + 1}/{n_folds} ===")

    train_set = torch.utils.data.Subset(train_val_data, train_idx)
    val_set = torch.utils.data.Subset(train_val_data, val_idx)

    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH_SIZE, shuffle=False)

    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_mse = float('inf')
    best_epoch = -1
    model_file = f'proegnngraphtop1_model_{model_st}_fold{fold + 1}.pt'
    best_val_dir = f'results/proegnngraphtop1-fold{fold+1}/best_val_data'
    os.makedirs(best_val_dir, exist_ok=True)

    patience_counter = 0
    early_stop = False

    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)

        val_G, val_P = predicting(model, device, val_loader)
        val_mse = mse(val_G, val_P)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            model_file = os.path.join(best_val_dir, f'proegnngraphtop1_model_{model_st}_fold{fold + 1}.pt')
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

    model_file = os.path.join(best_val_dir, f'proegnngraphtop1_model_{model_st}_fold{fold + 1}.pt')
    model.load_state_dict(torch.load(model_file))
    
    test_dir = f'results/proegnngraphtop1-fold{fold+1}/test_data'
    print(f"Evaluating fold {fold + 1} on core test set...")
    G, P = predicting(
        model, device, core_test_loader, 
        save_dir=test_dir,
        phase='test', fold=fold+1
    )

    results = {
        'rmse': rmse(G, P),
        'mse': mse(G, P),
        'mae': mae(G, P),
        'ci': ci(G, P),
        'pearson': pearson(G, P),
        'spearman': spearman(G, P)
    }

    print(f"\nFold {fold + 1} CORE_TEST Results:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  MSE: {results['mse']:.4f}")
    print(f"  MAE: {results['mae']:.4f}")
    print(f"  CI: {results['ci']:.4f}")
    print(f"  Pearson: {results['pearson']:.4f}")
    print(f"  Spearman: {results['spearman']:.4f}")

    for metric in ['rmse', 'mse', 'mae', 'ci', 'pearson', 'spearman']:
        fold_results['core_test'][metric].append(results[metric])

    result_file = f'proegnngraphtop1-results_{model_st}_fold{fold + 1}.csv'
    with open(result_file, 'w') as f:
        f.write(
            "Fold,Fold_RMSE,Fold_MSE,Fold_MAE,Fold_CI,Core_Test_RMSE,Core_Test_MSE,Core_Test_MAE,Core_Test_CI,Core_Test_Pearson,Core_Test_Spearman\n")
        row = [
            fold + 1,
            results['rmse'], results['mse'], results['mae'], results['ci'],
            results['rmse'], results['mse'], results['mae'], results['ci'],
            results['pearson'], results['spearman']
        ]
        f.write(','.join(map(str, row)) + '\n')
    fold_progress.update(1)

fold_progress.close()

print("\n=== Final Results Across All Folds ===")
for dataset in ['core_test']:
    print(f"\n{dataset.upper()} Results:")
    for metric in ['rmse', 'mse', 'mae', 'ci', 'pearson', 'spearman']:
        values = fold_results[dataset][metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

final_result_file = f'proegnngraphtop1-final_results_{model_st}.csv'
with open(final_result_file, 'w') as f:
    f.write("Dataset,Metric,Mean,Std\n")
    for dataset in ['core_test']:
        for metric in ['rmse', 'mse', 'mae', 'ci', 'pearson', 'spearman']:
            values = fold_results[dataset][metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{dataset},{metric},{mean_val},{std_val}\n")

print("\nTraining and evaluation completed!")
