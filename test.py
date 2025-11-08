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
TEST_BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "./results/test"
os.makedirs(RESULTS_DIR, exist_ok=True)

def test_core_set(model_path, model_type, fold_num):

    print("\nLoading the core test set...")
    core_test_data = TestbedDataset(
        root='data/processed', 
        dataset='test',
        interaction_graphs_root=INTERACTION_GRAPHS_ROOT
    )
    core_test_loader = DataLoader(core_test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    print(f"The size of the core test set: {len(core_test_data)}")

    model = TargetNPBind().to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"The TargetNPBind model (fold {fold_num}) has been loaded.")

    print("\nStart the test...")
    total_preds = []
    total_labels = []
    protein_ids = []
    
    with torch.no_grad():
        for data in tqdm(core_test_loader, desc="test progress"):
            data = data.to(device)
            output = model(data)
            total_preds.extend(output.cpu().numpy().flatten())
            total_labels.extend(data.y.cpu().numpy().flatten())
            protein_ids.extend(data.protein_seq)
    
    y_true = np.array(total_labels, dtype=np.float32)
    y_pred = np.array(total_preds, dtype=np.float32)
    
    # 计算各项指标
    metrics = {
        'RMSE': rmse(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),
        'CI': ci(y_true, y_pred),
        'Pearson': pearson(y_true, y_pred),
        'Spearman': spearman(y_true, y_pred)
    }

    print("\n=== Test set results ===")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    result_file = os.path.join(RESULTS_DIR, f"test_results_fold{fold_num}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Model: TargetNPBind (Fold {fold_num})\n")
        for name, val in metrics.items():
            f.write(f"{name}: {val:.4f}\n")
    print(f"\nThe indicator results have been saved to {result_file}")
    
    detail_file = os.path.join(RESULTS_DIR, fresults_fold{fold_num}.csv")
    df = pd.DataFrame({
        'protein_id': protein_ids,
        'true_value': total_labels,
        'pred_value': total_preds,
        'error': np.abs(y_true - y_pred)  
    })
    df.to_csv(detail_file, index=False)
    print(f"The detailed prediction results have been saved to {detail_file}")

if __name__ == "__main__":
    MODEL_TYPE = "TargetNPBind"
    FOLD_NUM = 1
    MODEL_PATH = f"./results/train/model_weights_test_fold{FOLD_NUM}.pth"                      
    test_core_set(MODEL_PATH, MODEL_TYPE, FOLD_NUM)