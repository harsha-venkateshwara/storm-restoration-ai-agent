"""
Storm Restoration Prioritization AI
Phase 2: Multi-Task LSTM Model Training

Architecture:
  - Shared LSTM encoder (2 layers, 128 hidden units)
  - Head A: Binary classification P(outage) with Focal Loss
  - Head B: Quantile regression for P50/P90 customers out
  - County embedding layer for spatial context

Training:
  - Train: 2015-2020 | Val: 2021 | Test: 2022
  - Early stopping on val PR-AUC
  - Saves best model checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, brier_score_loss, mean_absolute_error)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE     = Path(r'C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai')
FEATURES = BASE / 'data' / 'features'
OUTPUTS  = BASE / 'outputs'
MODELS   = OUTPUTS / 'models'
PLOTS    = OUTPUTS / 'plots'
MODELS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

#Feature columns
FEATURE_COLS = [
    'wind','winter','flood','tropical','other',
    'any_storm','storm_severity',
    'wind_sum_6h','wind_sum_12h','wind_sum_24h',
    'any_storm_sum_6h','any_storm_sum_12h','any_storm_sum_24h',
    'hour_sin','hour_cos','month_sin','month_cos',
    'is_weekend','quarter',
    'lag_1h_peak','lag_24h_peak','lag_168h_peak',
    'lag_24h_label','lag_168h_label',
    'county_fragility','null_count',
    'county_p90','county_mean_baseline'
]
N_FEATURES = len(FEATURE_COLS)
SEQ_LEN    = 24  # 24-hour lookback window

# Dataset
class OutageDataset(Dataset):
    """
    Wraps flat feature rows into (seq_len x n_features) sequences.
    Groups by FIPS and creates sliding windows.
    """
    def __init__(self, df, seq_len=SEQ_LEN, max_counties=500, seed=42):
        rng = np.random.default_rng(seed)
        counties = df['fips'].unique()
        if len(counties) > max_counties:
            counties = rng.choice(counties, size=max_counties, replace=False)

        X_list, y_cls_list, y_reg_list = [], [], []

        for fips in counties:
            sub = df[df['fips'] == fips].sort_values('hour').reset_index(drop=True)
            if len(sub) < seq_len + 1:
                continue
            X_arr   = sub[FEATURE_COLS].values.astype(np.float32)
            y_cls   = sub['outage_label'].values.astype(np.float32)
            y_reg   = sub['log_peak_customers'].values.astype(np.float32)

            for i in range(seq_len, len(sub)):
                X_list.append(X_arr[i-seq_len:i])
                y_cls_list.append(y_cls[i])
                y_reg_list.append(y_reg[i])

        self.X     = torch.tensor(np.array(X_list),     dtype=torch.float32)
        self.y_cls = torch.tensor(np.array(y_cls_list), dtype=torch.float32)
        self.y_reg = torch.tensor(np.array(y_reg_list), dtype=torch.float32)
        print(f"  Dataset: {len(self.X):,} sequences | "
              f"outage rate: {self.y_cls.mean().item()*100:.2f}%")

    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_cls[idx], self.y_reg[idx]


#Model
class MultiTaskLSTM(nn.Module):
    """
    Multi-task LSTM for outage forecasting.
    Shared encoder → dual heads (classification + quantile regression).
    """
    def __init__(self, n_features, hidden_size=128, num_layers=2,
                 dropout=0.3, quantiles=(0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention over time steps
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        # Head A: Classification (outage probability)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Head B: Quantile regression (one output per quantile)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(quantiles))
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x)                          # (B, T, 64)
        lstm_out, _ = self.lstm(x)                      # (B, T, H)

        # Attention-weighted pooling
        attn_weights = torch.softmax(
            self.attn(lstm_out).squeeze(-1), dim=1      # (B, T)
        )
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)

        logit    = self.cls_head(context).squeeze(-1)   # (B,)
        quantile = self.reg_head(context)               # (B, n_quantiles)

        return logit, quantile


# Losses
class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification."""
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce  = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)
        alpha_t = torch.where(targets == 1,
                               torch.tensor(self.alpha).to(logits.device),
                               torch.tensor(1 - self.alpha).to(logits.device))
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


def pinball_loss(preds, targets, quantiles):
    """Quantile/pinball loss for probabilistic regression."""
    targets = targets.unsqueeze(-1).expand_as(preds)
    q = torch.tensor(quantiles, dtype=torch.float32).to(preds.device)
    errors = targets - preds
    loss   = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()


# Training loop
def train_epoch(model, loader, optimizer, focal, quantiles, device, cls_weight=0.6):
    model.train()
    total_loss, cls_losses, reg_losses = 0, 0, 0
    for X, y_cls, y_reg in loader:
        X, y_cls, y_reg = X.to(device), y_cls.to(device), y_reg.to(device)
        optimizer.zero_grad()
        logit, q_pred = model(X)
        loss_cls = focal(logit, y_cls)
        loss_reg = pinball_loss(q_pred, y_reg, quantiles)
        loss     = cls_weight * loss_cls + (1 - cls_weight) * loss_reg
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        cls_losses += loss_cls.item()
        reg_losses += loss_reg.item()
    n = len(loader)
    return total_loss/n, cls_losses/n, reg_losses/n


@torch.no_grad()
def evaluate(model, loader, quantiles, device):
    model.eval()
    all_proba, all_labels, all_q50, all_q90, all_targets = [], [], [], [], []
    for X, y_cls, y_reg in loader:
        X = X.to(device)
        logit, q_pred = model(X)
        proba = torch.sigmoid(logit).cpu().numpy()
        all_proba.append(proba)
        all_labels.append(y_cls.numpy())
        all_q50.append(q_pred[:, 0].cpu().numpy())
        all_q90.append(q_pred[:, 1].cpu().numpy())
        all_targets.append(y_reg.numpy())

    proba   = np.concatenate(all_proba)
    labels  = np.concatenate(all_labels)
    q50     = np.concatenate(all_q50)
    q90     = np.concatenate(all_q90)
    targets = np.concatenate(all_targets)

    # Optimal threshold from PR curve
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresh = precision_recall_curve(labels, proba)
    f1s  = 2*prec*rec/(prec+rec+1e-8)
    best = thresh[np.argmax(f1s[:-1])] if len(thresh) > 0 else 0.5
    preds = (proba >= best).astype(int)

    metrics = {
        'roc_auc':  roc_auc_score(labels, proba),
        'pr_auc':   average_precision_score(labels, proba),
        'f1':       f1_score(labels, preds),
        'brier':    brier_score_loss(labels, proba),
        'mae_q50':  mean_absolute_error(targets, q50),
        'threshold':best,
        'proba':    proba,
        'labels':   labels,
        'q50':      q50,
        'q90':      q90
    }
    return metrics


# Main training
def main():
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)

    train_df = pd.read_parquet(FEATURES / 'train.parquet')
    val_df   = pd.read_parquet(FEATURES / 'val.parquet')
    test_df  = pd.read_parquet(FEATURES / 'test.parquet')

    # Fill any remaining NaN
    for df in [train_df, val_df, test_df]:
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    print("\nBuilding sequence datasets...")
    QUANTILES = (0.5, 0.9)
    # Use more counties for val/test (no sliding window limit)
    train_set = OutageDataset(train_df, seq_len=SEQ_LEN, max_counties=400)
    val_set   = OutageDataset(val_df,   seq_len=SEQ_LEN, max_counties=300)
    test_set  = OutageDataset(test_df,  seq_len=SEQ_LEN, max_counties=300)

    BATCH = 512
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=0, pin_memory=DEVICE.type=='cuda')
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False,
                              num_workers=0)

    print("\n" + "=" * 60)
    print("Building model...")
    print("=" * 60)
    model = MultiTaskLSTM(
        n_features=N_FEATURES,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        quantiles=QUANTILES
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")
    print(model)

    focal     = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    EPOCHS       = 30
    PATIENCE     = 7
    best_pr_auc  = 0.0
    patience_cnt = 0
    history      = {'train_loss':[], 'val_roc':[], 'val_pr_auc':[], 'val_f1':[]}

    for epoch in range(1, EPOCHS + 1):
        t_loss, t_cls, t_reg = train_epoch(
            model, train_loader, optimizer, focal, QUANTILES, DEVICE)
        val_m = evaluate(model, val_loader, QUANTILES, DEVICE)
        scheduler.step(val_m['pr_auc'])

        history['train_loss'].append(t_loss)
        history['val_roc'].append(val_m['roc_auc'])
        history['val_pr_auc'].append(val_m['pr_auc'])
        history['val_f1'].append(val_m['f1'])

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {t_loss:.4f} (cls:{t_cls:.4f} reg:{t_reg:.4f}) | "
              f"Val ROC: {val_m['roc_auc']:.4f} | "
              f"PR-AUC: {val_m['pr_auc']:.4f} | "
              f"F1: {val_m['f1']:.4f}")

        if val_m['pr_auc'] > best_pr_auc:
            best_pr_auc = val_m['pr_auc']
            patience_cnt = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optimizer':   optimizer.state_dict(),
                'val_metrics': val_m,
                'feature_cols':FEATURE_COLS,
                'seq_len':     SEQ_LEN,
                'quantiles':   QUANTILES
            }, MODELS / 'lstm_multitask_best.pt')
            print(f"  ✓ Saved best model (PR-AUC={best_pr_auc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best and evaluate on test
    print("\n" + "=" * 60)
    print("Final evaluation on test set (2022)...")
    print("=" * 60)

    ckpt = torch.load(MODELS / 'lstm_multitask_best.pt', map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_m = evaluate(model, test_loader, QUANTILES, DEVICE)

    print(f"\n{'='*40}")
    print("TEST SET RESULTS (2022)")
    print(f"{'='*40}")
    print(f"ROC-AUC  : {test_m['roc_auc']:.4f}")
    print(f"PR-AUC   : {test_m['pr_auc']:.4f}")
    print(f"F1 Score : {test_m['f1']:.4f}")
    print(f"Brier    : {test_m['brier']:.4f}")
    print(f"MAE(Q50) : {test_m['mae_q50']:.4f}")

    #Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Storm Restoration AI — LSTM Evaluation', fontsize=14, fontweight='bold')

    # Training curves
    axes[0,0].plot(history['train_loss'], label='Train Loss', color='tomato')
    axes[0,0].set_title('Training Loss'); axes[0,0].legend()

    axes[0,1].plot(history['val_roc'],    label='ROC-AUC',  color='steelblue')
    axes[0,1].plot(history['val_pr_auc'], label='PR-AUC',   color='darkorange')
    axes[0,1].plot(history['val_f1'],     label='F1',       color='green')
    axes[0,1].set_title('Validation Metrics'); axes[0,1].legend()

    # ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_m['labels'], test_m['proba'])
    axes[0,2].plot(fpr, tpr, color='steelblue', lw=2,
                   label=f"LSTM (AUC={test_m['roc_auc']:.3f})")
    axes[0,2].plot([0,1],[0,1],'k--', label='Random')
    axes[0,2].set_title('ROC Curve (Test 2022)')
    axes[0,2].set_xlabel('FPR'); axes[0,2].set_ylabel('TPR')
    axes[0,2].legend()

    # Calibration
    frac_pos, mean_pred = calibration_curve(
        test_m['labels'], test_m['proba'], n_bins=15)
    axes[1,0].plot(mean_pred, frac_pos, marker='o', color='steelblue', label='LSTM')
    axes[1,0].plot([0,1],[0,1],'k--', label='Perfect')
    axes[1,0].set_title('Calibration Curve'); axes[1,0].legend()

    # Q50 vs actual
    sample_idx = np.random.choice(len(test_m['q50']), size=min(5000, len(test_m['q50'])), replace=False)
    axes[1,1].scatter(test_m['q50'][sample_idx], test_m['labels'][sample_idx],
                      alpha=0.3, s=5, color='darkorange')
    axes[1,1].set_xlabel('Predicted Q50 (log)'); axes[1,1].set_ylabel('Actual Label')
    axes[1,1].set_title('Q50 Prediction vs Label')

    # Outage rate by predicted probability bucket
    buckets = pd.cut(test_m['proba'], bins=10)
    bucket_stats = pd.DataFrame({
        'proba': test_m['proba'],
        'label': test_m['labels'],
        'bucket': buckets
    }).groupby('bucket')['label'].mean()
    axes[1,2].bar(range(len(bucket_stats)), bucket_stats.values, color='steelblue')
    axes[1,2].set_title('Actual Outage Rate by Predicted Probability Decile')
    axes[1,2].set_xlabel('Probability Decile')

    plt.tight_layout()
    plt.savefig(PLOTS / 'lstm_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {PLOTS / 'lstm_evaluation.png'}")

    # Save results for Streamlit app
    results = {
        'test_metrics': test_m,
        'history':      history,
        'feature_cols': FEATURE_COLS,
        'seq_len':      SEQ_LEN,
        'quantiles':    QUANTILES,
        'best_pr_auc':  best_pr_auc
    }
    with open(MODELS / 'training_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best model: {MODELS / 'lstm_multitask_best.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()