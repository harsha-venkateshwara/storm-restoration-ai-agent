"""
Run only the test evaluation — no retraining.
Uses already saved lstm_multitask_best.pt
"""
import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, mean_absolute_error, roc_curve, precision_recall_curve
import pandas as pd

# Import model classes from train_model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_model import MultiTaskLSTM, OutageDataset, evaluate, FEATURE_COLS, SEQ_LEN, N_FEATURES

BASE    = Path(r'C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai')
MODELS  = BASE / 'outputs' / 'models'
PLOTS   = BASE / 'outputs' / 'plots'
FEATURES= BASE / 'data' / 'features'
PLOTS.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load test data
print("Loading test data...")
test_df = pd.read_parquet(FEATURES / 'test.parquet')
test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(0)

QUANTILES = (0.5, 0.9)
test_set    = OutageDataset(test_df, seq_len=SEQ_LEN, max_counties=300)
test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

# Load best model
print("Loading best model checkpoint...")
ckpt  = torch.load(MODELS / 'lstm_multitask_best.pt', map_location=DEVICE, weights_only=False)
model = MultiTaskLSTM(n_features=N_FEATURES, hidden_size=128,
                      num_layers=2, dropout=0.3, quantiles=QUANTILES).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
print(f"Loaded checkpoint from epoch {ckpt['epoch']} | Val PR-AUC: {ckpt['val_metrics']['pr_auc']:.4f}")

# Evaluate on test set
print("\nEvaluating on test set (2022)...")
test_m = evaluate(model, test_loader, QUANTILES, DEVICE)

print(f"\n{'='*40}")
print("TEST SET RESULTS (2022)")
print(f"{'='*40}")
print(f"ROC-AUC  : {test_m['roc_auc']:.4f}")
print(f"PR-AUC   : {test_m['pr_auc']:.4f}")
print(f"F1 Score : {test_m['f1']:.4f}")
print(f"Brier    : {test_m['brier']:.4f}")
print(f"MAE(Q50) : {test_m['mae_q50']:.4f}")

# Load training history from checkpoint
history = ckpt.get('history', {
    'train_loss': [], 'val_roc': [], 'val_pr_auc': [], 'val_f1': []
})

# Generate plots
print("\nGenerating evaluation plots...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Storm Restoration AI — LSTM Evaluation (Test 2022)',
             fontsize=14, fontweight='bold')

# ROC curve
fpr, tpr, _ = roc_curve(test_m['labels'], test_m['proba'])
axes[0,0].plot(fpr, tpr, color='steelblue', lw=2,
               label=f"LSTM (AUC={test_m['roc_auc']:.3f})")
axes[0,0].plot([0,1],[0,1],'k--', label='Random')
axes[0,0].set_title('ROC Curve (Test 2022)')
axes[0,0].set_xlabel('FPR'); axes[0,0].set_ylabel('TPR')
axes[0,0].legend()

# PR curve
prec, rec, _ = precision_recall_curve(test_m['labels'], test_m['proba'])
axes[0,1].plot(rec, prec, color='darkorange', lw=2,
               label=f"LSTM (PR-AUC={test_m['pr_auc']:.3f})")
axes[0,1].set_title('Precision-Recall Curve')
axes[0,1].set_xlabel('Recall'); axes[0,1].set_ylabel('Precision')
axes[0,1].legend()

# Calibration
frac_pos, mean_pred = calibration_curve(test_m['labels'], test_m['proba'], n_bins=15)
axes[0,2].plot(mean_pred, frac_pos, marker='o', color='steelblue', label='LSTM')
axes[0,2].plot([0,1],[0,1],'k--', label='Perfect')
axes[0,2].set_title('Calibration Curve')
axes[0,2].legend()

# Probability distribution
axes[1,0].hist(test_m['proba'][test_m['labels']==0], bins=50, alpha=0.6,
               color='steelblue', label='No Outage', density=True)
axes[1,0].hist(test_m['proba'][test_m['labels']==1], bins=50, alpha=0.6,
               color='tomato', label='Outage', density=True)
axes[1,0].set_title('Predicted Probability Distribution')
axes[1,0].set_xlabel('P(Outage)'); axes[1,0].legend()

# Q50 scatter
sample_idx = np.random.choice(len(test_m['q50']),
                               size=min(5000, len(test_m['q50'])), replace=False)
axes[1,1].scatter(test_m['q50'][sample_idx], test_m['labels'][sample_idx],
                  alpha=0.3, s=5, color='darkorange')
axes[1,1].set_xlabel('Predicted Q50 (log scale)')
axes[1,1].set_ylabel('Actual Label')
axes[1,1].set_title('Q50 Prediction vs Label')

# Outage rate by probability decile
buckets = pd.cut(test_m['proba'], bins=10)
bucket_stats = pd.DataFrame({
    'proba': test_m['proba'],
    'label': test_m['labels'],
    'bucket': buckets
}).groupby('bucket', observed=True)['label'].mean()
axes[1,2].bar(range(len(bucket_stats)), bucket_stats.values, color='steelblue')
axes[1,2].set_title('Actual Outage Rate by Predicted Probability Decile')
axes[1,2].set_xlabel('Probability Decile')

plt.tight_layout()
plt.savefig(PLOTS / 'lstm_evaluation.png', dpi=150, bbox_inches='tight')
print(f"Saved → {PLOTS / 'lstm_evaluation.png'}")

# Save results for evaluate.py and Streamlit
results = {
    'test_metrics': test_m,
    'history':      history,
    'feature_cols': FEATURE_COLS,
    'seq_len':      SEQ_LEN,
    'quantiles':    QUANTILES,
    'best_pr_auc':  ckpt['val_metrics']['pr_auc']
}
with open(MODELS / 'training_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print(f"Saved → training_results.pkl")
print("\nDone. Now run: python evaluate.py")