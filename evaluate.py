"""
Storm Restoration Prioritization AI
Phase 3: Priority Scoring + Full Evaluation

Generates:
  - Model comparison table (LR vs GB vs LSTM)
  - Top-K recall curves
  - Priority score ranked county list
  - outputs/evaluation_results.pkl  (for Streamlit)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, brier_score_loss, roc_curve,
                              precision_recall_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

BASE     = Path(r'C:\Users\harsh\OneDrive - Global Academy of Technology\Desktop\myprojects\storm_restoration_ai')
FEATURES = BASE / 'data' / 'features'
MODELS   = BASE / 'outputs' / 'models'
PLOTS    = BASE / 'outputs' / 'plots'
PLOTS.mkdir(parents=True, exist_ok=True)

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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load LSTM model
def load_lstm():
    from train_model import MultiTaskLSTM, OutageDataset
    ckpt = torch.load(MODELS / 'lstm_multitask_best.pt', map_location=DEVICE, weights_only=False)
    model = MultiTaskLSTM(n_features=len(FEATURE_COLS),
                          hidden_size=128, num_layers=2,
                          dropout=0.3, quantiles=(0.5, 0.9)).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model

# Priority score formula
def compute_priority_scores(df, proba, q90_log):
    """
    PriorityScore = P(outage) × exp(Q90_log) × VulnerabilityWeight
    VulnerabilityWeight = normalized county_p90 (proxy for population scale)
    """
    q90_customers = np.expm1(np.clip(q90_log, 0, 20))  # inverse log1p
    vuln_weight   = df['county_p90'].values / (df['county_p90'].values.max() + 1e-8)
    score = proba * q90_customers * (1 + vuln_weight)
    return score


# Top-K recall
def top_k_recall(labels, scores, customers, k_vals):
    """Fraction of true high-impact events captured in top-K predictions."""
    order = np.argsort(scores)[::-1]
    total_impact = customers[labels == 1].sum()
    results = []
    for k in k_vals:
        top_k_idx   = order[:k]
        top_k_cust  = customers[top_k_idx][labels[top_k_idx] == 1].sum()
        recall_k    = top_k_cust / (total_impact + 1e-8)
        results.append(recall_k)
    return np.array(results)


# Main evaluation
def main():
    print("=" * 60)
    print("Loading test data (2022)...")
    print("=" * 60)

    test_df = pd.read_parquet(FEATURES / 'test.parquet')
    test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(0)
    train_df = pd.read_parquet(FEATURES / 'train.parquet')
    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0)

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df['outage_label'].values
    X_test  = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test  = test_df['outage_label'].values

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    all_results = {}

    #Baseline 1: Logistic Regression
    print("\nTraining Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1, random_state=42)
    # Sample for speed
    idx = np.random.choice(len(X_train_sc), size=min(300000, len(X_train_sc)), replace=False)
    lr.fit(X_train_sc[idx], y_train[idx])
    lr_proba = lr.predict_proba(X_test_sc)[:,1]
    prec, rec, thresh = precision_recall_curve(y_test, lr_proba)
    f1s = 2*prec*rec/(prec+rec+1e-8)
    best_t = thresh[np.argmax(f1s[:-1])] if len(thresh) > 0 else 0.5
    lr_pred = (lr_proba >= best_t).astype(int)
    all_results['Logistic Regression'] = {
        'roc_auc': roc_auc_score(y_test, lr_proba),
        'pr_auc':  average_precision_score(y_test, lr_proba),
        'f1':      f1_score(y_test, lr_pred),
        'brier':   brier_score_loss(y_test, lr_proba),
        'proba':   lr_proba
    }
    print(f"  LR ROC-AUC: {all_results['Logistic Regression']['roc_auc']:.4f}")

    #Baseline 2: Gradient Boosting
    print("\nTraining Gradient Boosting baseline...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_s = min(150000, len(pos_idx))
    s_idx = np.concatenate([
        np.random.choice(pos_idx, n_s, replace=False),
        np.random.choice(neg_idx, n_s, replace=False)
    ])
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    gb.fit(X_train[s_idx], y_train[s_idx])
    gb_proba = gb.predict_proba(X_test)[:,1]
    prec, rec, thresh = precision_recall_curve(y_test, gb_proba)
    f1s = 2*prec*rec/(prec+rec+1e-8)
    best_t = thresh[np.argmax(f1s[:-1])] if len(thresh) > 0 else 0.5
    gb_pred = (gb_proba >= best_t).astype(int)
    all_results['Gradient Boosting'] = {
        'roc_auc': roc_auc_score(y_test, gb_proba),
        'pr_auc':  average_precision_score(y_test, gb_proba),
        'f1':      f1_score(y_test, gb_pred),
        'brier':   brier_score_loss(y_test, gb_proba),
        'proba':   gb_proba
    }
    print(f"  GB ROC-AUC: {all_results['Gradient Boosting']['roc_auc']:.4f}")

    #LSTM
    print("\nLoading LSTM results...")
    with open(MODELS / 'training_results.pkl', 'rb') as f:
        lstm_results = pickle.load(f)
    tm = lstm_results['test_metrics']
    all_results['Multi-Task LSTM'] = {
        'roc_auc': tm['roc_auc'],
        'pr_auc':  tm['pr_auc'],
        'f1':      tm['f1'],
        'brier':   tm['brier'],
        'proba':   tm['proba'],
        'q50':     tm['q50'],
        'q90':     tm['q90'],
        'labels':  tm['labels']
    }
    print(f"  LSTM ROC-AUC: {all_results['Multi-Task LSTM']['roc_auc']:.4f}")

    #Comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    comp = pd.DataFrame({
        'Model':    list(all_results.keys()),
        'ROC-AUC':  [v['roc_auc'] for v in all_results.values()],
        'PR-AUC':   [v['pr_auc']  for v in all_results.values()],
        'F1':       [v['f1']      for v in all_results.values()],
        'Brier':    [v['brier']   for v in all_results.values()],
    }).round(4)
    print(comp.to_string(index=False))

    #Priority scores
    print("\nComputing priority scores...")
    # Aligning LSTM test results with test_df (sample same counties/hours used in LSTM)
    # To Use GB probabilities on full test set for priority scoring (full coverage)
    test_df['outage_proba']  = gb_proba
    test_df['priority_score'] = compute_priority_scores(
        test_df, gb_proba,
        np.log1p(test_df['peak_customers_out'].values)
    )

    # County-level daily priority queue
    county_priority = (test_df
                       .groupby('fips')
                       .agg(
                           mean_outage_proba  =('outage_proba','mean'),
                           max_outage_proba   =('outage_proba','max'),
                           mean_priority_score=('priority_score','mean'),
                           max_priority_score =('priority_score','max'),
                           peak_customers_out =('peak_customers_out','max'),
                           outage_events      =('outage_label','sum'),
                           total_hours        =('outage_label','count'),
                           state              =('state','first')
                       )
                       .reset_index())
    county_priority['outage_rate'] = (
        county_priority['outage_events'] / county_priority['total_hours'])
    county_priority = county_priority.sort_values(
        'max_priority_score', ascending=False).reset_index(drop=True)
    county_priority['rank'] = county_priority.index + 1

    county_priority.to_parquet(MODELS / 'county_priority_2022.parquet', index=False)
    print(f"Top 10 counties by priority:")
    print(county_priority[['rank','fips','state','max_outage_proba',
                            'peak_customers_out','max_priority_score']].head(10).to_string(index=False))

    # Top-K recall curves
    k_vals = np.arange(10, min(501, len(y_test)//100), 10)
    customers = test_df['peak_customers_out'].values[:len(tm['labels'])]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Storm Restoration AI — Model Comparison (Test 2022)',
                 fontsize=13, fontweight='bold')

    # ROC curves
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for (name, res), color in zip(all_results.items(), colors):
        n = min(len(y_test), len(res['proba']))
        fpr, tpr, _ = roc_curve(y_test[:n], res['proba'][:n])
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name} ({res['roc_auc']:.3f})")
    axes[0].plot([0,1],[0,1],'k--', alpha=0.5)
    axes[0].set_title('ROC Curves')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(fontsize=8)

    # PR curves
    for (name, res), color in zip(all_results.items(), colors):
        n = min(len(y_test), len(res['proba']))
        prec, rec, _ = precision_recall_curve(y_test[:n], res['proba'][:n])
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{name} ({res['pr_auc']:.3f})")
    axes[1].set_title('Precision-Recall Curves')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(fontsize=8)

    # Metric comparison bar
    metrics_to_plot = ['roc_auc','pr_auc','f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, (name, res) in enumerate(all_results.items()):
        vals = [res[m] for m in metrics_to_plot]
        axes[2].bar(x + i*width, vals, width,
                    label=name, color=colors[i], alpha=0.85)
    axes[2].set_xticks(x + width)
    axes[2].set_xticklabels(['ROC-AUC','PR-AUC','F1'])
    axes[2].set_title('Metric Comparison')
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved model_comparison.png")

    # Save everything for Streamlit
    eval_output = {
        'all_results':       all_results,
        'comparison_table':  comp,
        'county_priority':   county_priority,
        'feature_cols':      FEATURE_COLS,
        'test_y':            y_test,
        'gb_model':          gb,
        'lr_model':          lr,
        'scaler':            scaler
    }
    with open(MODELS / 'evaluation_results.pkl', 'wb') as f:
        pickle.dump(eval_output, f)
    
    # Save baseline models separately
    with open(MODELS / 'gradient_boosting_final.pkl', 'wb') as f:
        pickle.dump(gb, f)
    with open(MODELS / 'logistic_regression_final.pkl', 'wb') as f:
        pickle.dump(lr, f)
    with open(MODELS / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(comp.to_string(index=False))


if __name__ == '__main__':
    main()