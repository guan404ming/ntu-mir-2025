"""Create visualizations for Task 1 report including epoch experiments."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Define paths
results_dir = Path("results")
assets_dir = Path("assets")
assets_dir.mkdir(exist_ok=True)

# Load all experiment data (including epoch experiments)
experiments = []
for exp_dir in sorted(results_dir.iterdir()):
    if not exp_dir.is_dir():
        continue

    task1_dir = exp_dir / "task1"
    config_path = task1_dir / "config.json"
    scores_path = task1_dir / "scores.csv"

    if not config_path.exists() or not scores_path.exists():
        continue

    with open(config_path) as f:
        config = json.load(f)

    scores = pd.read_csv(scores_path)

    # Parse experiment name for epoch info
    exp_name = exp_dir.name
    epoch_match = re.search(r'_epoch(\d+)$', exp_name)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        base_name = exp_name.replace(f'_epoch{epoch}', '')
    else:
        epoch = 100  # Default epoch for non-epoch experiments
        base_name = exp_name

    exp_data = {
        "name": exp_name,
        "base_name": base_name,
        "epoch": epoch,
        "model": config["checkpoint"]["model_name"].split("/")[-1],
        "tokenizer": config["checkpoint"]["tokenizer_name"],
        "top_k": config["top_k"],
        "temperature": config["temperature"],
        "repetition_penalty": config.get("repetition_penalty", 1.0),
        "h1_mean": scores["h1"].mean(),
        "h1_std": scores["h1"].std(),
        "h4_mean": scores["h4"].mean(),
        "h4_std": scores["h4"].std(),
        "gs_mean": scores["gs"].mean(),
        "gs_std": scores["gs"].std(),
    }
    experiments.append(exp_data)

df = pd.DataFrame(experiments)
print("Loaded experiments:")
print(df[["name", "model", "tokenizer", "epoch", "h4_mean", "gs_mean"]])

# Separate epoch experiments from final (epoch 100) experiments
df_epochs = df[df['epoch'] != 100].copy()
df_final = df[df['epoch'] == 100].copy()

# 1. Epoch Training Progress - GS over epochs for each configuration (including epoch 100)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Get unique configurations from all data
configs = df.groupby(['model', 'tokenizer']).first().reset_index()[['model', 'tokenizer']]
colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

for idx, (_, row) in enumerate(configs.iterrows()):
    model, tok = row['model'], row['tokenizer']
    mask = (df['model'] == model) & (df['tokenizer'] == tok)
    subset = df[mask].sort_values('epoch')

    label = f"{model}\n{tok}"
    axes[0].plot(subset['epoch'], subset['gs_mean'], 'o-', color=colors[idx], label=label, markersize=8)
    axes[1].plot(subset['epoch'], subset['h4_mean'], 'o-', color=colors[idx], label=label, markersize=8)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('GS (Grooving Pattern Similarity)')
axes[0].set_title('GS vs Training Epoch')
axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[0].set_xticks([10, 30, 50, 70, 90, 100])

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('H4 (Pitch-Class Histogram Entropy)')
axes[1].set_title('H4 vs Training Epoch')
axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
axes[1].set_xticks([10, 30, 50, 70, 90, 100])

plt.tight_layout()
plt.savefig(assets_dir / "task1_epoch_training_progress.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. Average metrics by epoch (aggregated across all configs, including epoch 100)
fig, ax = plt.subplots(figsize=(12, 5))

epoch_avg = df.groupby('epoch').agg({
    'h4_mean': 'mean',
    'gs_mean': 'mean'
}).reset_index()

x = np.arange(len(epoch_avg))
width = 0.35

bars1 = ax.bar(x - width/2, epoch_avg['h4_mean'], width, label='H4', color='coral', edgecolor='black')
bars2 = ax.bar(x + width/2, epoch_avg['gs_mean'], width, label='GS', color='steelblue', edgecolor='black')

ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Average Metrics by Training Epoch (All Configurations)')
ax.set_xticks(x)
ax.set_xticklabels(epoch_avg['epoch'].astype(int))
ax.legend()
ax.set_ylim(0, 3.0)  # Set fixed y-axis limit

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(assets_dir / "task1_epoch_average_metrics.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. All epochs configurations comparison - grouped bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Group by model and tokenizer, show all epochs
configs_list = df.groupby(['model', 'tokenizer']).first().reset_index()[['model', 'tokenizer']]
epochs = sorted(df["epoch"].unique())
n_configs = len(configs_list)
n_epochs = len(epochs)

x = np.arange(n_configs)
width = 0.12
colors = plt.cm.viridis(np.linspace(0, 0.9, n_epochs))

# GS comparison
for i, epoch in enumerate(epochs):
    gs_values = []
    for _, row in configs_list.iterrows():
        mask = (df["model"] == row['model']) & (df["tokenizer"] == row['tokenizer']) & (df["epoch"] == epoch)
        if mask.any():
            gs_values.append(df.loc[mask, "gs_mean"].values[0])
        else:
            gs_values.append(0)
    axes[0].bar(x + i * width, gs_values, width, label=f'Epoch {epoch}', color=colors[i], edgecolor='black', linewidth=0.3)

axes[0].set_ylabel("GS (Grooving Pattern Similarity)")
axes[0].set_title("GS Comparison - All Epochs")
axes[0].set_xticks(x + width * (n_epochs - 1) / 2)
axes[0].set_xticklabels([f"{r['model'][:10]}\n{r['tokenizer']}" for _, r in configs_list.iterrows()], rotation=45, ha="right", fontsize=8)
axes[0].legend(loc='upper right', fontsize=7)
axes[0].set_ylim(0.6, 1.0)

# H4 comparison
for i, epoch in enumerate(epochs):
    h4_values = []
    for _, row in configs_list.iterrows():
        mask = (df["model"] == row['model']) & (df["tokenizer"] == row['tokenizer']) & (df["epoch"] == epoch)
        if mask.any():
            h4_values.append(df.loc[mask, "h4_mean"].values[0])
        else:
            h4_values.append(0)
    axes[1].bar(x + i * width, h4_values, width, label=f'Epoch {epoch}', color=colors[i], edgecolor='black', linewidth=0.3)

axes[1].set_ylabel("H4 (Pitch-Class Histogram Entropy)")
axes[1].set_title("H4 Comparison - All Epochs")
axes[1].set_xticks(x + width * (n_epochs - 1) / 2)
axes[1].set_xticklabels([f"{r['model'][:10]}\n{r['tokenizer']}" for _, r in configs_list.iterrows()], rotation=45, ha="right", fontsize=8)
axes[1].legend(loc='upper right', fontsize=7)

plt.tight_layout()
plt.savefig(assets_dir / "task1_metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# 4. Heatmap: All epochs - Configuration (Model+Tokenizer) vs Epoch
models = df["model"].unique()
tokenizers = df["tokenizer"].unique()
epochs = sorted(df["epoch"].unique())

# Create configuration labels
config_labels = []
for model in models:
    for tok in tokenizers:
        mask = (df["model"] == model) & (df["tokenizer"] == tok)
        if mask.any():
            config_labels.append(f"{model[:10]}\n{tok}")

# Create GS heatmap: configs x epochs
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

n_configs = len(config_labels)
n_epochs = len(epochs)

gs_matrix = np.zeros((n_configs, n_epochs))
h4_matrix = np.zeros((n_configs, n_epochs))

config_idx = 0
for model in models:
    for tok in tokenizers:
        for j, epoch in enumerate(epochs):
            mask = (df["model"] == model) & (df["tokenizer"] == tok) & (df["epoch"] == epoch)
            if mask.any():
                gs_matrix[config_idx, j] = df.loc[mask, "gs_mean"].values[0]
                h4_matrix[config_idx, j] = df.loc[mask, "h4_mean"].values[0]
            else:
                gs_matrix[config_idx, j] = np.nan
                h4_matrix[config_idx, j] = np.nan
        config_idx += 1

# GS heatmap
im1 = axes[0].imshow(gs_matrix, cmap='YlGn', aspect='auto')
axes[0].set_xticks(range(n_epochs))
axes[0].set_xticklabels([str(e) for e in epochs])
axes[0].set_yticks(range(n_configs))
axes[0].set_yticklabels(config_labels, fontsize=8)
axes[0].set_title("GS by Configuration and Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Configuration")

for i in range(n_configs):
    for j in range(n_epochs):
        if not np.isnan(gs_matrix[i, j]):
            axes[0].text(j, i, f'{gs_matrix[i, j]:.2f}',
                        ha="center", va="center", fontsize=7, fontweight='bold')
plt.colorbar(im1, ax=axes[0])

# H4 heatmap
im2 = axes[1].imshow(h4_matrix, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(n_epochs))
axes[1].set_xticklabels([str(e) for e in epochs])
axes[1].set_yticks(range(n_configs))
axes[1].set_yticklabels(config_labels, fontsize=8)
axes[1].set_title("H4 by Configuration and Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Configuration")

for i in range(n_configs):
    for j in range(n_epochs):
        if not np.isnan(h4_matrix[i, j]):
            axes[1].text(j, i, f'{h4_matrix[i, j]:.2f}',
                        ha="center", va="center", fontsize=7, fontweight='bold')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig(assets_dir / "task1_model_tokenizer_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# 5. GS and H4 improvement from epoch 10 with finer stages
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Define epoch stages
target_epochs = [30, 50, 70, 90, 100]
colors_stages = plt.cm.viridis(np.linspace(0.2, 0.9, len(target_epochs)))
width = 0.15

# GS improvement
for idx, (_, row) in enumerate(configs.iterrows()):
    model, tok = row['model'], row['tokenizer']
    mask = (df['model'] == model) & (df['tokenizer'] == tok)
    subset = df[mask].sort_values('epoch')

    gs_10 = subset[subset['epoch'] == 10]['gs_mean'].values
    if len(gs_10) > 0:
        for i, epoch in enumerate(target_epochs):
            gs_target = subset[subset['epoch'] == epoch]['gs_mean'].values
            if len(gs_target) > 0:
                improvement = (gs_target[0] - gs_10[0]) / gs_10[0] * 100
                axes[0].bar(idx + (i - 2) * width, improvement, width,
                           label=f'10→{epoch}' if idx == 0 else '', color=colors_stages[i])

axes[0].set_xlabel('Configuration')
axes[0].set_ylabel('GS Improvement (%)')
axes[0].set_title('GS Improvement from Epoch 10')
axes[0].set_xticks(range(len(configs)))
axes[0].set_xticklabels([f"{r['model'][:10]}\n{r['tokenizer']}" for _, r in configs.iterrows()], rotation=45, ha='right')
axes[0].legend(loc='upper right', fontsize=7)
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# H4 change
for idx, (_, row) in enumerate(configs.iterrows()):
    model, tok = row['model'], row['tokenizer']
    mask = (df['model'] == model) & (df['tokenizer'] == tok)
    subset = df[mask].sort_values('epoch')

    h4_10 = subset[subset['epoch'] == 10]['h4_mean'].values
    if len(h4_10) > 0:
        for i, epoch in enumerate(target_epochs):
            h4_target = subset[subset['epoch'] == epoch]['h4_mean'].values
            if len(h4_target) > 0:
                change = (h4_target[0] - h4_10[0]) / h4_10[0] * 100
                axes[1].bar(idx + (i - 2) * width, change, width,
                           label=f'10→{epoch}' if idx == 0 else '', color=colors_stages[i])

axes[1].set_xlabel('Configuration')
axes[1].set_ylabel('H4 Change (%)')
axes[1].set_title('H4 Change from Epoch 10')
axes[1].set_xticks(range(len(configs)))
axes[1].set_xticklabels([f"{r['model'][:10]}\n{r['tokenizer']}" for _, r in configs.iterrows()], rotation=45, ha='right')
axes[1].legend(loc='upper right', fontsize=7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig(assets_dir / "task1_gs_improvement.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\nVisualizations saved to {assets_dir}")

# Print summary tables for report
print("\n" + "="*80)
print("EPOCH EXPERIMENT RESULTS")
print("="*80)
print(df_epochs[["model", "tokenizer", "epoch", "h1_mean", "h4_mean", "gs_mean"]].sort_values(['model', 'tokenizer', 'epoch']).to_string(index=False))

if len(df_final) > 0:
    print("\n" + "="*80)
    print("FINAL MODEL RESULTS (EPOCH 100)")
    print("="*80)
    print(df_final[["model", "tokenizer", "top_k", "temperature", "h1_mean", "h4_mean", "gs_mean"]].to_string(index=False))
