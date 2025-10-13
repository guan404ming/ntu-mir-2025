"""Visualize retrieval and evaluation results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Load results
results_dir = Path("results")
asset_dir = Path("../assets")
asset_dir.mkdir(exist_ok=True)

# Load evaluation results
encoders = ["clap", "music2latent", "muq"]
eval_data = {}
for encoder in encoders:
    eval_file = results_dir / f"evaluation_results_{encoder}.json"
    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data[encoder] = json.load(f)

# Get target names (shorten for display)
target_names = list(eval_data["clap"].keys())
target_short = [
    "Country", "Jazz", "Rock",
    "Hedwig(Dizi)", "Mussorgsky(Piano)", "Spirited Away(Piano)",
    "IRIS OUT(Piano)", "菊花台(Dizi)", "莫文蔚(Dizi)"
]

# 1. Compare Retrieval Similarity across encoders
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(target_names))
width = 0.25

for i, encoder in enumerate(encoders):
    similarities = [eval_data[encoder][target]["retrieval_similarity"]
                   for target in target_names]
    ax.bar(x + i*width, similarities, width, label=encoder.upper(), alpha=0.8)

ax.set_xlabel('Target Track')
ax.set_ylabel('Retrieval Similarity')
ax.set_title('Retrieval Similarity Comparison Across Encoders')
ax.set_xticks(x + width)
ax.set_xticklabels(target_short, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(asset_dir / "task1_retrieval_similarity.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'task1_retrieval_similarity.png'}")
plt.close()

# 2. CLAP Similarity Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for i, encoder in enumerate(encoders):
    clap_sims = [eval_data[encoder][target]["clap_similarity"]
                for target in target_names]
    ax.bar(x + i*width, clap_sims, width, label=encoder.upper(), alpha=0.8)

ax.set_xlabel('Target Track')
ax.set_ylabel('CLAP Similarity')
ax.set_title('CLAP Similarity (Evaluation Metric) Across Encoders')
ax.set_xticks(x + width)
ax.set_xticklabels(target_short, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(asset_dir / "clap_similarity_comparison.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'clap_similarity_comparison.png'}")
plt.close()

# 3. Melody Accuracy Comparison
fig, ax = plt.subplots(figsize=(12, 6))
for i, encoder in enumerate(encoders):
    melody_accs = [eval_data[encoder][target]["melody_accuracy"]
                  for target in target_names]
    ax.bar(x + i*width, melody_accs, width, label=encoder.upper(), alpha=0.8)

ax.set_xlabel('Target Track')
ax.set_ylabel('Melody Accuracy')
ax.set_title('Melody Accuracy Comparison Across Encoders')
ax.set_xticks(x + width)
ax.set_xticklabels(target_short, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(asset_dir / "melody_accuracy_comparison.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'melody_accuracy_comparison.png'}")
plt.close()

# 4. Audiobox Aesthetics - 4 dimensions
aesthetics_dims = ["ce", "cu", "pc", "pq"]
aesthetics_names = ["Content Enjoyment", "Content Usefulness",
                   "Production Complexity", "Production Quality"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for dim_idx, (dim, dim_name) in enumerate(zip(aesthetics_dims, aesthetics_names)):
    ax = axes[dim_idx]
    for i, encoder in enumerate(encoders):
        values = [eval_data[encoder][target]["aesthetics"][dim]
                 for target in target_names]
        ax.bar(x + i*width, values, width, label=encoder.upper(), alpha=0.8)

    ax.set_xlabel('Target Track')
    ax.set_ylabel(f'{dim_name} Score')
    ax.set_title(f'Audiobox Aesthetics: {dim_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(target_short, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(asset_dir / "aesthetics_comparison.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'aesthetics_comparison.png'}")
plt.close()

# 5. Overall Performance Heatmap
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["retrieval_similarity", "clap_similarity", "melody_accuracy"]
metric_names = ["Retrieval Similarity", "CLAP Similarity", "Melody Accuracy"]

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    data_matrix = []
    for encoder in encoders:
        row = [eval_data[encoder][target][metric] for target in target_names]
        data_matrix.append(row)

    im = axes[idx].imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[idx].set_xticks(np.arange(len(target_short)))
    axes[idx].set_yticks(np.arange(len(encoders)))
    axes[idx].set_xticklabels(target_short, rotation=45, ha='right')
    axes[idx].set_yticklabels([e.upper() for e in encoders])
    axes[idx].set_title(metric_name)

    # Add values as text
    for i in range(len(encoders)):
        for j in range(len(target_names)):
            text = axes[idx].text(j, i, f'{data_matrix[i][j]:.2f}',
                                ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig(asset_dir / "task1_performance_heatmap.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'task1_performance_heatmap.png'}")
plt.close()

# 6. Average metrics per encoder
fig, ax = plt.subplots(figsize=(10, 6))

all_metrics = ["retrieval_similarity", "clap_similarity", "melody_accuracy",
               "ce", "cu", "pc", "pq"]
metric_labels = ["Retrieval Sim", "CLAP Sim", "Melody Acc",
                "CE", "CU", "PC", "PQ"]

x_pos = np.arange(len(all_metrics))
width = 0.25

for i, encoder in enumerate(encoders):
    avg_values = []
    for metric in all_metrics:
        if metric in ["ce", "cu", "pc", "pq"]:
            values = [eval_data[encoder][target]["aesthetics"][metric]
                     for target in target_names]
            # Normalize aesthetics to 0-1 range (assuming max ~8)
            avg_values.append(np.mean(values) / 8.0)
        else:
            values = [eval_data[encoder][target][metric]
                     for target in target_names]
            avg_values.append(np.mean(values))

    ax.bar(x_pos + i*width, avg_values, width, label=encoder.upper(), alpha=0.8)

ax.set_xlabel('Metrics')
ax.set_ylabel('Average Score (Normalized)')
ax.set_title('Average Performance Across All Metrics')
ax.set_xticks(x_pos + width)
ax.set_xticklabels(metric_labels)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(asset_dir / "average_performance.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'average_performance.png'}")
plt.close()

# 7. Summary Statistics Table (as image)
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

summary_data = []
for encoder in encoders:
    row = [encoder.upper()]

    # Retrieval similarity
    ret_sim = [eval_data[encoder][t]["retrieval_similarity"] for t in target_names]
    row.append(f"{np.mean(ret_sim):.3f} ± {np.std(ret_sim):.3f}")

    # CLAP similarity
    clap_sim = [eval_data[encoder][t]["clap_similarity"] for t in target_names]
    row.append(f"{np.mean(clap_sim):.3f} ± {np.std(clap_sim):.3f}")

    # Melody accuracy
    melody = [eval_data[encoder][t]["melody_accuracy"] for t in target_names]
    row.append(f"{np.mean(melody):.3f} ± {np.std(melody):.3f}")

    # Aesthetics
    for dim in aesthetics_dims:
        aes_vals = [eval_data[encoder][t]["aesthetics"][dim] for t in target_names]
        row.append(f"{np.mean(aes_vals):.2f} ± {np.std(aes_vals):.2f}")

    summary_data.append(row)

columns = ["Encoder", "Retrieval Sim", "CLAP Sim", "Melody Acc", "CE", "CU", "PC", "PQ"]
table = ax.table(cellText=summary_data, colLabels=columns,
                cellLoc='center', loc='center',
                colColours=['lightgray']*len(columns))
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# No title for this image
plt.savefig(asset_dir / "task1_summary_statistics.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'task1_summary_statistics.png'}")
plt.close()

print("\n✓ All visualizations created successfully!")
print(f"Total images saved: 7")
print(f"Output directory: {asset_dir.absolute()}")
