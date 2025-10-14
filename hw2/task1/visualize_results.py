"""Visualize retrieval and evaluation results for Task 1.

Task 1 Visualization Overview:
This script creates visualizations comparing different audio encoders (CLAP, Music2Latent, MuQ)
for music retrieval. For each target music, we retrieved the most similar reference music
and evaluated it using the following metrics:

1. Retrieval Similarity: Encoder-specific cosine similarity used for retrieval
2. CLAP Similarity: CLAP-based cosine similarity between target and retrieved reference
3. Melody Accuracy: Chromagram-based melody matching between target and retrieved reference
4. Audiobox Aesthetics: Quality metrics (CE, CU, PC, PQ) of the retrieved reference music
   - CE: Content Enjoyment
   - CU: Content Usefulness
   - PC: Production Complexity
   - PQ: Production Quality

The visualizations help compare how well different encoders retrieve similar music
and how the retrieved music compares to the target across multiple dimensions.
"""

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
# This shows how similar the retrieved reference music is to the target,
# according to each encoder's embedding space
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(target_names))
width = 0.25

for i, encoder in enumerate(encoders):
    similarities = [eval_data[encoder][target]["retrieval_similarity"]
                   for target in target_names]
    ax.bar(x + i*width, similarities, width, label=encoder.upper(), alpha=0.8)

ax.set_xlabel('Target Track')
ax.set_ylabel('Retrieval Similarity (Cosine)')
ax.set_title('Retrieval Similarity: Target vs Retrieved Reference Music (by Encoder)')
ax.set_xticks(x + width)
ax.set_xticklabels(target_short, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(asset_dir / "task1_retrieval_similarity.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'task1_retrieval_similarity.png'}")
plt.close()

# Note: Only generating images that are used in report.md
# Skipping: clap_similarity_comparison.png, melody_accuracy_comparison.png,
#           aesthetics_comparison.png, average_performance.png

# Prepare aesthetics dims for summary table
aesthetics_dims = ["ce", "cu", "pc", "pq"]

# 2. Overall Performance Heatmap
# Heatmap showing three key metrics for comparing retrieval performance
# - Retrieval Similarity: How well the encoder retrieved similar music
# - CLAP Similarity: How similar target and retrieved are in CLAP space
# - Melody Accuracy: How well the melody matches between target and retrieved
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics = ["retrieval_similarity", "clap_similarity", "melody_accuracy"]
metric_names = ["Retrieval Similarity\n(Encoder-specific)",
                "CLAP Similarity\n(Target vs Retrieved)",
                "Melody Accuracy\n(Target vs Retrieved)"]

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

# 3. Summary Statistics Table (as image)
# Summary table showing mean ± std for core metrics across all target tracks
# - Retrieval Sim: How well encoder retrieved similar reference music
# - CLAP Sim: Similarity between target and retrieved (CLAP evaluation)
# - Melody Acc: Melody matching between target and retrieved
# Note: Removed CE/CU/PC/PQ columns to reduce table width
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')

summary_data = []
for encoder in encoders:
    row = [encoder.upper()]

    # Retrieval similarity (encoder-specific)
    ret_sim = [eval_data[encoder][t]["retrieval_similarity"] for t in target_names]
    row.append(f"{np.mean(ret_sim):.3f} ± {np.std(ret_sim):.3f}")

    # CLAP similarity (target vs retrieved)
    clap_sim = [eval_data[encoder][t]["clap_similarity"] for t in target_names]
    row.append(f"{np.mean(clap_sim):.3f} ± {np.std(clap_sim):.3f}")

    # Melody accuracy (target vs retrieved)
    melody = [eval_data[encoder][t]["melody_accuracy"] for t in target_names]
    row.append(f"{np.mean(melody):.3f} ± {np.std(melody):.3f}")

    summary_data.append(row)

columns = ["Encoder", "Retrieval Sim", "CLAP Sim", "Melody Acc"]
table = ax.table(cellText=summary_data, colLabels=columns,
                cellLoc='center', loc='center',
                colColours=['lightgray']*len(columns))
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# No title for this image
plt.savefig(asset_dir / "task1_summary_statistics.png", bbox_inches='tight')
print(f"Saved: {asset_dir / 'task1_summary_statistics.png'}")
plt.close()

print("\n✓ All visualizations created successfully!")
print("Total images saved: 3 (only those used in report.md)")
print("  - task1_retrieval_similarity.png")
print("  - task1_performance_heatmap.png")
print("  - task1_summary_statistics.png")
print(f"Output directory: {asset_dir.absolute()}")
