#!/usr/bin/env python3
"""
Generate visualizations for Task 2 (Music Generation) results
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# Configure font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load results
results_dir = Path("results")
modes = ["simple", "medium", "strong"]

# Load all evaluation results
all_results = {}
for mode in modes:
    with open(results_dir / f"evaluation_results_{mode}.json") as f:
        all_results[mode] = json.load(f)

# Prepare data
track_names = {
    "10_country_114_beat_4-4.wav": "Country 114 BPM",
    "4_jazz_120_beat_3-4.wav": "Jazz 120 BPM",
    "6_rock_102_beat_3-4.wav": "Rock 102 BPM",
    "Hedwig's theme x dizi ｜from Harry Potter ｜竹笛也能施魔法！Bamboo flute_60s.mp3": "Hedwig (Dizi)",
    "Mussorgsky： Pictures at an Exhibition (Pletnev, Andsnes)_60s.mp3": "Mussorgsky",
    "Spirited Away OST「Always With Me ⧸ Itsumo Nando Demo」Ru's Piano Cover [Sheet Music]_60s.mp3": "Spirited Away",
    "【楽譜あり】IRIS OUT⧸米津玄師（ピアノソロ上級）劇場版『チェンソーマン レゼ篇』主題歌【ピアノアレンジ楽譜】.mp3": "IRIS OUT",
    "【菊花台-周杰倫（D調）】附伴奏⧸鋼琴伴奏(竹笛Bamboo flute、Roland Aerophone AE-10) 演奏：蘇俊琪(PSR-S970)audio-technica AT-2035_60s.mp3": "菊花台",
    "竹笛｜这世界那么多人_cover 莫文蔚_60s.mp3": "莫文蔚"
}

# Prepare data
metrics_data = {mode: {
    'CLAP Similarity': [],
    'Melody Accuracy': [],
    'CE': [],
    'CU': [],
    'PC': [],
    'PQ': []
} for mode in modes}

for mode in modes:
    for track, data in all_results[mode].items():
        metrics_data[mode]['CLAP Similarity'].append(data['clap_similarity'])
        metrics_data[mode]['Melody Accuracy'].append(data['melody_accuracy'])
        metrics_data[mode]['CE'].append(data['aesthetics']['ce'])
        metrics_data[mode]['CU'].append(data['aesthetics']['cu'])
        metrics_data[mode]['PC'].append(data['aesthetics']['pc'])
        metrics_data[mode]['PQ'].append(data['aesthetics']['pq'])

# Calculate averages
avg_metrics = {}
for mode in modes:
    avg_metrics[mode] = {k: np.mean(v) for k, v in metrics_data[mode].items()}

# 1a. Mode Comparison - Bar Charts
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Plot 1: CLAP and Melody Comparison
x = np.arange(len(modes))
width = 0.35

ax = axes[0]
clap_avgs = [avg_metrics[m]['CLAP Similarity'] for m in modes]
melody_avgs = [avg_metrics[m]['Melody Accuracy'] for m in modes]

ax.bar(x - width/2, clap_avgs, width, label='CLAP Similarity', color='skyblue')
ax.bar(x + width/2, melody_avgs, width, label='Melody Accuracy', color='coral')
ax.set_xlabel('Mode')
ax.set_ylabel('Score')
ax.set_title('CLAP Similarity and Melody Accuracy by Mode')
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in modes])
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Aesthetics Comparison
ax = axes[1]
aesthetics_metrics = ['CE', 'CU', 'PC', 'PQ']
x = np.arange(len(aesthetics_metrics))
width = 0.25

for i, mode in enumerate(modes):
    values = [avg_metrics[mode][m] for m in aesthetics_metrics]
    ax.bar(x + i*width, values, width, label=mode.capitalize())

ax.set_xlabel('Aesthetics Metric')
ax.set_ylabel('Score')
ax.set_title('Meta Audiobox Aesthetics by Mode')
ax.set_xticks(x + width)
ax.set_xticklabels(aesthetics_metrics)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
Path('../assets').mkdir(exist_ok=True)
plt.savefig('../assets/task2_mode_comparison_metrics.png', dpi=150, bbox_inches='tight')
print("Saved: ../assets/task2_mode_comparison_metrics.png")
plt.close()

# 1b. Mode Comparison - Heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot 3: Per-Track CLAP Similarity Heatmap
ax = axes[0]
clap_matrix = []
track_labels = []
for track_id in all_results['simple'].keys():
    track_labels.append(track_names.get(track_id, track_id[:20]))
    row = [all_results[mode][track_id]['clap_similarity'] for mode in modes]
    clap_matrix.append(row)

clap_matrix = np.array(clap_matrix)
im = ax.imshow(clap_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels([m.capitalize() for m in modes])
ax.set_yticks(range(len(track_labels)))
ax.set_yticklabels(track_labels, fontsize=8)
ax.set_title('CLAP Similarity Heatmap')
plt.colorbar(im, ax=ax)

# Add text annotations
for i in range(len(track_labels)):
    for j in range(len(modes)):
        text = ax.text(j, i, f'{clap_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=7)

# Plot 4: Per-Track Melody Accuracy Heatmap
ax = axes[1]
melody_matrix = []
for track_id in all_results['simple'].keys():
    row = [all_results[mode][track_id]['melody_accuracy'] for mode in modes]
    melody_matrix.append(row)

melody_matrix = np.array(melody_matrix)
im = ax.imshow(melody_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels([m.capitalize() for m in modes])
ax.set_yticks(range(len(track_labels)))
ax.set_yticklabels(track_labels, fontsize=8)
ax.set_title('Melody Accuracy Heatmap')
plt.colorbar(im, ax=ax)

# Add text annotations
for i in range(len(track_labels)):
    for j in range(len(modes)):
        text = ax.text(j, i, f'{melody_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=7)

plt.tight_layout()
plt.savefig('../assets/task2_mode_comparison_heatmaps.png', dpi=150, bbox_inches='tight')
print("Saved: ../assets/task2_mode_comparison_heatmaps.png")
plt.close()

# 2. Detailed Performance Comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Create DataFrame for easy plotting
data_rows = []
for mode in modes:
    for track_id, results in all_results[mode].items():
        track_name = track_names.get(track_id, track_id[:20])
        data_rows.append({
            'Track': track_name,
            'Mode': mode.capitalize(),
            'CLAP': results['clap_similarity'],
            'Melody': results['melody_accuracy'],
            'CE': results['aesthetics']['ce'],
            'CU': results['aesthetics']['cu'],
            'PC': results['aesthetics']['pc'],
            'PQ': results['aesthetics']['pq']
        })

df = pd.DataFrame(data_rows)

# Plot 1: CLAP Similarity by Track
ax = axes[0]
pivot_clap = df.pivot(index='Track', columns='Mode', values='CLAP')
pivot_clap.plot(kind='barh', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_xlabel('CLAP Similarity')
ax.set_ylabel('Track')
ax.set_title('CLAP Similarity: Generated vs Target')
ax.legend(title='Mode', loc='lower right')
ax.grid(True, alpha=0.3)

# Plot 2: Melody Accuracy by Track
ax = axes[1]
pivot_melody = df.pivot(index='Track', columns='Mode', values='Melody')
pivot_melody.plot(kind='barh', ax=ax, color=['#3498db', '#e74c3c', '#2ecc71'])
ax.set_xlabel('Melody Accuracy')
ax.set_ylabel('Track')
ax.set_title('Melody Accuracy: Generated vs Target')
ax.legend(title='Mode', loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/task2_performance_by_track.png', dpi=150, bbox_inches='tight')
print("Saved: ../assets/task2_performance_by_track.png")
plt.close()

# 3. Summary Statistics
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Calculate summary statistics
summary_data = []
for mode in modes:
    mode_label = mode.capitalize()
    summary_data.append({
        'Mode': mode_label,
        'Metric': 'CLAP Similarity',
        'Mean': np.mean(metrics_data[mode]['CLAP Similarity']),
        'Std': np.std(metrics_data[mode]['CLAP Similarity']),
        'Min': np.min(metrics_data[mode]['CLAP Similarity']),
        'Max': np.max(metrics_data[mode]['CLAP Similarity'])
    })
    summary_data.append({
        'Mode': mode_label,
        'Metric': 'Melody Accuracy',
        'Mean': np.mean(metrics_data[mode]['Melody Accuracy']),
        'Std': np.std(metrics_data[mode]['Melody Accuracy']),
        'Min': np.min(metrics_data[mode]['Melody Accuracy']),
        'Max': np.max(metrics_data[mode]['Melody Accuracy'])
    })
    for aesthetic in ['CE', 'CU', 'PC', 'PQ']:
        summary_data.append({
            'Mode': mode_label,
            'Metric': aesthetic,
            'Mean': np.mean(metrics_data[mode][aesthetic]),
            'Std': np.std(metrics_data[mode][aesthetic]),
            'Min': np.min(metrics_data[mode][aesthetic]),
            'Max': np.max(metrics_data[mode][aesthetic])
        })

summary_df = pd.DataFrame(summary_data)

# Create grouped bar chart
metrics_order = ['CLAP Similarity', 'Melody Accuracy', 'CE', 'CU', 'PC', 'PQ']
x = np.arange(len(metrics_order))
width = 0.25

for i, mode in enumerate(['Simple', 'Medium', 'Strong']):
    mode_data = summary_df[summary_df['Mode'] == mode]
    means = [mode_data[mode_data['Metric'] == m]['Mean'].values[0] for m in metrics_order]
    stds = [mode_data[mode_data['Metric'] == m]['Std'].values[0] for m in metrics_order]

    ax.bar(x + i*width, means, width, yerr=stds, label=mode,
           capsize=5, alpha=0.8)

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Summary Statistics Across All Modes')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_order)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/task2_summary_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: ../assets/task2_summary_statistics.png")
plt.close()

# 4. CFG Scale Impact (Simple vs Medium vs Strong)
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

# Get CFG scales
cfg_scales = {mode: all_results[mode][list(all_results[mode].keys())[0]]['cfg_scale'] for mode in modes}

# Plot distributions
ax = axes[0, 0]
for mode in modes:
    clap_values = [all_results[mode][track]['clap_similarity'] for track in all_results[mode].keys()]
    ax.hist(clap_values, alpha=0.5, label=f'{mode.capitalize()} (CFG={cfg_scales[mode]})', bins=10)
ax.set_xlabel('CLAP Similarity')
ax.set_ylabel('Frequency')
ax.set_title('CLAP Similarity Distribution by Mode')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for mode in modes:
    melody_values = [all_results[mode][track]['melody_accuracy'] for track in all_results[mode].keys()]
    ax.hist(melody_values, alpha=0.5, label=f'{mode.capitalize()} (CFG={cfg_scales[mode]})', bins=10)
ax.set_xlabel('Melody Accuracy')
ax.set_ylabel('Frequency')
ax.set_title('Melody Accuracy Distribution by Mode')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for mode in modes:
    ce_values = [all_results[mode][track]['aesthetics']['ce'] for track in all_results[mode].keys()]
    cu_values = [all_results[mode][track]['aesthetics']['cu'] for track in all_results[mode].keys()]
    ax.scatter(ce_values, cu_values, alpha=0.6, label=f'{mode.capitalize()} (CFG={cfg_scales[mode]})', s=80)
ax.set_xlabel('Content Enjoyment (CE)')
ax.set_ylabel('Content Usefulness (CU)')
ax.set_title('Content Enjoyment vs Usefulness')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for mode in modes:
    pc_values = [all_results[mode][track]['aesthetics']['pc'] for track in all_results[mode].keys()]
    pq_values = [all_results[mode][track]['aesthetics']['pq'] for track in all_results[mode].keys()]
    ax.scatter(pc_values, pq_values, alpha=0.6, label=f'{mode.capitalize()} (CFG={cfg_scales[mode]})', s=80)
ax.set_xlabel('Production Complexity (PC)')
ax.set_ylabel('Production Quality (PQ)')
ax.set_title('Production Complexity vs Quality')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../assets/task2_cfg_impact.png', dpi=150, bbox_inches='tight')
print("Saved: ../assets/task2_cfg_impact.png")
plt.close()

print("\nAll visualizations created successfully!")
print(f"\nSummary Statistics:")
print(summary_df.to_string(index=False))
