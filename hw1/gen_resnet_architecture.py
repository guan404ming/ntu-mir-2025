"""
Generate detailed architecture diagram for ResNet CNN (No Pretrain)
Similar style to PANNs architecture diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")

# Title
ax.text(
    10,
    11.5,
    "ResNet-based CNN Architecture (No Pretrain)",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
)

# Color scheme
colors = {
    "input": "#E8F4FD",
    "preprocessing": "#D1ECF1",
    "conv": "#BEE5EB",
    "resblock": "#85C1E9",
    "pooling": "#5DADE2",
    "classifier": "#3498DB",
    "output": "#2E86AB",
    "details": "#1B4F72",
}


def draw_box(ax, x, y, width, height, text, color, fontsize=9, bold=True):
    """Draw a colored box with text"""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        multialignment="center",
    )


def draw_arrow(ax, x1, y1, x2, y2, style="->", lw=2.5):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        color="black",
        linewidth=lw,
        mutation_scale=20,
    )
    ax.add_patch(arrow)


def draw_detailed_resblock(ax, x, y, in_ch, out_ch, stride, num_blocks, label):
    """Draw a detailed residual block with internal structure"""
    box_height = 1.5
    box_width = 2.8

    # Main container
    container = FancyBboxPatch(
        (x - box_width / 2, y - box_height / 2 - 0.1),
        box_width,
        box_height + 0.2,
        boxstyle="round,pad=0.1",
        facecolor="white",
        edgecolor=colors["resblock"],
        linewidth=3,
        linestyle="--",
    )
    ax.add_patch(container)

    # Title
    ax.text(
        x,
        y + box_height / 2 + 0.3,
        label,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Internal structure (simplified view of residual block)
    small_box_w = 1.2
    small_box_h = 0.35

    # Conv1 + BN + ReLU
    draw_box(
        ax,
        x,
        y + 0.4,
        small_box_w,
        small_box_h,
        f"Conv 3×3\n{out_ch} ch",
        colors["conv"],
        fontsize=7,
    )

    # Conv2 + BN
    draw_box(
        ax,
        x,
        y,
        small_box_w,
        small_box_h,
        f"Conv 3×3\n{out_ch} ch",
        colors["conv"],
        fontsize=7,
    )

    # Shortcut (if stride > 1)
    if stride > 1:
        draw_box(
            ax,
            x,
            y - 0.4,
            small_box_w,
            small_box_h,
            f"Shortcut\nstride={stride}",
            colors["pooling"],
            fontsize=7,
        )
    else:
        ax.text(
            x,
            y - 0.4,
            f"×{num_blocks} blocks",
            ha="center",
            va="center",
            fontsize=8,
            style="italic",
        )


# ============================================================================
# Layer 1: Input
# ============================================================================
x_pos = 1.5
y_pos = 6

draw_box(
    ax,
    x_pos,
    y_pos,
    1.8,
    1.2,
    "Audio Input\n150s @ 16kHz\n(2,400,000 samples)",
    colors["input"],
    fontsize=9,
)

# ============================================================================
# Layer 2: Mel Spectrogram
# ============================================================================
x_pos += 2.5

draw_arrow(ax, x_pos - 1.6, y_pos, x_pos - 1.0, y_pos)

draw_box(
    ax,
    x_pos,
    y_pos,
    2.0,
    1.2,
    "Mel Spectrogram\n64 mels\nfmax=8kHz\nhop=512",
    colors["preprocessing"],
    fontsize=9,
)

# ============================================================================
# Layer 3: Initial Conv + MaxPool
# ============================================================================
x_pos += 2.8

draw_arrow(ax, x_pos - 2.0, y_pos, x_pos - 1.6, y_pos)

# Conv
draw_box(
    ax,
    x_pos,
    y_pos + 0.7,
    2.2,
    0.8,
    "Conv2d 7×7\n64 channels\nstride=2",
    colors["conv"],
    fontsize=8,
)

# BatchNorm + ReLU
draw_box(ax, x_pos, y_pos, 2.2, 0.5, "BatchNorm2d + ReLU", colors["conv"], fontsize=8)

# MaxPool
draw_box(
    ax,
    x_pos,
    y_pos - 0.7,
    2.2,
    0.6,
    "MaxPool2d 3×3\nstride=2",
    colors["pooling"],
    fontsize=8,
)

# ============================================================================
# Layer 4-6: Residual Blocks
# ============================================================================
x_pos += 3.2

draw_arrow(ax, x_pos - 2.3, y_pos - 0.7, x_pos - 1.5, y_pos + 1.5)

# ResBlock 1: 64 channels
draw_detailed_resblock(ax, x_pos, y_pos + 3.0, 64, 64, 1, 2, "ResBlock Layer 1")

draw_arrow(ax, x_pos, y_pos + 1.8, x_pos, y_pos + 1.2)

# ResBlock 2: 128 channels
draw_detailed_resblock(ax, x_pos, y_pos, 64, 128, 2, 2, "ResBlock Layer 2")

draw_arrow(ax, x_pos, y_pos - 1.2, x_pos, y_pos - 1.8)

# ResBlock 3: 256 channels
draw_detailed_resblock(ax, x_pos, y_pos - 3.0, 128, 256, 2, 2, "ResBlock Layer 3")

# ============================================================================
# Layer 7: Dual Pooling
# ============================================================================
x_pos += 3.2

# Arrow from ResBlock 3
draw_arrow(ax, x_pos - 1.5, y_pos - 3.0, x_pos - 1.0, y_pos + 0.5)

# Adaptive Average Pooling
draw_box(
    ax,
    x_pos,
    y_pos + 1.2,
    2.0,
    0.8,
    "Adaptive\nAvgPool2d(1×1)\n256-dim",
    colors["pooling"],
    fontsize=8,
)

# Adaptive Max Pooling
draw_box(
    ax,
    x_pos,
    y_pos - 1.2,
    2.0,
    0.8,
    "Adaptive\nMaxPool2d(1×1)\n256-dim",
    colors["pooling"],
    fontsize=8,
)

# Concatenate
x_pos += 2.5

draw_arrow(ax, x_pos - 1.5, y_pos + 1.2, x_pos - 0.8, y_pos)
draw_arrow(ax, x_pos - 1.5, y_pos - 1.2, x_pos - 0.8, y_pos)

draw_box(
    ax, x_pos, y_pos, 1.5, 0.8, "Concatenate\n512-dim", colors["classifier"], fontsize=9
)

# ============================================================================
# Layer 8-9: Classifier Head
# ============================================================================
x_pos += 2.3

draw_arrow(ax, x_pos - 1.5, y_pos, x_pos - 0.9, y_pos)

# Dropout 1
draw_box(
    ax, x_pos, y_pos + 1.5, 1.8, 0.5, "Dropout(0.5)", colors["classifier"], fontsize=8
)

draw_arrow(ax, x_pos, y_pos + 1.2, x_pos, y_pos + 0.9)

# Linear 1
draw_box(
    ax,
    x_pos,
    y_pos + 0.5,
    1.8,
    0.6,
    "Linear\n512→256",
    colors["classifier"],
    fontsize=8,
)

draw_arrow(ax, x_pos, y_pos + 0.2, x_pos, y_pos - 0.1)

# BatchNorm + ReLU
draw_box(
    ax,
    x_pos,
    y_pos - 0.5,
    1.8,
    0.6,
    "BatchNorm1d\n+ ReLU",
    colors["classifier"],
    fontsize=8,
)

draw_arrow(ax, x_pos, y_pos - 0.8, x_pos, y_pos - 1.1)

# Dropout 2
draw_box(
    ax, x_pos, y_pos - 1.5, 1.8, 0.5, "Dropout(0.3)", colors["classifier"], fontsize=8
)

# ============================================================================
# Layer 10: Output
# ============================================================================
x_pos += 2.3

draw_arrow(ax, x_pos - 1.4, y_pos - 1.5, x_pos - 1.0, y_pos)

# Final Linear
draw_box(
    ax, x_pos, y_pos + 0.6, 1.8, 0.8, "Linear\n256→20", colors["output"], fontsize=9
)

draw_arrow(ax, x_pos, y_pos + 0.2, x_pos, y_pos - 0.2)

# Output
draw_box(
    ax,
    x_pos,
    y_pos - 0.8,
    1.8,
    0.8,
    "Artist\nPrediction\n(20 classes)",
    colors["details"],
    fontsize=9,
)

# ============================================================================
# Add annotations
# ============================================================================

# Feature dimensions annotation
ax.text(1.5, 0.5, "Input: 2.4M samples", fontsize=8, style="italic", color="gray")
ax.text(4.0, 0.5, "Mel: 64×~4688", fontsize=8, style="italic", color="gray")
ax.text(7.2, 0.5, "After Conv+Pool: 64×~294", fontsize=8, style="italic", color="gray")
ax.text(10.4, 0.5, "After ResBlocks: 256×~18", fontsize=8, style="italic", color="gray")
ax.text(13.4, 0.5, "After Pooling: 512-dim", fontsize=8, style="italic", color="gray")
ax.text(17.7, 0.5, "Output: 20 logits", fontsize=8, style="italic", color="gray")

# Add legend for components
legend_x = 1
legend_y = 10.5
legend_spacing = 0.6

components = [
    ("Input/Preprocessing", colors["preprocessing"]),
    ("Convolution", colors["conv"]),
    ("Residual Block", colors["resblock"]),
    ("Pooling", colors["pooling"]),
    ("Classifier", colors["classifier"]),
    ("Output", colors["details"]),
]

ax.text(legend_x, legend_y + 0.3, "Components:", fontsize=10, fontweight="bold")

for i, (label, color) in enumerate(components):
    y = legend_y - i * legend_spacing
    rect = mpatches.Rectangle(
        (legend_x - 0.2, y - 0.15),
        0.3,
        0.3,
        facecolor=color,
        edgecolor="black",
        linewidth=1,
    )
    ax.add_patch(rect)
    ax.text(legend_x + 0.3, y, label, fontsize=8, va="center")

# Add model statistics
stats_x = 16
stats_y = 10.5
ax.text(stats_x, stats_y + 0.3, "Model Statistics:", fontsize=10, fontweight="bold")
ax.text(stats_x, stats_y - 0.3, "• Parameters: ~2.7M", fontsize=8)
ax.text(stats_x, stats_y - 0.6, "• Input: 150s audio", fontsize=8)
ax.text(stats_x, stats_y - 0.9, "• Batch size: 32", fontsize=8)
ax.text(stats_x, stats_y - 1.2, "• Training: Mixup + Label Smoothing", fontsize=8)
ax.text(stats_x, stats_y - 1.5, "• Optimizer: AdamW", fontsize=8)
ax.text(stats_x, stats_y - 1.8, "• LR Schedule: CosineAnnealing", fontsize=8)

plt.tight_layout()
plt.savefig(
    "assets/task2_architecture_diagram_resnet.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
print("✓ Architecture diagram saved to assets/task2_architecture_diagram_resnet.png")
plt.close()
