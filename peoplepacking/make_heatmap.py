"""Generate a venues x poses heatmap from existing packing data."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'text.color': '#333333',
    'savefig.facecolor': '#FAFAFA',
})

with open("results3d/venue_packing_3d.json") as f:
    data = json.load(f)

venues = list(data.keys())
poses = list(data[venues[0]].keys())

# Build matrix
matrix = np.array([[data[v][p] for p in poses] for v in venues], dtype=float)

# Normalize each row (venue) to 0-1 for better color contrast
row_max = matrix.max(axis=1, keepdims=True)
row_max[row_max == 0] = 1
norm_matrix = matrix / row_max

fig, ax = plt.subplots(figsize=(18, 8))

# Use a rich warm colormap
im = ax.imshow(norm_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# Annotate cells with raw counts
for i in range(len(venues)):
    for j in range(len(poses)):
        val = int(matrix[i, j])
        color = 'white' if norm_matrix[i, j] > 0.65 else '#333333'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=7.5,
                fontweight='bold', color=color)

ax.set_xticks(range(len(poses)))
ax.set_xticklabels(poses, rotation=55, ha='right', fontsize=9)
ax.set_yticks(range(len(venues)))
ax.set_yticklabels(venues, fontsize=10)

ax.set_title("Human Packing Capacity: Venues \u00d7 Poses\n(darker = closer to venue maximum)",
             fontsize=15, fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Fraction of venue maximum", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Add subtle gridlines
ax.set_xticks([x - 0.5 for x in range(1, len(poses))], minor=True)
ax.set_yticks([y - 0.5 for y in range(1, len(venues))], minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.tick_params(which='minor', size=0)

plt.tight_layout()
plt.savefig("results3d/heatmap_venues_poses.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved results3d/heatmap_venues_poses.png")
