"""Generate a venues x poses heatmap from existing packing data."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

fig, ax = plt.subplots(figsize=(16, 7))

im = ax.imshow(norm_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

# Annotate cells with raw counts
for i in range(len(venues)):
    for j in range(len(poses)):
        val = int(matrix[i, j])
        color = 'white' if norm_matrix[i, j] > 0.65 else 'black'
        ax.text(j, i, str(val), ha='center', va='center', fontsize=7,
                fontweight='bold', color=color)

ax.set_xticks(range(len(poses)))
ax.set_xticklabels(poses, rotation=55, ha='right', fontsize=8)
ax.set_yticks(range(len(venues)))
ax.set_yticklabels(venues, fontsize=9)

ax.set_title("Human Packing Capacity: Venues x Poses\n(darker = closer to venue maximum)", fontsize=13, pad=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Fraction of venue maximum", fontsize=9)

plt.tight_layout()
plt.savefig("results3d/heatmap_venues_poses.png", dpi=200, bbox_inches='tight')
plt.close()
print("Saved results3d/heatmap_venues_poses.png")
