"""Create a 2x3 grid of key poses for the paper figure."""
from PIL import Image

poses = [
    "Standing_arms_at_sides",
    "T-Pose",
    "Fetal_Position",
    "Dab",
    "Squat_Slav",
    "Coffin_Dance",
]

cols, rows = 3, 2
cell_w, cell_h = 600, 600
grid = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')

for i, name in enumerate(poses):
    path = f"results3d/pose_{name}.png"
    try:
        img = Image.open(path).resize((cell_w, cell_h))
        r, c = divmod(i, cols)
        grid.paste(img, (c * cell_w, r * cell_h))
    except Exception as e:
        print(f"Failed to load {path}: {e}")

grid.save("results3d/pose_selection_grid.png")
print("Saved results3d/pose_selection_grid.png")
