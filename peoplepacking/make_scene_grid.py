"""Create a 2x2 grid of interesting packing scenes for the paper."""
from PIL import Image

scenes = [
    ("results3d/scene_Elevator_Standing_arms_at_sides.png", "Elevator (Standing)"),
    ("results3d/scene_Elevator_Fetal_Position.png", "Elevator (Fetal)"),
    ("results3d/scene_Shipping_Container_Standing_arms_at_sides.png", "Container (Standing)"),
    ("results3d/scene_Shipping_Container_Squat_Slav.png", "Container (Squat)"),
]

cols, rows = 2, 2
cell_w, cell_h = 800, 600
grid = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')

for i, (path, label) in enumerate(scenes):
    try:
        img = Image.open(path).resize((cell_w, cell_h))
        r, c = divmod(i, cols)
        grid.paste(img, (c * cell_w, r * cell_h))
    except Exception as e:
        print(f"Failed to load {path}: {e}")

grid.save("results3d/packing_scenes_grid.png")
print("Saved results3d/packing_scenes_grid.png")
