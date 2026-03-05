"""
OPTIMAL ANTHROPOMORPHIC POLYGON PACKING:
A Rigorous Analysis of Human Stacking Configurations

Experiment code for generating human silhouettes as 2D polygons,
running packing algorithms across poses, and comparing efficiency.

Now with 3D volumetric packing for realistic space utilization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import rotate, translate, scale
from shapely import unary_union
import json
import os
from itertools import product

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# PART 1: Human silhouette generation as polygons
# ============================================================

def make_limb(cx, cy, w, h):
    """Rectangle centered at (cx, cy) with width w, height h."""
    return box(cx - w/2, cy - h/2, cx + w/2, cy + h/2)

def human_standing(scale_factor=1.0):
    """Standing human, arms at sides. ~1.75m tall, ~0.45m wide."""
    # Head
    head = Polygon([(-.08, 1.58), (.08, 1.58), (.10, 1.75), (.08, 1.82),
                     (-.08, 1.82), (-.10, 1.75)])
    # Torso
    torso = box(-.18, 0.85, .18, 1.58)
    # Left arm
    l_arm = box(-.25, 0.85, -.18, 1.45)
    # Right arm
    r_arm = box(.18, 0.85, .25, 1.45)
    # Left leg
    l_leg = box(-.16, 0.0, -.02, 0.85)
    # Right leg
    r_leg = box(.02, 0.0, .16, 0.85)

    body = unary_union([head, torso, l_arm, r_arm, l_leg, r_leg])
    if scale_factor != 1.0:
        body = scale(body, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    return body

def human_tpose(scale_factor=1.0):
    """The dreaded T-pose. Maximum bounding box waste."""
    head = Polygon([(-.08, 1.58), (.08, 1.58), (.10, 1.75), (.08, 1.82),
                     (-.08, 1.82), (-.10, 1.75)])
    torso = box(-.18, 0.85, .18, 1.58)
    # Arms extended horizontally
    l_arm = box(-.75, 1.30, -.18, 1.40)
    r_arm = box(.18, 1.30, .75, 1.40)
    l_leg = box(-.16, 0.0, -.02, 0.85)
    r_leg = box(.02, 0.0, .16, 0.85)

    body = unary_union([head, torso, l_arm, r_arm, l_leg, r_leg])
    if scale_factor != 1.0:
        body = scale(body, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    return body

def human_fetal(scale_factor=1.0):
    """Fetal position. The hypothesized optimal packing pose."""
    # Curled up ball-ish shape
    head = Polygon([(-.08, 0.55), (.08, 0.55), (.10, 0.72), (.08, 0.78),
                     (-.08, 0.78), (-.10, 0.72)])
    # Torso curled - more like a C shape
    torso = Polygon([(-.20, 0.15), (.15, 0.10), (.20, 0.30), (.18, 0.55),
                      (-.08, 0.55), (-.22, 0.40)])
    # Legs tucked
    legs = Polygon([(-.20, 0.15), (.15, 0.10), (.20, 0.0), (.10, -0.08),
                     (-.15, -0.05), (-.25, 0.05)])
    # Arms wrapped
    arms = Polygon([(-.22, 0.25), (-.28, 0.35), (-.25, 0.50), (-.20, 0.50),
                     (-.20, 0.30)])

    body = unary_union([head, torso, legs, arms])
    if scale_factor != 1.0:
        body = scale(body, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    return body

def human_star(scale_factor=1.0):
    """Star / spread eagle. The absolute worst packing pose."""
    head = Polygon([(-.08, 1.58), (.08, 1.58), (.10, 1.75), (.08, 1.82),
                     (-.08, 1.82), (-.10, 1.75)])
    torso = box(-.18, 0.85, .18, 1.58)
    # Arms extended diagonally up
    l_arm = Polygon([(-.18, 1.45), (-.70, 1.70), (-.65, 1.78), (-.18, 1.55)])
    r_arm = Polygon([(.18, 1.45), (.70, 1.70), (.65, 1.78), (.18, 1.55)])
    # Legs spread
    l_leg = Polygon([(-.16, 0.85), (-.02, 0.85), (-.15, 0.0), (-.50, -0.10), (-.55, 0.0)])
    r_leg = Polygon([(.16, 0.85), (.02, 0.85), (.15, 0.0), (.50, -0.10), (.55, 0.0)])

    body = unary_union([head, torso, l_arm, r_arm, l_leg, r_leg])
    if scale_factor != 1.0:
        body = scale(body, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    return body

def human_pike(scale_factor=1.0):
    """Pike position - folded in half, touching toes. Surprisingly compact."""
    # Head near knees
    head = Polygon([(-.08, 0.45), (.08, 0.45), (.10, 0.62), (.08, 0.68),
                     (-.08, 0.68), (-.10, 0.62)])
    # Torso bent forward horizontally
    torso = Polygon([(-.18, 0.45), (.18, 0.45), (.18, 0.85), (.15, 0.90),
                      (-.15, 0.90), (-.18, 0.85)])
    # Legs vertical
    l_leg = box(-.16, 0.0, -.02, 0.50)
    r_leg = box(.02, 0.0, .16, 0.50)
    # Arms reaching down
    l_arm = box(-.12, -0.05, -.04, 0.45)
    r_arm = box(.04, -0.05, .12, 0.45)

    body = unary_union([head, torso, l_leg, r_leg, l_arm, r_arm])
    if scale_factor != 1.0:
        body = scale(body, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    return body

def human_superman(scale_factor=1.0):
    """Arms extended overhead, legs together. Diving position."""
    head = Polygon([(-.08, 1.58), (.08, 1.58), (.10, 1.75), (.08, 1.82),
                     (-.08, 1.82), (-.10, 1.75)])
    torso = box(-.18, 0.85, .18, 1.58)
    # Arms straight up
    l_arm = box(-.12, 1.75, -.04, 2.20)
    r_arm = box(.04, 1.75, .12, 2.20)
    # Legs together
    l_leg = box(-.12, 0.0, -.02, 0.85)
    r_leg = box(.02, 0.0, .12, 0.85)

    body = unary_union([head, torso, l_arm, r_arm, l_leg, r_leg])
    if scale_factor != 1.0:
        body = scale(body, xfact=scale_factor, yfact=scale_factor, origin=(0,0))
    return body


POSES = {
    "Standing (arms at sides)": human_standing,
    "T-Pose": human_tpose,
    "Fetal Position": human_fetal,
    "Star / Spread Eagle": human_star,
    "Pike (folded)": human_pike,
    "Superman (arms up)": human_superman,
}

# 3D depth estimates per pose (meters) - the "thickness" of a human in the
# axis perpendicular to the 2D silhouette. Based on average adult dimensions.
# Front-view silhouettes: depth = front-to-back (chest depth ~0.25m)
# Side-view / lying: depth = shoulder width (~0.45m)
# Fetal: roughly spherical, depth ~ max lateral extent
POSE_DEPTHS = {
    "Standing (arms at sides)": 0.30,   # chest depth, standing upright
    "T-Pose": 0.30,                     # same, arms don't add depth
    "Fetal Position": 0.45,             # curled on side, shoulder width
    "Star / Spread Eagle": 0.30,        # flat on back, chest depth
    "Pike (folded)": 0.35,              # bent over, slightly wider
    "Superman (arms up)": 0.30,         # streamlined, chest depth
}

# For "lying down" scenarios, the silhouette is viewed from above (top-down),
# and the depth is the person's actual height dimension compressed into
# the vertical axis. We model two viewing orientations:
# FRONT VIEW: silhouette is front-facing, depth = chest-to-back
# TOP VIEW: silhouette is top-down, depth = height of pose

POSE_ORIENTATIONS = {
    # (2D silhouette view, depth) for each packing orientation
    # "upright": person standing/posed, packed side by side on the floor
    # "stacked": person lying down, viewed from above, stacked vertically
    "Standing (arms at sides)": {"upright_depth": 0.30, "lying_depth": 1.82},
    "T-Pose":                   {"upright_depth": 0.30, "lying_depth": 1.82},
    "Fetal Position":           {"upright_depth": 0.45, "lying_depth": 0.86},
    "Star / Spread Eagle":      {"upright_depth": 0.30, "lying_depth": 1.92},
    "Pike (folded)":            {"upright_depth": 0.35, "lying_depth": 0.95},
    "Superman (arms up)":       {"upright_depth": 0.30, "lying_depth": 2.20},
}


# ============================================================
# PART 2: Packing metrics
# ============================================================

def packing_efficiency(polygon):
    """Ratio of polygon area to bounding box area. Higher = more packable."""
    minx, miny, maxx, maxy = polygon.bounds
    bb_area = (maxx - minx) * (maxy - miny)
    if bb_area == 0:
        return 0
    return polygon.area / bb_area

def convex_ratio(polygon):
    """Ratio of polygon area to convex hull area. 1.0 = already convex."""
    hull_area = polygon.convex_hull.area
    if hull_area == 0:
        return 0
    return polygon.area / hull_area

def aspect_ratio(polygon):
    """Aspect ratio of bounding box (always >= 1)."""
    minx, miny, maxx, maxy = polygon.bounds
    w = maxx - minx
    h = maxy - miny
    if min(w, h) == 0:
        return float('inf')
    return max(w, h) / min(w, h)

def best_rotation_efficiency(polygon, steps=36):
    """Find the rotation angle that maximizes bounding-box packing efficiency."""
    best_eff = 0
    best_angle = 0
    for angle in np.linspace(0, 180, steps, endpoint=False):
        rotated = rotate(polygon, angle, origin='centroid')
        eff = packing_efficiency(rotated)
        if eff > best_eff:
            best_eff = eff
            best_angle = angle
    return best_angle, best_eff


# ============================================================
# PART 3: Grid packing simulation
# ============================================================

def pack_grid(polygon, container_w, container_h, allow_rotation=True):
    """
    Pack copies of polygon into a rectangular container using axis-aligned grid.
    If allow_rotation, try 0 and 90 degrees and pick whichever fits more.
    Returns (count, placed_polygons).
    """
    def try_pack(poly):
        placed = []
        minx, miny, maxx, maxy = poly.bounds
        pw = maxx - minx
        ph = maxy - miny
        # Shift so polygon starts at origin
        shifted = translate(poly, -minx, -miny)

        nx = int(container_w // pw)
        ny = int(container_h // ph)

        for ix in range(nx):
            for iy in range(ny):
                placed.append(translate(shifted, ix * pw, iy * ph))
        return len(placed), placed

    count0, placed0 = try_pack(polygon)

    if allow_rotation:
        rot90 = rotate(polygon, 90, origin='centroid')
        count90, placed90 = try_pack(rot90)
        if count90 > count0:
            return count90, placed90

    return count0, placed0

def pack_grid_with_nesting(polygon, container_w, container_h, flip=True):
    """
    More sophisticated packing: try alternating rotations (0/180) for
    nesting irregular shapes. Also try flipping (mirroring).
    Returns (count, placed_polygons).
    """
    best_count = 0
    best_placed = []

    rotations_to_try = [0, 90, 180, 270]

    for base_rot in rotations_to_try:
        poly = rotate(polygon, base_rot, origin='centroid')

        # Try alternating rows with 180-degree rotation for nesting
        minx, miny, maxx, maxy = poly.bounds
        pw = maxx - minx
        ph = maxy - miny
        if pw == 0 or ph == 0:
            continue

        shifted = translate(poly, -minx, -miny)
        flipped = rotate(shifted, 180, origin=(pw/2, ph/2))

        nx = int(container_w // pw)
        ny = int(container_h // ph)

        placed = []
        for ix in range(nx):
            for iy in range(ny):
                if (ix + iy) % 2 == 0:
                    p = translate(shifted, ix * pw, iy * ph)
                else:
                    p = translate(flipped, ix * pw, iy * ph)
                placed.append(p)

        if len(placed) > best_count:
            best_count = len(placed)
            best_placed = placed

    return best_count, best_placed


def pack_with_optimal_rotation(polygon, container_w, container_h, steps=36):
    """
    Try all rotation angles and pick the one that fits the most copies.
    """
    best_count = 0
    best_placed = []
    best_angle = 0

    for angle in np.linspace(0, 180, steps, endpoint=False):
        rotated = rotate(polygon, angle, origin='centroid')
        count, placed = pack_grid(rotated, container_w, container_h, allow_rotation=False)
        if count > best_count:
            best_count = count
            best_placed = placed
            best_angle = angle

    return best_count, best_placed, best_angle


# ============================================================
# PART 4: The T-Pose Tax
# ============================================================

def tpose_tax(pose_polygon, tpose_polygon):
    """
    The T-Pose Tax: ratio of bounding box area wasted by T-pose vs this pose.
    Formally: (BB_tpose / A_tpose) / (BB_pose / A_pose)
    i.e., how many times worse is T-pose at filling its bounding box.
    """
    tpose_eff = packing_efficiency(tpose_polygon)
    pose_eff = packing_efficiency(pose_polygon)
    if pose_eff == 0:
        return float('inf')
    return (1/tpose_eff) / (1/pose_eff)  # = pose_eff / tpose_eff


# ============================================================
# PART 5: Visualization
# ============================================================

def plot_single_pose(name, polygon, ax):
    """Plot a single human silhouette."""
    if isinstance(polygon, MultiPolygon):
        for geom in polygon.geoms:
            x, y = geom.exterior.xy
            ax.fill(x, y, alpha=0.6, fc='steelblue', ec='navy', linewidth=1.5)
    else:
        x, y = polygon.exterior.xy
        ax.fill(x, y, alpha=0.6, fc='steelblue', ec='navy', linewidth=1.5)

    # Draw bounding box
    minx, miny, maxx, maxy = polygon.bounds
    bb = plt.Rectangle((minx, miny), maxx-minx, maxy-miny,
                         fill=False, ec='red', linestyle='--', linewidth=1)
    ax.add_patch(bb)

    eff = packing_efficiency(polygon)
    ax.set_title(f"{name}\nBB Efficiency: {eff:.1%}", fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

def plot_all_poses():
    """Generate the pose comparison figure."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    fig.suptitle("Human Pose Bounding-Box Packing Efficiency", fontsize=14, fontweight='bold')

    for ax, (name, func) in zip(axes.flat, POSES.items()):
        poly = func()
        plot_single_pose(name, poly, ax)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pose_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pose_comparison.png")

def plot_packing(name, placed, container_w, container_h, count):
    """Visualize a packing arrangement."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = plt.cm.Set3(np.linspace(0, 1, min(count, 12)))

    for i, poly in enumerate(placed):
        color = colors[i % len(colors)]
        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, alpha=0.6, fc=color, ec='black', linewidth=0.5)
        else:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.6, fc=color, ec='black', linewidth=0.5)

    # Container outline
    rect = plt.Rectangle((0, 0), container_w, container_h,
                           fill=False, ec='black', linewidth=2)
    ax.add_patch(rect)

    total_area = sum(p.area for p in placed)
    container_area = container_w * container_h
    density = total_area / container_area if container_area > 0 else 0

    ax.set_title(f"{name}: {count} humans packed\nDensity: {density:.1%}", fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-0.1, container_w + 0.1)
    ax.set_ylim(-0.1, container_h + 0.1)

    safe_name = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    plt.savefig(f"{OUTPUT_DIR}/packing_{safe_name}.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# PART 6: Run experiments
# ============================================================

def experiment_1_pose_metrics():
    """Compare all poses on basic metrics."""
    print("=" * 60)
    print("EXPERIMENT 1: Pose Packing Metrics")
    print("=" * 60)

    results = {}
    tpose_poly = POSES["T-Pose"]()

    for name, func in POSES.items():
        poly = func()
        eff = packing_efficiency(poly)
        cr = convex_ratio(poly)
        ar = aspect_ratio(poly)
        best_angle, best_eff = best_rotation_efficiency(poly)
        tax = tpose_tax(poly, tpose_poly)

        results[name] = {
            "area": float(poly.area),
            "bb_efficiency": float(eff),
            "convex_ratio": float(cr),
            "aspect_ratio": float(ar),
            "best_rotation_angle": float(best_angle),
            "best_rotation_efficiency": float(best_eff),
            "tpose_tax": float(tax),
        }

        print(f"\n{name}:")
        print(f"  Body area:           {poly.area:.4f} m²")
        print(f"  BB efficiency:       {eff:.1%}")
        print(f"  Convex ratio:        {cr:.1%}")
        print(f"  Aspect ratio:        {ar:.2f}")
        print(f"  Best rotation:       {best_angle:.0f}° -> {best_eff:.1%}")
        print(f"  T-Pose Tax:          {tax:.2f}x")

    # Save results
    with open(f"{OUTPUT_DIR}/pose_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_all_poses()
    return results


def experiment_2_elevator_packing():
    """
    How many humans fit in a standard elevator?
    Standard elevator interior: ~2.0m wide x 1.5m deep x 2.4m tall
    Now with FULL 3D volumetric analysis.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Elevator Packing Optimization (3D)")
    print("=" * 60)

    ELEVATOR_W = 2.0   # meters (width)
    ELEVATOR_D = 1.5   # meters (depth)
    ELEVATOR_H = 2.4   # meters (ceiling height)

    print(f"  Elevator dimensions: {ELEVATOR_W}m x {ELEVATOR_D}m x {ELEVATOR_H}m")
    print(f"  Volume: {ELEVATOR_W * ELEVATOR_D * ELEVATOR_H:.1f} m^3")

    results = {}

    for name, func in POSES.items():
        poly = func()
        orient = POSE_ORIENTATIONS[name]

        # Strategy A: People standing upright, packed in floor plan (2D)
        # Floor area = W x D, silhouette packed into floor as top-down projection
        # Each person occupies their bounding box footprint x upright_depth
        upright_depth = orient["upright_depth"]
        minx, miny, maxx, maxy = poly.bounds
        poly_w = maxx - minx  # shoulder width from silhouette
        poly_h = maxy - miny  # height from silhouette

        # Standing upright: pack shoulder-width x depth into W x D floor
        # Only 1 layer vertically (person is already full height)
        n_across = int(ELEVATOR_W // poly_w)
        n_deep = int(ELEVATOR_D // upright_depth)
        count_upright = n_across * n_deep

        # Strategy B: People lying down, stacked in layers
        # Silhouette is the top-down view, pack into W x D floor
        # Stack layers using lying_depth as layer height
        lying_depth = orient["lying_depth"]
        count_per_layer_opt, placed_layer, _ = pack_with_optimal_rotation(
            poly, ELEVATOR_W, ELEVATOR_D, steps=72)
        count_per_layer_nest, placed_nest = pack_grid_with_nesting(
            poly, ELEVATOR_W, ELEVATOR_D)
        count_per_layer = max(count_per_layer_opt, count_per_layer_nest)
        best_layer_placed = placed_layer if count_per_layer_opt >= count_per_layer_nest else placed_nest

        n_layers = int(ELEVATOR_H // lying_depth)
        count_stacked = count_per_layer * n_layers

        # Strategy C: People lying on their side, stacked
        # Use the 2D silhouette (front view) packed into W x H wall,
        # then stack rows of depth into the D dimension
        count_sidepack, placed_side, _ = pack_with_optimal_rotation(
            poly, ELEVATOR_W, ELEVATOR_H, steps=72)
        n_rows_deep = int(ELEVATOR_D // upright_depth)
        count_side = count_sidepack * n_rows_deep

        best_count = max(count_upright, count_stacked, count_side)
        if best_count == count_stacked:
            method = f"stacked ({n_layers} layers)"
            best_placed = best_layer_placed
            container_dims = (ELEVATOR_W, ELEVATOR_D)
        elif best_count == count_side:
            method = f"side-packed ({n_rows_deep} rows deep)"
            best_placed = placed_side
            container_dims = (ELEVATOR_W, ELEVATOR_H)
        else:
            method = "standing upright"
            # Generate placed polygons for visualization
            _, best_placed = pack_grid(poly, ELEVATOR_W, ELEVATOR_D)
            container_dims = (ELEVATOR_W, ELEVATOR_D)

        results[name] = {
            "count_upright": count_upright,
            "count_stacked": count_stacked,
            "n_layers": n_layers,
            "count_per_layer": count_per_layer,
            "count_side_packed": count_side,
            "best_count": best_count,
            "best_method": method,
        }

        print(f"\n{name}:")
        print(f"  Standing upright:    {count_upright} humans")
        print(f"  Stacked lying down:  {count_stacked} humans ({count_per_layer}/layer x {n_layers} layers)")
        print(f"  Side-packed:         {count_side} humans ({count_sidepack}/slice x {n_rows_deep} rows)")
        print(f"  BEST:                {best_count} humans ({method})")

        plot_packing(f"Elevator - {name}", best_placed,
                     container_dims[0], container_dims[1], best_count)

    with open(f"{OUTPUT_DIR}/elevator_packing.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_3_container_shipping():
    """
    How many humans fit in a standard 20ft shipping container?
    Interior: ~5.9m long x 2.35m wide x 2.39m tall
    Full 3D volumetric packing.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Shipping Container Packing (3D)")
    print("(For purely academic purposes)")
    print("=" * 60)

    CONTAINER_L = 5.9    # length
    CONTAINER_W = 2.35   # width
    CONTAINER_H = 2.39   # height
    CONTAINER_VOL = CONTAINER_L * CONTAINER_W * CONTAINER_H

    print(f"  Container: {CONTAINER_L}m x {CONTAINER_W}m x {CONTAINER_H}m")
    print(f"  Volume: {CONTAINER_VOL:.1f} m^3")

    results = {}

    for name, func in POSES.items():
        poly = func()
        orient = POSE_ORIENTATIONS[name]
        upright_depth = orient["upright_depth"]
        lying_depth = orient["lying_depth"]

        # Strategy A: Standing upright in container
        minx, miny, maxx, maxy = poly.bounds
        poly_w = maxx - minx
        n_along = int(CONTAINER_L // poly_w)
        n_across = int(CONTAINER_W // upright_depth)
        count_upright = n_along * n_across

        # Strategy B: Lying down, stacked in layers (top-down packing into L x W)
        count_per_layer_opt, placed_opt, _ = pack_with_optimal_rotation(
            poly, CONTAINER_L, CONTAINER_W, steps=72)
        count_per_layer_nest, placed_nest = pack_grid_with_nesting(
            poly, CONTAINER_L, CONTAINER_W)
        count_per_layer = max(count_per_layer_opt, count_per_layer_nest)
        best_layer_placed = placed_opt if count_per_layer_opt >= count_per_layer_nest else placed_nest
        n_layers = int(CONTAINER_H // lying_depth)
        count_stacked = count_per_layer * n_layers

        # Strategy C: Side-packed (silhouette into L x H wall, rows into W)
        count_side_opt, placed_side, _ = pack_with_optimal_rotation(
            poly, CONTAINER_L, CONTAINER_H, steps=72)
        n_rows = int(CONTAINER_W // upright_depth)
        count_side = count_side_opt * n_rows

        # Strategy D: End-packed (silhouette into W x H wall, rows into L)
        count_end_opt, placed_end, _ = pack_with_optimal_rotation(
            poly, CONTAINER_W, CONTAINER_H, steps=72)
        n_rows_long = int(CONTAINER_L // upright_depth)
        count_end = count_end_opt * n_rows_long

        best_count = max(count_upright, count_stacked, count_side, count_end)
        if best_count == count_stacked:
            method = f"stacked ({n_layers} layers x {count_per_layer}/layer)"
            best_placed = best_layer_placed
            vis_dims = (CONTAINER_L, CONTAINER_W)
        elif best_count == count_side:
            method = f"side-packed ({n_rows} rows)"
            best_placed = placed_side
            vis_dims = (CONTAINER_L, CONTAINER_H)
        elif best_count == count_end:
            method = f"end-packed ({n_rows_long} rows)"
            best_placed = placed_end
            vis_dims = (CONTAINER_W, CONTAINER_H)
        else:
            method = "standing upright"
            _, best_placed = pack_grid(poly, CONTAINER_L, CONTAINER_W)
            vis_dims = (CONTAINER_L, CONTAINER_W)

        # Volumetric efficiency
        body_vol = poly.area * upright_depth  # rough body volume
        total_body_vol = body_vol * best_count
        vol_efficiency = total_body_vol / CONTAINER_VOL

        results[name] = {
            "count_upright": count_upright,
            "count_stacked": count_stacked,
            "count_side_packed": count_side,
            "count_end_packed": count_end,
            "best_count": best_count,
            "best_method": method,
            "volumetric_efficiency": float(vol_efficiency),
        }

        print(f"\n  {name}:")
        print(f"    Standing:      {count_upright}")
        print(f"    Stacked:       {count_stacked} ({count_per_layer}/layer x {n_layers} layers)")
        print(f"    Side-packed:   {count_side} ({count_side_opt}/slice x {n_rows} rows)")
        print(f"    End-packed:    {count_end} ({count_end_opt}/slice x {n_rows_long} rows)")
        print(f"    BEST:          {best_count} ({method})")
        print(f"    Vol. efficiency: {vol_efficiency:.1%}")

        plot_packing(f"Container - {name}", best_placed,
                     vis_dims[0], vis_dims[1], best_count)

    with open(f"{OUTPUT_DIR}/container_packing.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_4_tpose_tax_visualization():
    """Visualize the T-Pose Tax across all poses."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: The T-Pose Tax")
    print("=" * 60)

    tpose_poly = POSES["T-Pose"]()

    names = []
    taxes = []
    efficiencies = []

    for name, func in POSES.items():
        poly = func()
        tax = tpose_tax(poly, tpose_poly)
        eff = packing_efficiency(poly)
        names.append(name.replace(" / ", "\n"))
        taxes.append(tax)
        efficiencies.append(eff)
        print(f"  {name}: T-Pose Tax = {tax:.2f}x, Efficiency = {eff:.1%}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['#2ecc71' if t < 1.2 else '#e74c3c' if t > 1.5 else '#f39c12' for t in taxes]

    ax1.barh(names, taxes, color=colors, edgecolor='black')
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='T-Pose baseline')
    ax1.set_xlabel("T-Pose Tax (lower is better)")
    ax1.set_title("The T-Pose Tax\n(Relative BB waste vs T-Pose)")
    ax1.legend()

    ax2.barh(names, [e * 100 for e in efficiencies], color='steelblue', edgecolor='black')
    ax2.set_xlabel("Bounding Box Efficiency (%)")
    ax2.set_title("Bounding Box Fill Ratio by Pose")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tpose_tax.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved tpose_tax.png")


def experiment_5_pose_optimization():
    """
    'Discover' the optimal pose via brute-force rotation of body parts.
    (Spoiler: it converges near the fetal position.)
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Automated Pose Discovery")
    print("=" * 60)

    # We'll parameterize a simple body and vary limb angles
    # This is intentionally overcomplicated for comedic effect

    def parametric_human(arm_angle, leg_spread, torso_bend):
        """
        arm_angle: 0 = at sides, 90 = T-pose, 180 = overhead
        leg_spread: 0 = together, 45 = spread
        torso_bend: 0 = straight, 90 = folded
        """
        arm_angle_rad = np.radians(arm_angle)
        leg_rad = np.radians(leg_spread)
        bend_rad = np.radians(torso_bend)

        # Torso (always centered)
        torso = box(-0.18, 0.85, 0.18, 1.58)

        # Head
        head_x = -np.sin(bend_rad) * 0.3
        head_y = 1.58 + np.cos(bend_rad) * 0.2
        head = translate(
            Polygon([(-.08, 0), (.08, 0), (.10, .17), (.08, .24), (-.08, .24), (-.10, .17)]),
            head_x, head_y
        )

        # Arms
        arm_len = 0.55
        arm_dx = np.cos(arm_angle_rad) * arm_len
        arm_dy = np.sin(arm_angle_rad) * arm_len

        l_arm_pts = [
            (-0.18, 1.45),
            (-0.18 - arm_dx, 1.45 + arm_dy),
            (-0.18 - arm_dx - 0.04, 1.45 + arm_dy + 0.04),
            (-0.18 - 0.04, 1.45 + 0.04),
        ]
        r_arm_pts = [
            (0.18, 1.45),
            (0.18 + arm_dx, 1.45 + arm_dy),
            (0.18 + arm_dx + 0.04, 1.45 + arm_dy + 0.04),
            (0.18 + 0.04, 1.45 + 0.04),
        ]

        # Legs
        leg_len = 0.85
        l_leg_pts = [
            (-0.02, 0.85), (-0.16, 0.85),
            (-0.16 - np.sin(leg_rad) * leg_len, 0.85 - np.cos(leg_rad) * leg_len),
            (-0.02 - np.sin(leg_rad) * leg_len, 0.85 - np.cos(leg_rad) * leg_len),
        ]
        r_leg_pts = [
            (0.02, 0.85), (0.16, 0.85),
            (0.16 + np.sin(leg_rad) * leg_len, 0.85 - np.cos(leg_rad) * leg_len),
            (0.02 + np.sin(leg_rad) * leg_len, 0.85 - np.cos(leg_rad) * leg_len),
        ]

        parts = [torso, head]
        for pts in [l_arm_pts, r_arm_pts, l_leg_pts, r_leg_pts]:
            try:
                p = Polygon(pts)
                if p.is_valid and p.area > 0:
                    parts.append(p)
            except:
                pass

        return unary_union(parts)

    # Grid search over pose parameters
    best_eff = 0
    best_params = (0, 0, 0)
    all_results = []

    arm_angles = np.linspace(0, 180, 19)
    leg_spreads = np.linspace(0, 45, 10)
    torso_bends = [0]  # Keep simple for now

    for aa in arm_angles:
        for ls in leg_spreads:
            for tb in torso_bends:
                try:
                    body = parametric_human(aa, ls, tb)
                    eff = packing_efficiency(body)
                    all_results.append((aa, ls, tb, eff))
                    if eff > best_eff:
                        best_eff = eff
                        best_params = (aa, ls, tb)
                except:
                    pass

    print(f"\n  Tested {len(all_results)} pose configurations")
    print(f"  Optimal pose found:")
    print(f"    Arm angle:    {best_params[0]:.0f}°")
    print(f"    Leg spread:   {best_params[1]:.0f}°")
    print(f"    Torso bend:   {best_params[2]:.0f}°")
    print(f"    BB Efficiency: {best_eff:.1%}")

    # Heatmap of arm angle vs leg spread
    arm_vals = sorted(set(r[0] for r in all_results))
    leg_vals = sorted(set(r[1] for r in all_results))

    eff_grid = np.zeros((len(leg_vals), len(arm_vals)))
    for aa, ls, tb, eff in all_results:
        ai = arm_vals.index(aa)
        li = leg_vals.index(ls)
        eff_grid[li, ai] = max(eff_grid[li, ai], eff)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(eff_grid, aspect='auto', origin='lower', cmap='RdYlGn',
                    extent=[min(arm_angles), max(arm_angles), min(leg_spreads), max(leg_spreads)])
    ax.set_xlabel("Arm Angle (degrees)")
    ax.set_ylabel("Leg Spread (degrees)")
    ax.set_title("Pose Parameter Space: Bounding-Box Efficiency\n(Green = more packable, Red = less packable)")
    plt.colorbar(im, label="BB Efficiency")

    ax.plot(best_params[0], best_params[1], 'k*', markersize=20, label=f'Optimal ({best_eff:.1%})')
    ax.legend()

    plt.savefig(f"{OUTPUT_DIR}/pose_optimization.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved pose_optimization.png")

    return best_params, best_eff


def experiment_6_evolutionary():
    """
    Evolve human body proportions for optimal packing.
    Hypothesis: converges to a sphere (or at least a square).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Evolutionary Body Optimization")
    print("(What Should Humans Look Like for Optimal Packing?)")
    print("=" * 60)

    def body_from_genes(genes):
        """
        genes: [head_r, torso_w, torso_h, arm_len, arm_w, leg_len, leg_w]
        All normalized 0-1, scaled to reasonable ranges.
        """
        head_r = 0.05 + genes[0] * 0.15
        torso_w = 0.1 + genes[1] * 0.4
        torso_h = 0.2 + genes[2] * 0.8
        arm_len = 0.1 + genes[3] * 0.6
        arm_w = 0.02 + genes[4] * 0.1
        leg_len = 0.2 + genes[5] * 0.8
        leg_w = 0.03 + genes[6] * 0.15

        # Build body - arms at sides, legs together
        torso = box(-torso_w/2, leg_len, torso_w/2, leg_len + torso_h)
        head_cy = leg_len + torso_h + head_r
        head = Polygon([
            (head_r * np.cos(a), head_cy + head_r * np.sin(a))
            for a in np.linspace(0, 2*np.pi, 20)
        ])
        l_arm = box(-torso_w/2 - arm_w, leg_len + torso_h - arm_len,
                     -torso_w/2, leg_len + torso_h)
        r_arm = box(torso_w/2, leg_len + torso_h - arm_len,
                     torso_w/2 + arm_w, leg_len + torso_h)
        l_leg = box(-leg_w - 0.01, 0, -0.01, leg_len)
        r_leg = box(0.01, 0, leg_w + 0.01, leg_len)

        return unary_union([torso, head, l_arm, r_arm, l_leg, r_leg])

    # Simple evolutionary algorithm
    POP_SIZE = 100
    GENERATIONS = 50
    MUTATION_RATE = 0.3
    N_GENES = 7

    # Initialize population
    population = np.random.rand(POP_SIZE, N_GENES)

    history = []

    for gen in range(GENERATIONS):
        # Evaluate fitness
        fitnesses = []
        for individual in population:
            try:
                body = body_from_genes(individual)
                fit = packing_efficiency(body)
            except:
                fit = 0
            fitnesses.append(fit)

        fitnesses = np.array(fitnesses)
        best_idx = np.argmax(fitnesses)
        history.append(float(fitnesses[best_idx]))

        if gen % 10 == 0:
            print(f"  Gen {gen:3d}: best fitness = {fitnesses[best_idx]:.4f}")

        # Selection (tournament)
        new_pop = []
        # Elitism - keep best
        new_pop.append(population[best_idx].copy())

        for _ in range(POP_SIZE - 1):
            # Tournament selection
            i, j = np.random.randint(0, POP_SIZE, 2)
            parent1 = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            i, j = np.random.randint(0, POP_SIZE, 2)
            parent2 = population[i] if fitnesses[i] > fitnesses[j] else population[j]

            # Crossover
            mask = np.random.rand(N_GENES) > 0.5
            child = np.where(mask, parent1, parent2)

            # Mutation
            mutations = np.random.rand(N_GENES) < MUTATION_RATE
            child[mutations] += np.random.randn(mutations.sum()) * 0.1
            child = np.clip(child, 0, 1)

            new_pop.append(child)

        population = np.array(new_pop)

    # Final best
    best_genes = population[0]
    best_body = body_from_genes(best_genes)
    best_eff = packing_efficiency(best_body)

    print(f"\n  Final evolved human:")
    print(f"    Head radius:  {0.05 + best_genes[0] * 0.15:.2f}m")
    print(f"    Torso width:  {0.1 + best_genes[1] * 0.4:.2f}m")
    print(f"    Torso height: {0.2 + best_genes[2] * 0.8:.2f}m")
    print(f"    Arm length:   {0.1 + best_genes[3] * 0.6:.2f}m")
    print(f"    Arm width:    {0.02 + best_genes[4] * 0.1:.2f}m")
    print(f"    Leg length:   {0.2 + best_genes[5] * 0.8:.2f}m")
    print(f"    Leg width:    {0.03 + best_genes[6] * 0.15:.2f}m")
    print(f"    BB Efficiency: {best_eff:.1%}")

    # Plot evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(history, 'b-', linewidth=2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Best Fitness (BB Efficiency)")
    ax1.set_title("Evolutionary Convergence\n(Optimizing Human Body for Packing)")
    ax1.grid(True, alpha=0.3)

    # Plot the evolved human vs normal human
    normal = human_standing()
    evolved = best_body

    for poly, label, color, offset in [(normal, "Homo sapiens", "steelblue", 0),
                                         (evolved, "Homo packabilis", "coral", 1.5)]:
        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                x, y = geom.exterior.xy
                ax2.fill([xi + offset for xi in x], y, alpha=0.6, fc=color, ec='black')
        else:
            x, y = poly.exterior.xy
            ax2.fill([xi + offset for xi in x], y, alpha=0.6, fc=color, ec='black')

        minx, miny, maxx, maxy = poly.bounds
        bb = plt.Rectangle((minx + offset, miny), maxx-minx, maxy-miny,
                             fill=False, ec='red', linestyle='--')
        ax2.add_patch(bb)
        ax2.text(offset + (minx+maxx)/2, maxy + 0.1, label, ha='center', fontsize=10, fontweight='bold')

    ax2.set_aspect('equal')
    ax2.set_title(f"Normal vs Evolved Human\n(BB Eff: {packing_efficiency(normal):.1%} vs {best_eff:.1%})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved evolution.png")


def experiment_7_venue_comparison():
    """
    The grand comparison: how many humans fit in various real-world spaces?
    Full 3D packing across multiple venues and all poses.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Venue Capacity Optimization (3D)")
    print("=" * 60)

    VENUES = {
        "Elevator":          {"L": 2.0,  "W": 1.5,   "H": 2.4},
        "Phone Booth":       {"L": 0.9,  "W": 0.9,   "H": 2.3},
        "Shipping Container":{"L": 5.9,  "W": 2.35,  "H": 2.39},
        "Minivan":           {"L": 2.0,  "W": 1.5,   "H": 1.3},
        "Boeing 737 Cabin":  {"L": 28.0, "W": 3.54,  "H": 2.2},
        "Subway Car":        {"L": 15.5, "W": 2.5,   "H": 2.1},
        "School Bus":        {"L": 7.3,  "W": 2.3,   "H": 1.8},
        "Hot Tub":           {"L": 2.1,  "W": 2.1,   "H": 0.9},
    }

    all_results = {}

    for venue_name, dims in VENUES.items():
        L, W, H = dims["L"], dims["W"], dims["H"]
        vol = L * W * H
        print(f"\n  {venue_name}: {L}m x {W}m x {H}m (vol={vol:.1f}m^3)")

        venue_results = {}

        for pose_name, func in POSES.items():
            poly = func()
            orient = POSE_ORIENTATIONS[pose_name]
            upright_depth = orient["upright_depth"]
            lying_depth = orient["lying_depth"]

            best_count = 0

            # Try all 3 packing faces: LxW, LxH, WxH
            for face, face_dims, stack_dim in [
                ("floor (LxW)", (L, W), H),
                ("side (LxH)",  (L, H), W),
                ("end (WxH)",   (W, H), L),
            ]:
                # Try stacking with lying depth
                ct_opt, _, _ = pack_with_optimal_rotation(
                    poly, face_dims[0], face_dims[1], steps=36)
                ct_nest, _ = pack_grid_with_nesting(
                    poly, face_dims[0], face_dims[1])
                per_layer = max(ct_opt, ct_nest)

                for depth in [upright_depth, lying_depth]:
                    layers = int(stack_dim // depth)
                    total = per_layer * max(layers, 1)
                    if total > best_count:
                        best_count = total

            venue_results[pose_name] = best_count

        # Print results for this venue
        best_pose = max(venue_results, key=venue_results.get)
        for pose_name, count in venue_results.items():
            marker = " <-- BEST" if pose_name == best_pose else ""
            print(f"    {pose_name}: {count}{marker}")

        all_results[venue_name] = venue_results

    # Big comparison bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    venue_names = list(VENUES.keys())
    pose_names = list(POSES.keys())
    x = np.arange(len(venue_names))
    width = 0.12

    colors = plt.cm.Set2(np.linspace(0, 1, len(pose_names)))

    for i, pose_name in enumerate(pose_names):
        counts = [all_results[v][pose_name] for v in venue_names]
        bars = ax.bar(x + i * width, counts, width, label=pose_name,
                       color=colors[i], edgecolor='black', linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(count), ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xlabel("Venue")
    ax.set_ylabel("Number of Humans")
    ax.set_title("Optimal Human Packing by Venue and Pose (3D Volumetric)")
    ax.set_xticks(x + width * (len(pose_names) - 1) / 2)
    ax.set_xticklabels(venue_names, rotation=30, ha='right')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/venue_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved venue_comparison.png")

    # Also save a nice summary table
    with open(f"{OUTPUT_DIR}/venue_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print the grand summary
    print("\n  GRAND SUMMARY - Best packing per venue:")
    print(f"  {'Venue':<22} {'Best Pose':<28} {'Count':>6}")
    print("  " + "-" * 58)
    for venue_name in venue_names:
        best_pose = max(all_results[venue_name], key=all_results[venue_name].get)
        count = all_results[venue_name][best_pose]
        print(f"  {venue_name:<22} {best_pose:<28} {count:>6}")

    return all_results


def experiment_8_spooning():
    """
    The spooning efficiency analysis.
    Can nesting concave humans (spooning) beat grid packing?
    Model two humans as interlocking concave shapes.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 8: The Spooning Hypothesis")
    print("(Is spooning mathematically optimal?)")
    print("=" * 60)

    # Side-lying human silhouette (simplified)
    def human_side_lying():
        """Side view of a lying human - the spooning profile."""
        # Head
        head = Polygon([(.0, .22), (.15, .22), (.18, .30), (.15, .38),
                         (.0, .38), (-.02, .30)])
        # Torso (curved back)
        torso = Polygon([(.0, .10), (.25, .08), (.30, .15), (.28, .22),
                          (.0, .22), (-.05, .18)])
        # Legs (slightly bent for natural lying pose)
        legs = Polygon([(.25, .08), (.55, .02), (.60, .06), (.58, .12),
                         (.30, .15)])
        # Arm in front
        arm = Polygon([(.05, .22), (.15, .25), (.20, .28), (.10, .28)])

        return unary_union([head, torso, legs, arm])

    def human_side_lying_flipped():
        """Mirrored side-lying human for spooning."""
        from shapely.affinity import scale as sscale
        body = human_side_lying()
        # Flip horizontally
        return sscale(body, xfact=-1, yfact=1, origin='centroid')

    person = human_side_lying()

    # Grid packing (no spooning)
    BED_W = 2.0  # king bed width
    BED_H = 2.0  # bed length

    count_grid, placed_grid = pack_grid(person, BED_W, BED_H)

    # Spooning: alternate facing directions, try to nest closer
    minx, miny, maxx, maxy = person.bounds
    pw = maxx - minx
    ph = maxy - miny
    shifted = translate(person, -minx, -miny)
    flipped = human_side_lying_flipped()
    fminx, fminy, fmaxx, fmaxy = flipped.bounds
    flipped_shifted = translate(flipped, -fminx, -fminy)

    # Try overlapping the bounding boxes by sliding the flipped person closer
    best_spoon_count = 0
    best_spoon_placed = []
    best_overlap = 0

    for overlap_frac in np.linspace(0, 0.5, 50):
        effective_w = pw * (1 - overlap_frac)
        nx = int(BED_W // effective_w)
        ny = int(BED_H // ph)

        placed = []
        valid = True
        for ix in range(nx):
            for iy in range(ny):
                if ix % 2 == 0:
                    p = translate(shifted, ix * effective_w, iy * ph)
                else:
                    p = translate(flipped_shifted, ix * effective_w, iy * ph)
                placed.append(p)

        # Check for actual overlaps between adjacent people
        has_overlap = False
        for i in range(len(placed)):
            for j in range(i+1, min(i+5, len(placed))):
                if placed[i].intersects(placed[j]):
                    inter = placed[i].intersection(placed[j])
                    if inter.area > 0.0001:  # Non-trivial overlap
                        has_overlap = True
                        break
            if has_overlap:
                break

        if not has_overlap and len(placed) > best_spoon_count:
            best_spoon_count = len(placed)
            best_spoon_placed = placed
            best_overlap = overlap_frac

    improvement = (best_spoon_count - count_grid) / max(count_grid, 1) * 100

    print(f"\n  Bed dimensions: {BED_W}m x {BED_H}m")
    print(f"  Grid packing (no spooning): {count_grid} humans")
    print(f"  Optimal spooning:           {best_spoon_count} humans")
    print(f"  BB overlap factor:          {best_overlap:.1%}")
    print(f"  Spooning improvement:       {improvement:+.0f}%")
    if improvement > 0:
        print(f"  CONCLUSION: Spooning IS mathematically optimal.")
    else:
        print(f"  CONCLUSION: Spooning provides no packing advantage.")
        print(f"              It must serve some other evolutionary purpose.")

    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for poly in placed_grid[:count_grid]:
        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                x, y = geom.exterior.xy
                ax1.fill(x, y, alpha=0.5, fc='steelblue', ec='navy')
        else:
            x, y = poly.exterior.xy
            ax1.fill(x, y, alpha=0.5, fc='steelblue', ec='navy')
    ax1.add_patch(plt.Rectangle((0,0), BED_W, BED_H, fill=False, ec='black', lw=2))
    ax1.set_title(f"Grid Packing: {count_grid} humans")
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.1, BED_W + 0.1)
    ax1.set_ylim(-0.1, BED_H + 0.1)

    spoon_colors = ['coral', 'steelblue']
    for i, poly in enumerate(best_spoon_placed):
        color = spoon_colors[i % 2]
        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                x, y = geom.exterior.xy
                ax2.fill(x, y, alpha=0.5, fc=color, ec='black')
        else:
            x, y = poly.exterior.xy
            ax2.fill(x, y, alpha=0.5, fc=color, ec='black')
    ax2.add_patch(plt.Rectangle((0,0), BED_W, BED_H, fill=False, ec='black', lw=2))
    ax2.set_title(f"Spooning Config: {best_spoon_count} humans")
    ax2.set_aspect('equal')
    ax2.set_xlim(-0.1, BED_W + 0.1)
    ax2.set_ylim(-0.1, BED_H + 0.1)

    fig.suptitle("The Spooning Hypothesis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/spooning.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved spooning.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  OPTIMAL ANTHROPOMORPHIC POLYGON PACKING")
    print("  A Rigorous Analysis of Human Stacking Configurations")
    print("=" * 60)
    print()

    experiment_1_pose_metrics()
    experiment_2_elevator_packing()
    experiment_3_container_shipping()
    experiment_4_tpose_tax_visualization()
    experiment_5_pose_optimization()
    experiment_6_evolutionary()
    experiment_7_venue_comparison()
    experiment_8_spooning()

    print("\n" + "=" * 60)
    print("All experiments complete. Results saved to ./results/")
    print("=" * 60)
