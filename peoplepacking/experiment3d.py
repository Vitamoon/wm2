"""
OPTIMAL ANTHROPOMORPHIC VOLUMETRIC PACKING:
A Rigorous 3D Analysis of Human Stacking Configurations

Full 3D experiment with articulated human meshes,
volumetric packing with rotation search, and solid 3D rendering.
"""

import numpy as np
import trimesh
import json
import os
import itertools

from human3d import (
    build_posed_human, POSES_3D, HumanRig,
    get_bounding_box, get_bb_volume, packing_efficiency_3d,
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
)

import pyvista as pv
import matplotlib.pyplot as plt

OUTPUT_DIR = "results3d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use offscreen rendering for pyvista
pv.OFF_SCREEN = True


# ============================================================
# Pyvista rendering helpers
# ============================================================

def trimesh_to_pyvista(mesh):
    """Convert trimesh to pyvista PolyData."""
    faces = mesh.faces
    n_faces = len(faces)
    # pyvista wants [n_verts, v0, v1, v2, ...] format
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=int),
        faces
    ]).ravel()
    return pv.PolyData(mesh.vertices, pv_faces)


def render_mesh(mesh, filename, title="", color='steelblue', show_bb=True,
                window_size=(800, 800), camera_position=None):
    """Render a single mesh to an image file with solid shading."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')

    pv_mesh = trimesh_to_pyvista(mesh)
    pl.add_mesh(pv_mesh, color=color, opacity=0.85, smooth_shading=True,
                show_edges=False, specular=0.3)

    if show_bb:
        bounds = mesh.bounds
        bb = pv.Box(bounds=[bounds[0][0], bounds[1][0],
                            bounds[0][1], bounds[1][1],
                            bounds[0][2], bounds[1][2]])
        pl.add_mesh(bb, color='red', style='wireframe', line_width=1.5, opacity=0.4)

    if title:
        pl.add_title(title, font_size=12)

    if camera_position:
        pl.camera_position = camera_position
    else:
        pl.camera_position = 'xy'
        pl.camera.azimuth = 30
        pl.camera.elevation = 20

    pl.save_graphic(filename) if filename.endswith('.svg') else pl.screenshot(filename)
    pl.close()


def render_packing_scene(meshes_with_offsets, container_dims, filename, title="",
                          window_size=(1200, 800), max_render=80):
    """Render a 3D packing scene with multiple colored humans in a container."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')

    # Color palette
    palette = [
        '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
        '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
        '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF', '#AEC7E8',
    ]

    n_to_render = min(len(meshes_with_offsets), max_render)

    for i, (mesh, offset) in enumerate(meshes_with_offsets[:n_to_render]):
        shifted = mesh.copy()
        shifted.vertices += offset
        pv_mesh = trimesh_to_pyvista(shifted)
        color = palette[i % len(palette)]
        pl.add_mesh(pv_mesh, color=color, opacity=0.7, smooth_shading=True,
                    show_edges=False, specular=0.2)

    # Container wireframe
    Lx, Ly, Lz = container_dims
    container = pv.Box(bounds=[0, Lx, 0, Ly, 0, Lz])
    pl.add_mesh(container, color='black', style='wireframe', line_width=2.5, opacity=0.9)

    if title:
        pl.add_title(title, font_size=11)

    pl.camera_position = 'xy'
    pl.camera.azimuth = 35
    pl.camera.elevation = 25
    pl.camera.zoom(0.85)

    pl.screenshot(filename)
    pl.close()


# ============================================================
# 3D Packing with rotation search
# ============================================================

def rotated_bb(mesh, rx, ry, rz):
    """Get bounding box of mesh after rotation (without modifying mesh)."""
    R = rotation_matrix_z(rz) @ rotation_matrix_y(ry) @ rotation_matrix_x(rx)
    rotated_verts = mesh.vertices @ R.T
    bb_min = rotated_verts.min(axis=0)
    bb_max = rotated_verts.max(axis=0)
    return bb_max - bb_min, R


def pack_3d_grid(mesh, container_dims):
    """
    Pack copies of mesh into container using AABB grid packing.
    Tries all 6 axis permutations of the mesh's bounding box.
    Returns: (count, list_of_offsets, bb_dims_used)
    """
    bb = get_bounding_box(mesh)
    Lx, Ly, Lz = container_dims

    best_count = 0
    best_offsets = []
    best_bb = bb

    for perm in itertools.permutations([0, 1, 2]):
        bw, bd, bh = bb[perm[0]], bb[perm[1]], bb[perm[2]]

        nx = int(Lx // bw) if bw > 0 else 0
        ny = int(Ly // bd) if bd > 0 else 0
        nz = int(Lz // bh) if bh > 0 else 0

        count = nx * ny * nz
        if count > best_count:
            best_count = count
            best_bb = np.array([bw, bd, bh])
            offsets = []
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        offsets.append(np.array([ix * bw, iy * bd, iz * bh]))
            best_offsets = offsets

    return best_count, best_offsets, best_bb


def pack_3d_rotation_search(mesh, container_dims, angle_steps=12):
    """
    Try rotating the mesh at various angles before grid packing.
    Searches over yaw, pitch, roll to find the rotation that
    minimizes bounding box and maximizes packing count.

    Returns: (count, offsets, bb_dims, rotation_matrix)
    """
    Lx, Ly, Lz = container_dims
    angles = np.linspace(0, 180, angle_steps, endpoint=False)

    best_count = 0
    best_offsets = []
    best_bb = None
    best_R = np.eye(3)

    for rx in angles:
        for ry in angles:
            bb_dims, R = rotated_bb(mesh, rx, ry, 0)

            # Try all 6 permutations of this BB
            for perm in itertools.permutations([0, 1, 2]):
                bw, bd, bh = bb_dims[perm[0]], bb_dims[perm[1]], bb_dims[perm[2]]

                nx = int(Lx // bw) if bw > 0 else 0
                ny = int(Ly // bd) if bd > 0 else 0
                nz = int(Lz // bh) if bh > 0 else 0

                count = nx * ny * nz
                if count > best_count:
                    best_count = count
                    best_bb = np.array([bw, bd, bh])
                    best_R = R
                    offsets = []
                    for ix in range(nx):
                        for iy in range(ny):
                            for iz in range(nz):
                                offsets.append(np.array([ix * bw, iy * bd, iz * bh]))
                    best_offsets = offsets

    return best_count, best_offsets, best_bb, best_R


def apply_rotation_to_mesh(mesh, R):
    """Apply a 3x3 rotation matrix to mesh vertices (returns copy)."""
    m = mesh.copy()
    m.vertices = m.vertices @ R.T
    # Re-center to put min at origin
    m.vertices -= m.vertices.min(axis=0)
    return m


# ============================================================
# Experiments
# ============================================================

def experiment_1_pose_gallery():
    """Render all poses with solid 3D meshes and compute metrics."""
    print("=" * 60)
    print("EXPERIMENT 1: 3D Pose Gallery & Metrics")
    print("=" * 60)

    results = {}

    # Render individual pose images
    for name in POSES_3D:
        mesh = build_posed_human(name)

        bb_dims = get_bounding_box(mesh)
        bb_vol = get_bb_volume(mesh)
        hull_vol = mesh.convex_hull.volume
        eff = packing_efficiency_3d(mesh)

        results[name] = {
            "bb_dims": [float(d) for d in bb_dims],
            "bb_volume": float(bb_vol),
            "hull_volume": float(hull_vol),
            "packing_efficiency": float(eff),
        }

        print(f"  {name}: BB={bb_dims[0]:.2f}x{bb_dims[1]:.2f}x{bb_dims[2]:.2f}m  "
              f"vol={bb_vol:.3f}m3  eff={eff:.0%}")

        safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        render_mesh(
            mesh, f"{OUTPUT_DIR}/pose_{safe}.png",
            title=f"{name}  |  BB Eff: {eff:.0%}  |  Vol: {bb_vol:.2f}m3",
            color='#4C72B0',
        )

    # Also make a combined gallery using matplotlib subplots of the screenshots
    # (pyvista doesn't do subplots well, so we compose from individual renders)
    from PIL import Image
    pose_names = list(POSES_3D.keys())
    cols = 5
    rows = (len(pose_names) + cols - 1) // cols
    cell_w, cell_h = 800, 800
    gallery = Image.new('RGB', (cols * cell_w, rows * cell_h), 'white')

    for i, name in enumerate(pose_names):
        safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        img_path = f"{OUTPUT_DIR}/pose_{safe}.png"
        try:
            img = Image.open(img_path).resize((cell_w, cell_h))
            r, c = divmod(i, cols)
            gallery.paste(img, (c * cell_w, r * cell_h))
        except Exception:
            pass

    gallery.save(f"{OUTPUT_DIR}/pose_gallery_3d.png")
    print(f"\nSaved pose_gallery_3d.png ({cols}x{rows} grid)")

    with open(f"{OUTPUT_DIR}/pose_metrics_3d.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_2_packing_comparison():
    """Compare how many humans fit in venues using rotation-search packing."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 3D Venue Packing (with rotation search)")
    print("=" * 60)

    VENUES = {
        "Elevator":           (2.0, 1.5, 2.4),
        "Phone Booth":        (0.9, 0.9, 2.3),
        "Shipping Container": (5.9, 2.35, 2.39),
        "Minivan":            (2.0, 1.5, 1.3),
        "Boeing 737 Cabin":   (28.0, 3.54, 2.2),
        "Subway Car":         (15.5, 2.5, 2.1),
        "School Bus":         (7.3, 2.3, 1.8),
        "Hot Tub":            (2.1, 2.1, 0.9),
    }

    all_results = {}

    for venue_name, dims in VENUES.items():
        vol = dims[0] * dims[1] * dims[2]
        print(f"\n  {venue_name}: {dims[0]}x{dims[1]}x{dims[2]}m (vol={vol:.1f}m3)")

        venue_results = {}

        for pose_name in POSES_3D:
            mesh = build_posed_human(pose_name)

            # Basic grid (6 orientations)
            count_basic, _, _ = pack_3d_grid(mesh, dims)

            # Rotation search (many angles)
            count_rot, _, _, _ = pack_3d_rotation_search(mesh, dims, angle_steps=10)

            best = max(count_basic, count_rot)
            venue_results[pose_name] = best

        best_pose = max(venue_results, key=venue_results.get)
        for pose_name, count in venue_results.items():
            marker = " <-- BEST" if pose_name == best_pose else ""
            print(f"    {pose_name}: {count}{marker}")

        all_results[venue_name] = venue_results

    # Comparison bar chart (matplotlib - better for bar charts)
    fig, ax = plt.subplots(figsize=(18, 9))

    venue_names = list(VENUES.keys())
    pose_names = list(POSES_3D.keys())
    x = np.arange(len(venue_names))
    width = 0.8 / len(pose_names)

    colors = plt.cm.tab20(np.linspace(0, 1, len(pose_names)))

    for i, pose_name in enumerate(pose_names):
        counts = [all_results[v].get(pose_name, 0) for v in venue_names]
        ax.bar(x + i * width, counts, width, label=pose_name,
               color=colors[i], edgecolor='black', linewidth=0.3)

    ax.set_xlabel("Venue")
    ax.set_ylabel("Number of Humans (log scale)")
    ax.set_title("Optimal Human Packing: 3D Volumetric Analysis by Venue and Pose\n(with rotation search)")
    ax.set_xticks(x + width * (len(pose_names) - 1) / 2)
    ax.set_xticklabels(venue_names, rotation=30, ha='right')
    ax.legend(fontsize=6, loc='upper left', ncol=2)
    ax.set_yscale('symlog', linthresh=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/venue_comparison_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved venue_comparison_3d.png")

    # Summary
    print("\n  GRAND SUMMARY (3D with rotation search):")
    print(f"  {'Venue':<22} {'Best Pose':<28} {'Count':>6}")
    print("  " + "-" * 58)
    for venue_name in venue_names:
        best_pose = max(all_results[venue_name], key=all_results[venue_name].get)
        count = all_results[venue_name][best_pose]
        print(f"  {venue_name:<22} {best_pose:<28} {count:>6}")

    with open(f"{OUTPUT_DIR}/venue_packing_3d.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def experiment_3_packing_visualization():
    """Render solid 3D packing scenes for interesting venue/pose combos."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: 3D Packing Visualizations (Solid)")
    print("=" * 60)

    SCENES = [
        ("Elevator",           (2.0, 1.5, 2.4),   "Standing (arms at sides)"),
        ("Elevator",           (2.0, 1.5, 2.4),   "T-Pose"),
        ("Elevator",           (2.0, 1.5, 2.4),   "Fetal Position"),
        ("Shipping Container", (5.9, 2.35, 2.39),  "Standing (arms at sides)"),
        ("Shipping Container", (5.9, 2.35, 2.39),  "Fetal Position"),
        ("Shipping Container", (5.9, 2.35, 2.39),  "Squat (Slav)"),
        ("Phone Booth",        (0.9, 0.9, 2.3),    "Standing (arms at sides)"),
        ("Hot Tub",            (2.1, 2.1, 0.9),    "Planking"),
        ("Minivan",            (2.0, 1.5, 1.3),    "Fetal Position"),
        ("School Bus",         (7.3, 2.3, 1.8),    "Dab"),
        ("School Bus",         (7.3, 2.3, 1.8),    "Standing (arms at sides)"),
        ("Boeing 737 Cabin",   (28.0, 3.54, 2.2),  "Standing (arms at sides)"),
    ]

    for venue_name, dims, pose_name in SCENES:
        print(f"\n  Rendering: {venue_name} x {pose_name}")

        mesh = build_posed_human(pose_name)

        # Try both basic and rotation-search packing
        count_basic, offsets_basic, bb_basic = pack_3d_grid(mesh, dims)
        count_rot, offsets_rot, bb_rot, R_rot = pack_3d_rotation_search(mesh, dims, angle_steps=10)

        if count_rot > count_basic:
            count, offsets, bb_used = count_rot, offsets_rot, bb_rot
            mesh_packed = apply_rotation_to_mesh(mesh, R_rot)
            method = "rotation-optimized"
        else:
            count, offsets, bb_used = count_basic, offsets_basic, bb_basic
            # Still need to orient mesh to match the best permutation
            mesh_packed = mesh.copy()
            mesh_packed.vertices -= mesh_packed.vertices.min(axis=0)
            method = "axis-aligned"

        print(f"    Packed: {count} humans ({method})")

        if count == 0:
            print(f"    (No humans fit - skipping)")
            continue

        # Build mesh+offset pairs for rendering
        meshes_with_offsets = [(mesh_packed, offset) for offset in offsets]

        vol_eff = count * mesh.convex_hull.volume / (dims[0] * dims[1] * dims[2])

        safe = f"{venue_name}_{pose_name}".replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        render_packing_scene(
            meshes_with_offsets, dims,
            f"{OUTPUT_DIR}/scene_{safe}.png",
            title=f"{venue_name}: {count} humans in {pose_name}  |  "
                  f"Vol. Eff: {vol_eff:.0%}  |  {method}",
        )
        print(f"    Saved scene_{safe}.png")


def experiment_4_tpose_tax_3d():
    """The T-Pose Tax in 3D."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: 3D T-Pose Tax")
    print("=" * 60)

    tpose_mesh = build_posed_human("T-Pose")
    tpose_eff = packing_efficiency_3d(tpose_mesh)

    names = []
    taxes = []
    effs = []

    for pose_name in POSES_3D:
        mesh = build_posed_human(pose_name)
        eff = packing_efficiency_3d(mesh)
        tax = eff / tpose_eff if tpose_eff > 0 else 0

        names.append(pose_name)
        taxes.append(tax)
        effs.append(eff)

        print(f"  {pose_name}: eff={eff:.1%}, tax={tax:.2f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    sorted_idx = np.argsort(taxes)[::-1]
    sorted_names = [names[i] for i in sorted_idx]
    sorted_taxes = [taxes[i] for i in sorted_idx]
    sorted_effs = [effs[i] for i in sorted_idx]

    colors = ['#2ecc71' if t > 1.5 else '#e74c3c' if t < 1.0 else '#f39c12' for t in sorted_taxes]

    ax1.barh(sorted_names, sorted_taxes, color=colors, edgecolor='black')
    ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='T-Pose baseline')
    ax1.set_xlabel("T-Pose Tax (higher = more efficient than T-Pose)")
    ax1.set_title("3D T-Pose Tax by Pose")
    ax1.legend()

    ax2.barh(sorted_names, [e * 100 for e in sorted_effs], color='steelblue', edgecolor='black')
    ax2.set_xlabel("3D BB Packing Efficiency (%)")
    ax2.set_title("3D Bounding Box Fill Ratio")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tpose_tax_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved tpose_tax_3d.png")


def experiment_5_efficiency_ranking():
    """Rank all poses with rotation-search packing."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Definitive Pose Efficiency Ranking")
    print("=" * 60)

    CONTAINER = (5.9, 2.35, 2.39)

    data = []
    for pose_name in POSES_3D:
        mesh = build_posed_human(pose_name)
        bb = get_bounding_box(mesh)
        bb_vol = get_bb_volume(mesh)
        hull_vol = mesh.convex_hull.volume
        eff = packing_efficiency_3d(mesh)

        count_basic, _, _ = pack_3d_grid(mesh, CONTAINER)
        count_rot, _, _, _ = pack_3d_rotation_search(mesh, CONTAINER, angle_steps=10)
        count = max(count_basic, count_rot)

        data.append({
            "name": pose_name,
            "bb_vol": bb_vol,
            "hull_vol": hull_vol,
            "efficiency": eff,
            "container_count": count,
            "bb_dims": bb,
        })

    data.sort(key=lambda d: d["container_count"], reverse=True)

    print(f"\n  {'Rank':<5} {'Pose':<28} {'BB Vol':>8} {'Eff':>6} {'Container':>10}")
    print("  " + "-" * 65)
    for i, d in enumerate(data):
        print(f"  {i+1:<5} {d['name']:<28} {d['bb_vol']:>7.3f}m3 {d['efficiency']:>5.0%} {d['container_count']:>10}")

    with open(f"{OUTPUT_DIR}/efficiency_ranking.json", "w") as f:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for d in data], f, indent=2)

    return data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  OPTIMAL ANTHROPOMORPHIC VOLUMETRIC PACKING (3D)")
    print("  Solid Meshes | Rotation Search | Articulated Rig")
    print("=" * 60)
    print()

    experiment_1_pose_gallery()
    experiment_2_packing_comparison()
    experiment_3_packing_visualization()
    experiment_4_tpose_tax_3d()
    experiment_5_efficiency_ranking()

    print("\n" + "=" * 60)
    print("All 3D experiments complete. Results saved to ./results3d/")
    print("=" * 60)
