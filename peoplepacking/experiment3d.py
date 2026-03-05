"""
OPTIMAL ANTHROPOMORPHIC VOLUMETRIC PACKING:
A Rigorous 3D Analysis of Human Stacking Configurations

Full 3D experiment with articulated human meshes,
volumetric packing, and proper 3D visualization.
"""

import numpy as np
import trimesh
import json
import os

from human3d import (
    build_posed_human, POSES_3D, HumanRig,
    get_bounding_box, get_bb_volume, packing_efficiency_3d,
)

# Use matplotlib for 3D visualization (more portable than pyvista for saving)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUTPUT_DIR = "results3d"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Visualization helpers
# ============================================================

def mesh_to_poly3d(mesh, color='steelblue', alpha=0.6):
    """Convert a trimesh to matplotlib Poly3DCollection."""
    verts = mesh.vertices
    faces = mesh.faces
    poly3d = Poly3DCollection(
        [verts[face] for face in faces],
        alpha=alpha,
        facecolor=color,
        edgecolor='none',
    )
    return poly3d


def plot_mesh_3d(mesh, title="", ax=None, color='steelblue', alpha=0.6):
    """Plot a single mesh in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    poly3d = mesh_to_poly3d(mesh, color=color, alpha=alpha)
    ax.add_collection3d(poly3d)

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    max_range = (bounds[1] - bounds[0]).max() / 2

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    return ax


def simplify_mesh_for_plot(mesh, max_faces=500):
    """Reduce face count for faster matplotlib rendering."""
    if len(mesh.faces) > max_faces:
        # Subsample faces uniformly
        indices = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=int)
        return trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[indices],
            process=False,
        )
    return mesh


def draw_wireframe_box(ax, origin, dims, color='red', linestyle='--', alpha=0.3):
    """Draw a wireframe rectangular box."""
    x0, y0, z0 = origin
    dx, dy, dz = dims

    # 12 edges of a box
    for s, e in [
        ([x0,y0,z0], [x0+dx,y0,z0]),
        ([x0,y0,z0], [x0,y0+dy,z0]),
        ([x0,y0,z0], [x0,y0,z0+dz]),
        ([x0+dx,y0+dy,z0], [x0+dx,y0,z0]),
        ([x0+dx,y0+dy,z0], [x0,y0+dy,z0]),
        ([x0+dx,y0+dy,z0], [x0+dx,y0+dy,z0+dz]),
        ([x0,y0,z0+dz], [x0+dx,y0,z0+dz]),
        ([x0,y0,z0+dz], [x0,y0+dy,z0+dz]),
        ([x0+dx,y0,z0+dz], [x0+dx,y0+dy,z0+dz]),
        ([x0,y0+dy,z0+dz], [x0+dx,y0+dy,z0+dz]),
        ([x0+dx,y0,z0], [x0+dx,y0,z0+dz]),
        ([x0,y0+dy,z0], [x0,y0+dy,z0+dz]),
    ]:
        ax.plot3D(*zip(s, e), color=color, linestyle=linestyle, alpha=alpha, linewidth=1)


# ============================================================
# 3D Packing
# ============================================================

def pack_3d_grid(mesh, container_dims):
    """
    Pack copies of mesh bounding box into container.
    Try all 6 orientations (3 axes x 2 flips per axis don't matter for AABB).

    container_dims: (Lx, Ly, Lz)
    Returns: (count, list_of_offsets, bb_dims_used)
    """
    bb = get_bounding_box(mesh)
    Lx, Ly, Lz = container_dims

    best_count = 0
    best_offsets = []
    best_bb = bb

    # Try all 6 permutations of (w, d, h) orientation
    import itertools
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


# ============================================================
# Experiments
# ============================================================

def experiment_1_pose_gallery():
    """Render all poses and compute 3D metrics."""
    print("=" * 60)
    print("EXPERIMENT 1: 3D Pose Gallery & Metrics")
    print("=" * 60)

    results = {}
    n_poses = len(POSES_3D)
    cols = 4
    rows = (n_poses + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.suptitle("Human Pose Gallery (3D Articulated Model)", fontsize=16, fontweight='bold')

    for i, (name, pose) in enumerate(POSES_3D.items()):
        mesh = build_posed_human(name)
        mesh_simple = simplify_mesh_for_plot(mesh, max_faces=400)

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

        print(f"\n  {name}:")
        print(f"    BB: {bb_dims[0]:.2f} x {bb_dims[1]:.2f} x {bb_dims[2]:.2f} m")
        print(f"    BB Vol: {bb_vol:.3f} m^3")
        print(f"    Hull Vol: {hull_vol:.3f} m^3")
        print(f"    Efficiency: {eff:.1%}")

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        poly3d = mesh_to_poly3d(mesh_simple, color='steelblue', alpha=0.5)
        ax.add_collection3d(poly3d)

        # BB wireframe
        bounds = mesh.bounds
        draw_wireframe_box(ax, bounds[0], bb_dims)

        center = (bounds[0] + bounds[1]) / 2
        max_range = bb_dims.max() / 2 * 1.2
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(bounds[0][2] - 0.05, bounds[1][2] + 0.05)
        ax.set_title(f"{name}\nEff: {eff:.0%}  Vol: {bb_vol:.2f}m3", fontsize=8)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.tick_params(labelsize=5)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pose_gallery_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved pose_gallery_3d.png")

    with open(f"{OUTPUT_DIR}/pose_metrics_3d.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def experiment_2_packing_comparison():
    """Compare how many humans fit in venues, 3D edition."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: 3D Venue Packing Comparison")
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
            count, offsets, bb_used = pack_3d_grid(mesh, dims)
            venue_results[pose_name] = count

        best_pose = max(venue_results, key=venue_results.get)
        for pose_name, count in venue_results.items():
            marker = " <-- BEST" if pose_name == best_pose else ""
            print(f"    {pose_name}: {count}{marker}")

        all_results[venue_name] = venue_results

    # Big comparison chart
    fig, ax = plt.subplots(figsize=(18, 9))

    venue_names = list(VENUES.keys())
    pose_names = list(POSES_3D.keys())
    x = np.arange(len(venue_names))
    width = 0.8 / len(pose_names)

    colors = plt.cm.tab20(np.linspace(0, 1, len(pose_names)))

    for i, pose_name in enumerate(pose_names):
        counts = [all_results[v].get(pose_name, 0) for v in venue_names]
        bars = ax.bar(x + i * width, counts, width, label=pose_name,
                       color=colors[i], edgecolor='black', linewidth=0.3)

    ax.set_xlabel("Venue")
    ax.set_ylabel("Number of Humans (log scale)")
    ax.set_title("Optimal Human Packing: 3D Volumetric Analysis by Venue and Pose")
    ax.set_xticks(x + width * (len(pose_names) - 1) / 2)
    ax.set_xticklabels(venue_names, rotation=30, ha='right')
    ax.legend(fontsize=6, loc='upper left', ncol=2)
    ax.set_yscale('symlog', linthresh=1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/venue_comparison_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved venue_comparison_3d.png")

    # Summary table
    print("\n  GRAND SUMMARY (3D):")
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
    """
    Render 3D packing scenes for the most interesting venue/pose combos.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: 3D Packing Visualizations")
    print("=" * 60)

    SCENES = [
        ("Elevator", (2.0, 1.5, 2.4), "Fetal Position"),
        ("Elevator", (2.0, 1.5, 2.4), "T-Pose"),
        ("Shipping Container", (5.9, 2.35, 2.39), "Fetal Position"),
        ("Shipping Container", (5.9, 2.35, 2.39), "Pike (folded)"),
        ("Phone Booth", (0.9, 0.9, 2.3), "Fetal Position"),
        ("Hot Tub", (2.1, 2.1, 0.9), "Planking"),
        ("Minivan", (2.0, 1.5, 1.3), "Naruto Run"),
        ("School Bus", (7.3, 2.3, 1.8), "Dab"),
    ]

    for venue_name, dims, pose_name in SCENES:
        print(f"\n  Rendering: {venue_name} x {pose_name}")

        mesh = build_posed_human(pose_name)
        count, offsets, bb_used = pack_3d_grid(mesh, dims)
        print(f"    Packed: {count} humans")

        if count == 0:
            print(f"    (No humans fit - skipping)")
            continue

        # Build scene
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Color palette
        n_colors = min(count, 20)
        colors = plt.cm.Set3(np.linspace(0, 1, n_colors))

        # We need to shift mesh to origin-based BB first
        mesh_origin = mesh.copy()
        mesh_origin.vertices -= mesh.bounds[0]

        # Simplify for rendering
        mesh_simple = simplify_mesh_for_plot(mesh_origin, max_faces=200)

        max_to_render = min(count, 60)  # Cap for rendering speed

        for i, offset in enumerate(offsets[:max_to_render]):
            shifted = mesh_simple.copy()
            shifted.vertices += offset
            color = colors[i % n_colors]
            poly3d = mesh_to_poly3d(shifted, color=color, alpha=0.4)
            ax.add_collection3d(poly3d)

        # Container wireframe
        draw_wireframe_box(ax, [0, 0, 0], dims, color='black', linestyle='-', alpha=0.8)

        ax.set_xlim(0, dims[0])
        ax.set_ylim(0, dims[1])
        ax.set_zlim(0, dims[2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        vol_eff = count * mesh.convex_hull.volume / (dims[0] * dims[1] * dims[2])
        ax.set_title(
            f"{venue_name}: {count} humans in {pose_name}\n"
            f"Container: {dims[0]}x{dims[1]}x{dims[2]}m | "
            f"Vol. Efficiency: {vol_eff:.0%}",
            fontsize=11
        )

        safe = f"{venue_name}_{pose_name}".replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/scene_{safe}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved scene_{safe}.png")


def experiment_4_tpose_tax_3d():
    """The T-Pose Tax, now in glorious 3D."""
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

    # Sort by tax
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
    """Rank all poses by various 3D metrics and produce a definitive leaderboard."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Definitive Pose Efficiency Ranking")
    print("=" * 60)

    data = []
    for pose_name in POSES_3D:
        mesh = build_posed_human(pose_name)
        bb = get_bounding_box(mesh)
        bb_vol = get_bb_volume(mesh)
        hull_vol = mesh.convex_hull.volume
        eff = packing_efficiency_3d(mesh)

        # How many fit in a standard shipping container
        count, _, _ = pack_3d_grid(mesh, (5.9, 2.35, 2.39))

        data.append({
            "name": pose_name,
            "bb_vol": bb_vol,
            "hull_vol": hull_vol,
            "efficiency": eff,
            "container_count": count,
            "bb_dims": bb,
        })

    # Sort by container count
    data.sort(key=lambda d: d["container_count"], reverse=True)

    print(f"\n  {'Rank':<5} {'Pose':<28} {'BB Vol':>8} {'Eff':>6} {'Container':>10}")
    print("  " + "-" * 65)
    for i, d in enumerate(data):
        print(f"  {i+1:<5} {d['name']:<28} {d['bb_vol']:>7.3f}m3 {d['efficiency']:>5.0%} {d['container_count']:>10}")

    # Save
    with open(f"{OUTPUT_DIR}/efficiency_ranking.json", "w") as f:
        json.dump([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()} for d in data], f, indent=2)

    return data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  OPTIMAL ANTHROPOMORPHIC VOLUMETRIC PACKING (3D)")
    print("  Now With Actual Human Meshes")
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
