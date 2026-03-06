"""
THE TRIANGLE THEOREM: On the Packing Implications of
PS1-Era Polygonal Thorax Geometry

Compares packing density of a classic PS1 Lara Croft model
(triangular chest geometry) vs the standard capsule human and
a "modern Lara" (smooth geometry, same proportions).

Key question: Do the triangle tits actually matter for packing?
"""

import numpy as np
import trimesh
import json
import os

from human3d import (
    HumanRig, build_posed_human, POSES_3D,
    get_bounding_box, get_bb_volume, packing_efficiency_3d,
    rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
    make_capsule, make_ellipsoid,
)
from experiment3d import (
    render_mesh, render_packing_scene, trimesh_to_pyvista,
    pack_3d_grid, pack_3d_rotation_search, apply_rotation_to_mesh,
    OUTPUT_DIR, SKIN_TONE,
)

import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# PS1 Lara Croft Rig
# ============================================================

def make_cone(radius, height, sections=8):
    """Create a cone mesh (low-poly for PS1 aesthetic)."""
    return trimesh.creation.cone(radius=radius, height=height, sections=sections)


class LaraRig(HumanRig):
    """
    PS1-era Low-Poly Archaeologist with triangular thorax geometry.

    Key differences from standard HumanRig:
    - Narrower waist, wider hips (classic Lara proportions)
    - TWO CONICAL primitives on chest (the iconic triangles)
    - Optional ponytail (long capsule from head)
    - Lower polygon count for authentic PS1 aesthetic
    """

    # Lara proportions: narrower waist, slightly wider hips, longer legs
    BODY_PARTS = {
        "pelvis":       {"type": "ellipsoid", "rx": 0.17, "ry": 0.12, "rz": 0.09},
        "spine":        {"type": "capsule", "radius": 0.10, "height": 0.18},  # narrower waist
        "chest":        {"type": "ellipsoid", "rx": 0.18, "ry": 0.10, "rz": 0.15},  # flatter chest base
        "neck":         {"type": "capsule", "radius": 0.04, "height": 0.10},
        "head":         {"type": "ellipsoid", "rx": 0.08, "ry": 0.09, "rz": 0.09},
        "l_upper_arm":  {"type": "capsule", "radius": 0.035, "height": 0.17},
        "l_forearm":    {"type": "capsule", "radius": 0.028, "height": 0.154},
        "l_hand":       {"type": "ellipsoid", "rx": 0.035, "ry": 0.025, "rz": 0.05},
        "r_upper_arm":  {"type": "capsule", "radius": 0.035, "height": 0.17},
        "r_forearm":    {"type": "capsule", "radius": 0.028, "height": 0.154},
        "r_hand":       {"type": "ellipsoid", "rx": 0.035, "ry": 0.025, "rz": 0.05},
        "l_thigh":      {"type": "capsule", "radius": 0.065, "height": 0.28},   # longer legs
        "l_shin":       {"type": "capsule", "radius": 0.045, "height": 0.30},
        "l_foot":       {"type": "ellipsoid", "rx": 0.045, "ry": 0.09, "rz": 0.035},
        "r_thigh":      {"type": "capsule", "radius": 0.065, "height": 0.28},
        "r_shin":       {"type": "capsule", "radius": 0.045, "height": 0.30},
        "r_foot":       {"type": "ellipsoid", "rx": 0.045, "ry": 0.09, "rz": 0.035},
        # Extra parts unique to Lara
        "ponytail":     {"type": "capsule", "radius": 0.025, "height": 0.25},
    }

    # Triangular breast geometry parameters
    TRIANGLE_RADIUS = 0.07   # base radius of each cone
    TRIANGLE_HEIGHT = 0.12   # how far they protrude forward
    TRIANGLE_SPREAD = 0.08   # X distance from centerline

    MESH_OFFSETS = {
        "l_upper_arm":  [0, 0, -0.125],
        "r_upper_arm":  [0, 0, -0.125],
        "l_forearm":    [0, 0, -0.11],
        "r_forearm":    [0, 0, -0.11],
        "l_thigh":      [0, 0, -0.21],
        "r_thigh":      [0, 0, -0.21],
        "l_shin":       [0, 0, -0.20],
        "r_shin":       [0, 0, -0.20],
        "l_foot":       [0, -0.02, -0.03],
        "r_foot":       [0, -0.02, -0.03],
        "ponytail":     [0, -0.04, -0.14],  # hangs down behind head
    }

    SKELETON = {
        "pelvis":       {"parent": None,          "offset": [0, 0, 0.93]},
        "spine":        {"parent": "pelvis",      "offset": [0, 0, 0.10]},
        "chest":        {"parent": "spine",       "offset": [0, 0, 0.20]},
        "neck":         {"parent": "chest",       "offset": [0, 0, 0.16]},
        "head":         {"parent": "neck",        "offset": [0, 0, 0.12]},
        "l_upper_arm":  {"parent": "chest",       "offset": [-0.20, 0, 0.06]},
        "l_forearm":    {"parent": "l_upper_arm", "offset": [0, 0, -0.25]},
        "l_hand":       {"parent": "l_forearm",   "offset": [0, 0, -0.22]},
        "r_upper_arm":  {"parent": "chest",       "offset": [0.20, 0, 0.06]},
        "r_forearm":    {"parent": "r_upper_arm", "offset": [0, 0, -0.25]},
        "r_hand":       {"parent": "r_forearm",   "offset": [0, 0, -0.22]},
        "l_thigh":      {"parent": "pelvis",      "offset": [-0.10, 0, -0.06]},
        "l_shin":       {"parent": "l_thigh",     "offset": [0, 0, -0.42]},
        "l_foot":       {"parent": "l_shin",      "offset": [0, 0.08, -0.40]},
        "r_thigh":      {"parent": "pelvis",      "offset": [0.10, 0, -0.06]},
        "r_shin":       {"parent": "r_thigh",     "offset": [0, 0, -0.42]},
        "r_foot":       {"parent": "r_shin",      "offset": [0, 0.08, -0.40]},
        "ponytail":     {"parent": "head",        "offset": [0, -0.06, 0.02]},
    }

    def __init__(self, scale=1.0, triangles=True):
        super().__init__(scale=scale)
        self.triangles = triangles

    def _make_triangle_breast(self):
        """Create a single PS1-era triangular breast cone (low-poly)."""
        cone = make_cone(
            radius=self.TRIANGLE_RADIUS * self.scale,
            height=self.TRIANGLE_HEIGHT * self.scale,
            sections=4,  # 4 sides = pyramid = maximum PS1 energy
        )
        # Cone is created along +Z from z=0 to z=height.
        # Rotate 90 degrees around X to point in +Y direction (forward).
        R = rotation_matrix_x(90)
        T = np.eye(4)
        T[:3, :3] = R
        cone.apply_transform(T)
        return cone

    def build(self, pose=None):
        """Build Lara mesh, optionally with triangular chest geometry."""
        # Build base body using parent's method
        base_mesh = super().build(pose=pose)

        if not self.triangles:
            return base_mesh

        # Add triangular breasts attached to chest
        # Need to compute chest transform
        if pose is None:
            pose = {}

        transforms = {}

        def compute_transform(part_name):
            if part_name in transforms:
                return transforms[part_name]
            skel = self.SKELETON[part_name]
            parent = skel["parent"]
            offset = np.array(skel["offset"]) * self.scale
            if parent is None:
                parent_transform = np.eye(4)
            else:
                parent_transform = compute_transform(parent)
            local_rot = np.eye(3)
            if part_name in pose:
                rx, ry, rz = pose[part_name]
                local_rot = rotation_matrix_z(rz) @ rotation_matrix_y(ry) @ rotation_matrix_x(rx)
            T = np.eye(4)
            T[:3, :3] = parent_transform[:3, :3] @ local_rot
            T[:3, 3] = parent_transform[:3, 3] + parent_transform[:3, :3] @ offset
            transforms[part_name] = T
            return T

        chest_T = compute_transform("chest")

        meshes = [base_mesh]

        for side in [-1, 1]:  # left, right
            cone = self._make_triangle_breast()
            # Position on chest: offset in X for side, forward in Y
            local_offset = np.array([
                side * self.TRIANGLE_SPREAD * self.scale,
                0.10 * self.scale,    # forward from chest center
                0.02 * self.scale,    # slightly above chest center
            ])
            cone.vertices += local_offset
            cone.apply_transform(chest_T)
            meshes.append(cone)

        return trimesh.util.concatenate(meshes)


class SmoothLaraRig(LaraRig):
    """Modern Lara: same proportions but smooth (ellipsoid) chest, no triangles."""

    def __init__(self, scale=1.0):
        super().__init__(scale=scale, triangles=False)

    def build(self, pose=None):
        """Build with smooth hemispheres instead of triangles."""
        base_mesh = LaraRig.build(self, pose)  # triangles=False, so just base

        if pose is None:
            pose = {}

        transforms = {}

        def compute_transform(part_name):
            if part_name in transforms:
                return transforms[part_name]
            skel = self.SKELETON[part_name]
            parent = skel["parent"]
            offset = np.array(skel["offset"]) * self.scale
            if parent is None:
                parent_transform = np.eye(4)
            else:
                parent_transform = compute_transform(parent)
            local_rot = np.eye(3)
            if part_name in pose:
                rx, ry, rz = pose[part_name]
                local_rot = rotation_matrix_z(rz) @ rotation_matrix_y(ry) @ rotation_matrix_x(rx)
            T = np.eye(4)
            T[:3, :3] = parent_transform[:3, :3] @ local_rot
            T[:3, 3] = parent_transform[:3, 3] + parent_transform[:3, :3] @ offset
            transforms[part_name] = T
            return T

        chest_T = compute_transform("chest")

        meshes = [base_mesh]
        for side in [-1, 1]:
            # Smooth hemisphere instead of cone
            sphere = make_ellipsoid(
                0.06 * self.scale,  # rx
                0.06 * self.scale,  # ry (same as rx = smooth)
                0.05 * self.scale,  # rz
            )
            local_offset = np.array([
                side * self.TRIANGLE_SPREAD * self.scale,
                0.10 * self.scale,
                0.02 * self.scale,
            ])
            sphere.vertices += local_offset
            sphere.apply_transform(chest_T)
            meshes.append(sphere)

        return trimesh.util.concatenate(meshes)


def build_lara(pose_name, triangles=True, scale=1.0):
    """Build PS1 Lara (triangles=True) or Modern Lara (triangles=False)."""
    if triangles:
        rig = LaraRig(scale=scale, triangles=True)
    else:
        rig = SmoothLaraRig(scale=scale)
    # Check Lara-specific poses first, then fall back to standard poses
    pose = LARA_POSES.get(pose_name, POSES_3D.get(pose_name, {}))
    return rig.build(pose=pose)


# ============================================================
# Lara-specific poses
# ============================================================

LARA_POSES = {
    "Standing (arms at sides)": POSES_3D["Standing (arms at sides)"],
    "T-Pose": POSES_3D["T-Pose"],
    "Dual Pistols": {
        # Iconic pose: both arms forward, slightly spread
        "l_upper_arm": (70, 30, 0),
        "r_upper_arm": (70, -30, 0),
        "l_forearm":   (-20, 0, 0),
        "r_forearm":   (-20, 0, 0),
    },
    "Handstand": {
        # Upside down - arms straight, body inverted
        "pelvis":      (180, 0, 0),  # flip upside down
    },
    "Crouch": {
        # Classic stealth crouch
        "l_thigh":     (90, 10, 0),
        "r_thigh":     (90, -10, 0),
        "l_shin":      (-80, 0, 0),
        "r_shin":      (-80, 0, 0),
        "spine":       (20, 0, 0),
        "chest":       (10, 0, 0),
    },
    "Swan Dive": {
        # Arms back, body leaning forward
        "pelvis":      (70, 0, 0),     # body tilted forward
        "l_upper_arm": (-40, 60, 0),   # arms swept back
        "r_upper_arm": (-40, -60, 0),
    },
    "Planking": POSES_3D["Planking"],
    "Coffin Dance": POSES_3D["Coffin Dance"],
    "Fetal Position": POSES_3D["Fetal Position"],
}


# ============================================================
# Experiments
# ============================================================

VENUES = {
    "Elevator":           (2.0, 1.5, 2.4),
    "Phone Booth":        (0.9, 0.9, 2.3),
    "Shipping Container": (5.9, 2.35, 2.39),
    "Minivan":            (2.0, 1.5, 1.3),
    "Boeing 737 Cabin":   (28.0, 3.54, 2.2),
    "Subway Car":         (15.5, 2.5, 2.1),
    "School Bus":         (7.3, 2.3, 1.8),
    "Hot Tub":            (2.1, 2.1, 0.9),
    # Lara-specific venues
    "Tomb Corridor":      (12.0, 1.2, 2.5),  # Long narrow tomb passage
    "Sarcophagus":        (2.1, 0.7, 0.8),    # Stone coffin
}


def experiment_lara_gallery():
    """Render PS1 Lara in key poses."""
    print("=" * 60)
    print("LARA EXPERIMENT 1: PS1-Era Pose Gallery")
    print("=" * 60)

    for name in LARA_POSES:
        mesh_tri = build_lara(name, triangles=True)
        mesh_smooth = build_lara(name, triangles=False)

        bb_tri = get_bounding_box(mesh_tri)
        bb_smooth = get_bounding_box(mesh_smooth)
        vol_tri = get_bb_volume(mesh_tri)
        vol_smooth = get_bb_volume(mesh_smooth)
        diff_pct = (vol_tri - vol_smooth) / vol_smooth * 100 if vol_smooth > 0 else 0

        print(f"  {name}:")
        print(f"    Triangle: BB={bb_tri[0]:.3f}x{bb_tri[1]:.3f}x{bb_tri[2]:.3f}  vol={vol_tri:.4f}")
        print(f"    Smooth:   BB={bb_smooth[0]:.3f}x{bb_smooth[1]:.3f}x{bb_smooth[2]:.3f}  vol={vol_smooth:.4f}")
        print(f"    Delta:    {diff_pct:+.1f}% BB volume")

        safe = name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        render_mesh(mesh_tri, f"{OUTPUT_DIR}/lara_ps1_{safe}.png",
                    title=f"PS1 Lara: {name}", color='#D4956A')
        render_mesh(mesh_smooth, f"{OUTPUT_DIR}/lara_modern_{safe}.png",
                    title=f"Modern Lara: {name}", color='#D4956A')


def experiment_triangle_packing():
    """The main event: where do triangles actually matter?"""
    print("\n" + "=" * 60)
    print("LARA EXPERIMENT 2: Triangle Geometry Packing Impact")
    print("=" * 60)

    results = {}

    for venue_name, dims in VENUES.items():
        print(f"\n  {venue_name}: {dims[0]}x{dims[1]}x{dims[2]}m")
        venue_results = {}

        for pose_name in LARA_POSES:
            mesh_tri = build_lara(pose_name, triangles=True)
            mesh_smooth = build_lara(pose_name, triangles=False)
            mesh_standard = build_posed_human(pose_name)

            # Pack all three
            ct_basic, _, _ = pack_3d_grid(mesh_tri, dims)
            ct_rot, _, _, _ = pack_3d_rotation_search(mesh_tri, dims, angle_steps=12)
            count_tri = max(ct_basic, ct_rot)

            cs_basic, _, _ = pack_3d_grid(mesh_smooth, dims)
            cs_rot, _, _, _ = pack_3d_rotation_search(mesh_smooth, dims, angle_steps=12)
            count_smooth = max(cs_basic, cs_rot)

            cn_basic, _, _ = pack_3d_grid(mesh_standard, dims)
            cn_rot, _, _, _ = pack_3d_rotation_search(mesh_standard, dims, angle_steps=12)
            count_standard = max(cn_basic, cn_rot)

            delta = count_tri - count_smooth
            marker = ""
            if delta != 0:
                marker = f" *** TRIANGLE EFFECT: {delta:+d} ***"

            venue_results[pose_name] = {
                "ps1_triangle": count_tri,
                "modern_smooth": count_smooth,
                "standard_human": count_standard,
                "triangle_delta": delta,
            }

            print(f"    {pose_name:<28} Tri:{count_tri:>4}  Smooth:{count_smooth:>4}  "
                  f"Std:{count_standard:>4}{marker}")

        results[venue_name] = venue_results

    # Find ALL cases where triangles make a difference
    print("\n" + "=" * 60)
    print("  TRIANGLE IMPACT SUMMARY")
    print("=" * 60)

    triangle_effects = []
    for venue_name, venue_data in results.items():
        for pose_name, data in venue_data.items():
            if data["triangle_delta"] != 0:
                triangle_effects.append({
                    "venue": venue_name,
                    "pose": pose_name,
                    "delta": data["triangle_delta"],
                    "tri": data["ps1_triangle"],
                    "smooth": data["modern_smooth"],
                })

    if triangle_effects:
        print(f"\n  Found {len(triangle_effects)} venue/pose combinations where triangles matter:\n")
        for effect in sorted(triangle_effects, key=lambda e: abs(e["delta"]), reverse=True):
            direction = "BETTER" if effect["delta"] > 0 else "WORSE"
            print(f"  {effect['venue']:<22} {effect['pose']:<28} "
                  f"Tri={effect['tri']:>4}  Smooth={effect['smooth']:>4}  "
                  f"Delta={effect['delta']:+d} ({direction})")
    else:
        print("\n  NO DIFFERENCES FOUND - triangles are purely cosmetic for AABB packing!")
        print("  (Which is itself a notable finding for the paper)")

    with open(f"{OUTPUT_DIR}/lara_packing.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, triangle_effects


def experiment_triangle_rotation_analysis():
    """
    Detailed rotation-angle analysis: at which specific angles
    does the triangle geometry create a different bounding box?
    """
    print("\n" + "=" * 60)
    print("LARA EXPERIMENT 3: Rotation-Angle BB Analysis")
    print("=" * 60)

    from experiment3d import rotated_bb

    pose_name = "Standing (arms at sides)"
    mesh_tri = build_lara(pose_name, triangles=True)
    mesh_smooth = build_lara(pose_name, triangles=False)

    angles = np.linspace(0, 180, 36, endpoint=False)  # 5-degree steps

    bb_diffs = []
    max_diff = 0
    max_diff_angle = (0, 0)

    for rx in angles:
        for ry in angles:
            bb_tri, _ = rotated_bb(mesh_tri, rx, ry, 0)
            bb_smooth, _ = rotated_bb(mesh_smooth, rx, ry, 0)

            vol_tri = bb_tri[0] * bb_tri[1] * bb_tri[2]
            vol_smooth = bb_smooth[0] * bb_smooth[1] * bb_smooth[2]

            diff = vol_tri - vol_smooth
            bb_diffs.append({
                "rx": rx, "ry": ry,
                "vol_tri": vol_tri, "vol_smooth": vol_smooth,
                "diff": diff,
            })

            if abs(diff) > abs(max_diff):
                max_diff = diff
                max_diff_angle = (rx, ry)

    print(f"  Max BB volume difference: {max_diff:.6f} m3 at rx={max_diff_angle[0]:.0f}, ry={max_diff_angle[1]:.0f}")
    print(f"  (Triangle {'larger' if max_diff > 0 else 'smaller'} than smooth)")

    # Find angles where triangle BB is SMALLER than smooth (the interesting case)
    better_angles = [d for d in bb_diffs if d["diff"] < -0.001]
    if better_angles:
        print(f"\n  Found {len(better_angles)} rotation angles where triangles produce SMALLER BB!")
        best = min(better_angles, key=lambda d: d["diff"])
        print(f"  Best: rx={best['rx']:.0f}, ry={best['ry']:.0f}: "
              f"tri={best['vol_tri']:.4f} vs smooth={best['vol_smooth']:.4f} "
              f"({best['diff']:.4f} m3)")
    else:
        print("\n  No rotation angles where triangles are smaller.")

    # Heatmap of BB volume difference across rotation angles
    n = len(angles)
    diff_matrix = np.zeros((n, n))
    for d in bb_diffs:
        ix = int(d["rx"] / 5)
        iy = int(d["ry"] / 5)
        if ix < n and iy < n:
            diff_matrix[ix, iy] = d["diff"]

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    im = ax.imshow(diff_matrix * 1000, cmap='RdBu_r', aspect='auto',
                   extent=[0, 180, 180, 0], vmin=-vmax*1000, vmax=vmax*1000)
    ax.set_xlabel("Ry (yaw) degrees", fontsize=12)
    ax.set_ylabel("Rx (pitch) degrees", fontsize=12)
    ax.set_title("BB Volume Difference: Triangle - Smooth (mL)\n"
                 "Blue = triangles produce smaller BB (better for packing)",
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Volume difference (mL)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lara_rotation_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved lara_rotation_analysis.png")

    return bb_diffs


def experiment_triangle_visualization():
    """Render comparison scenes."""
    print("\n" + "=" * 60)
    print("LARA EXPERIMENT 4: Visual Comparison")
    print("=" * 60)

    from PIL import Image

    # Render PS1 vs Modern Lara side by side for key poses
    key_poses = ["Standing (arms at sides)", "Dual Pistols", "Crouch", "Swan Dive"]

    for pose_name in key_poses:
        safe = pose_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        # Already rendered in experiment 1, just make comparison grid
        pass

    # Make a 2x4 grid: top row PS1, bottom row Modern
    cols = 4
    cell_w, cell_h = 600, 600
    grid = Image.new('RGB', (cols * cell_w, 2 * cell_h), 'white')

    for i, pose_name in enumerate(key_poses):
        safe = pose_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        for row, variant in enumerate(["ps1", "modern"]):
            path = f"{OUTPUT_DIR}/lara_{variant}_{safe}.png"
            try:
                img = Image.open(path).resize((cell_w, cell_h))
                grid.paste(img, (i * cell_w, row * cell_h))
            except Exception as e:
                print(f"  Failed: {path}: {e}")

    grid.save(f"{OUTPUT_DIR}/lara_comparison_grid.png")
    print("  Saved lara_comparison_grid.png")

    # Packing scene: PS1 Lara in Tomb Corridor
    dims = VENUES["Tomb Corridor"]
    mesh = build_lara("Standing (arms at sides)", triangles=True)
    count_basic, offsets_basic, _ = pack_3d_grid(mesh, dims)
    count_rot, offsets_rot, _, R_rot = pack_3d_rotation_search(mesh, dims, angle_steps=12)

    if count_rot > count_basic:
        count, offsets = count_rot, offsets_rot
        mesh_packed = apply_rotation_to_mesh(mesh, R_rot)
    else:
        count, offsets = count_basic, offsets_basic
        mesh_packed = mesh.copy()
        mesh_packed.vertices -= mesh_packed.vertices.min(axis=0)

    if count > 0:
        meshes_with_offsets = [(mesh_packed, o) for o in offsets]
        render_packing_scene(
            meshes_with_offsets, dims,
            f"{OUTPUT_DIR}/lara_tomb_corridor_packing.png",
            title=f"Tomb Corridor: {count} PS1 Laras packed",
        )
        print(f"  Saved tomb corridor scene ({count} Laras)")

    # Sarcophagus packing
    dims = VENUES["Sarcophagus"]
    for label, tri in [("PS1", True), ("Modern", False)]:
        mesh = build_lara("Coffin Dance", triangles=tri)
        c1, _, _ = pack_3d_grid(mesh, dims)
        c2, off2, _, R2 = pack_3d_rotation_search(mesh, dims, angle_steps=12)
        count = max(c1, c2)
        print(f"  Sarcophagus ({label}): {count} Laras in Coffin Dance pose")


def make_summary_chart(results, triangle_effects):
    """Create a summary visualization of the triangle impact."""
    print("\n  Creating summary chart...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: BB volume comparison across poses
    ax1 = axes[0]
    poses = list(LARA_POSES.keys())
    tri_vols = []
    smooth_vols = []
    for pose_name in poses:
        mesh_tri = build_lara(pose_name, triangles=True)
        mesh_smooth = build_lara(pose_name, triangles=False)
        tri_vols.append(get_bb_volume(mesh_tri))
        smooth_vols.append(get_bb_volume(mesh_smooth))

    x = np.arange(len(poses))
    width = 0.35
    bars1 = ax1.barh(x - width/2, tri_vols, width, label='PS1 (Triangles)',
                     color='#E74C3C', edgecolor='white', linewidth=0.5)
    bars2 = ax1.barh(x + width/2, smooth_vols, width, label='Modern (Smooth)',
                     color='#3498DB', edgecolor='white', linewidth=0.5)

    ax1.set_yticks(x)
    ax1.set_yticklabels(poses, fontsize=9)
    ax1.set_xlabel("Bounding Box Volume (m\u00b3)")
    ax1.set_title("BB Volume: PS1 vs Modern Geometry", fontsize=14, fontweight='bold')
    ax1.legend(framealpha=0.9, edgecolor='#CCCCCC')

    # Annotate differences
    for i, (tv, sv) in enumerate(zip(tri_vols, smooth_vols)):
        if tv != sv:
            diff_pct = (tv - sv) / sv * 100
            ax1.text(max(tv, sv) + 0.005, i,
                     f'{diff_pct:+.1f}%', va='center', fontsize=7, color='#555555')

    # Right: packing count comparison for Shipping Container
    ax2 = axes[1]
    container = VENUES["Shipping Container"]
    tri_counts = []
    smooth_counts = []
    std_counts = []
    for pose_name in poses:
        if "Shipping Container" in results and pose_name in results["Shipping Container"]:
            data = results["Shipping Container"][pose_name]
            tri_counts.append(data["ps1_triangle"])
            smooth_counts.append(data["modern_smooth"])
            std_counts.append(data["standard_human"])
        else:
            tri_counts.append(0)
            smooth_counts.append(0)
            std_counts.append(0)

    width = 0.25
    ax2.barh(x - width, tri_counts, width, label='PS1 Lara (Triangles)',
             color='#E74C3C', edgecolor='white', linewidth=0.5)
    ax2.barh(x, smooth_counts, width, label='Modern Lara (Smooth)',
             color='#3498DB', edgecolor='white', linewidth=0.5)
    ax2.barh(x + width, std_counts, width, label='Standard Human',
             color='#95A5A6', edgecolor='white', linewidth=0.5)

    ax2.set_yticks(x)
    ax2.set_yticklabels(poses, fontsize=9)
    ax2.set_xlabel("Humans per Shipping Container")
    ax2.set_title("Packing Count: Triangle vs Smooth vs Standard",
                  fontsize=14, fontweight='bold')
    ax2.legend(framealpha=0.9, edgecolor='#CCCCCC', fontsize=8)

    plt.tight_layout(w_pad=3)
    plt.savefig(f"{OUTPUT_DIR}/lara_summary.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved lara_summary.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  THE TRIANGLE THEOREM")
    print("  PS1-Era Polygonal Thorax Geometry Analysis")
    print("=" * 60)
    print()

    experiment_lara_gallery()
    results, effects = experiment_triangle_packing()
    experiment_triangle_rotation_analysis()
    experiment_triangle_visualization()
    make_summary_chart(results, effects)

    print("\n" + "=" * 60)
    print("  All Lara experiments complete.")
    print("=" * 60)
