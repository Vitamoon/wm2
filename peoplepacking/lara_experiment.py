"""
THE TRIANGLE THEOREM: On the Packing Implications of
PS1-Era Polygonal Thorax Geometry

Compares packing density of a PS1-era low-poly Lara Croft model
(angular geometry with triangular chest protrusions) vs a smooth
high-poly variant and the standard capsule human.

Key question: Do the triangle tits actually matter for packing?

The PS1 Lara model is constructed from angular primitives (boxes,
cones, low-poly shapes) to faithfully recreate the ~230-polygon
aesthetic of the 1996 original, complete with per-part coloring.
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
    render_mesh, trimesh_to_pyvista,
    pack_3d_grid, pack_3d_rotation_search, apply_rotation_to_mesh,
    OUTPUT_DIR, SKIN_TONE,
)

import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# PS1-Era Color Palette (Tomb Raider 1, 1996)
# ============================================================
LARA_COLORS = {
    "skin":     "#D4956A",
    "top":      "#1F8A8A",   # turquoise/teal tank top
    "shorts":   "#6B3A1F",   # brown shorts
    "boots":    "#3D2B1F",   # dark brown boots
    "hair":     "#2E1A0E",   # dark brown hair
    "braid":    "#3A2215",   # slightly lighter braid
    "holster":  "#2A2A2A",   # gun holsters (dark grey)
    "belt":     "#4A3728",   # belt
}

# Per-body-part colors for PS1 Lara outfit
PART_COLORS = {
    "pelvis":       LARA_COLORS["shorts"],
    "spine":        LARA_COLORS["top"],
    "chest":        LARA_COLORS["top"],
    "neck":         LARA_COLORS["skin"],
    "head":         LARA_COLORS["skin"],
    "l_upper_arm":  LARA_COLORS["skin"],
    "l_forearm":    LARA_COLORS["skin"],
    "l_hand":       LARA_COLORS["skin"],
    "r_upper_arm":  LARA_COLORS["skin"],
    "r_forearm":    LARA_COLORS["skin"],
    "r_hand":       LARA_COLORS["skin"],
    "l_thigh":      LARA_COLORS["skin"],
    "r_thigh":      LARA_COLORS["skin"],
    "l_shin":       LARA_COLORS["boots"],
    "r_shin":       LARA_COLORS["boots"],
    "l_foot":       LARA_COLORS["boots"],
    "r_foot":       LARA_COLORS["boots"],
    "ponytail":     LARA_COLORS["braid"],
    "hair_cap":     LARA_COLORS["hair"],
    "l_holster":    LARA_COLORS["holster"],
    "r_holster":    LARA_COLORS["holster"],
    "belt":         LARA_COLORS["belt"],
    "l_triangle":   LARA_COLORS["top"],
    "r_triangle":   LARA_COLORS["top"],
}


def make_box(sx, sy, sz):
    """Create a box mesh centered at origin."""
    return trimesh.creation.box(extents=[sx, sy, sz])


def make_cone(radius, height, sections=4):
    """Create a cone mesh (low-poly for PS1 aesthetic)."""
    return trimesh.creation.cone(radius=radius, height=height, sections=sections)


def make_low_poly_sphere(radius, subdivisions=1):
    """Low-poly sphere for PS1 aesthetic."""
    s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return s


# ============================================================
# PS1 Lara Croft Rig - Angular Geometry
# ============================================================

class PS1LaraRig:
    """
    PS1-era Low-Poly Archaeologist (~230 polygon aesthetic).

    Uses BOXES for limbs/torso instead of capsules, creating the
    angular, faceted look of the 1996 original. Each body part
    has a distinct color matching Lara's iconic outfit.

    Key features:
    - Angular box-based geometry (not smooth capsules)
    - Triangular (pyramidal) chest protrusions
    - Ponytail braid
    - Turquoise tank top, brown shorts, dark boots
    """

    # Body part dimensions: (sx, sy, sz) for boxes
    # Sizes are slightly oversized to ensure overlap at joints (no gaps)
    BODY_PARTS = {
        "pelvis":       {"type": "box", "sx": 0.32, "sy": 0.18, "sz": 0.16},
        "spine":        {"type": "box", "sx": 0.26, "sy": 0.18, "sz": 0.22},
        "chest":        {"type": "box", "sx": 0.34, "sy": 0.20, "sz": 0.26},
        "neck":         {"type": "box", "sx": 0.09, "sy": 0.09, "sz": 0.14},
        "head":         {"type": "low_sphere", "radius": 0.095},
        "l_upper_arm":  {"type": "box", "sx": 0.07, "sy": 0.07, "sz": 0.27},
        "l_forearm":    {"type": "box", "sx": 0.06, "sy": 0.06, "sz": 0.24},
        "l_hand":       {"type": "box", "sx": 0.06, "sy": 0.04, "sz": 0.08},
        "r_upper_arm":  {"type": "box", "sx": 0.07, "sy": 0.07, "sz": 0.27},
        "r_forearm":    {"type": "box", "sx": 0.06, "sy": 0.06, "sz": 0.24},
        "r_hand":       {"type": "box", "sx": 0.06, "sy": 0.04, "sz": 0.08},
        "l_thigh":      {"type": "box", "sx": 0.14, "sy": 0.14, "sz": 0.44},
        "l_shin":       {"type": "box", "sx": 0.11, "sy": 0.11, "sz": 0.42},
        "l_foot":       {"type": "box", "sx": 0.09, "sy": 0.18, "sz": 0.07},
        "r_thigh":      {"type": "box", "sx": 0.14, "sy": 0.14, "sz": 0.44},
        "r_shin":       {"type": "box", "sx": 0.11, "sy": 0.11, "sz": 0.42},
        "r_foot":       {"type": "box", "sx": 0.09, "sy": 0.18, "sz": 0.07},
        "ponytail":     {"type": "box", "sx": 0.04, "sy": 0.04, "sz": 0.28},
    }

    # Triangular breast geometry
    TRIANGLE_RADIUS = 0.07
    TRIANGLE_HEIGHT = 0.12
    TRIANGLE_SPREAD = 0.09

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
        "ponytail":     [0, -0.04, -0.14],
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
        self.scale = scale
        self.triangles = triangles

    def _make_part_mesh(self, part_name):
        info = self.BODY_PARTS[part_name]
        s = self.scale
        if info["type"] == "box":
            return make_box(info["sx"] * s, info["sy"] * s, info["sz"] * s)
        elif info["type"] == "low_sphere":
            return make_low_poly_sphere(info["radius"] * s, subdivisions=1)
        elif info["type"] == "capsule":
            return make_capsule(info["radius"] * s, info["height"] * s)
        elif info["type"] == "ellipsoid":
            return make_ellipsoid(info["rx"] * s, info["ry"] * s, info["rz"] * s)

    def _compute_transforms(self, pose):
        """Compute all bone transforms for a pose."""
        if pose is None:
            pose = {}
        transforms = {}

        def compute(part_name):
            if part_name in transforms:
                return transforms[part_name]
            skel = self.SKELETON[part_name]
            parent = skel["parent"]
            offset = np.array(skel["offset"]) * self.scale
            if parent is None:
                parent_transform = np.eye(4)
            else:
                parent_transform = compute(parent)
            local_rot = np.eye(3)
            if part_name in pose:
                rx, ry, rz = pose[part_name]
                local_rot = rotation_matrix_z(rz) @ rotation_matrix_y(ry) @ rotation_matrix_x(rx)
            T = np.eye(4)
            T[:3, :3] = parent_transform[:3, :3] @ local_rot
            T[:3, 3] = parent_transform[:3, 3] + parent_transform[:3, :3] @ offset
            transforms[part_name] = T
            return T

        for part_name in self.SKELETON:
            compute(part_name)
        return transforms

    def build(self, pose=None):
        """Build full Lara mesh (single color, for packing calculations)."""
        parts = self.build_parts(pose)
        meshes = [m for m, _ in parts]
        return trimesh.util.concatenate(meshes)

    def build_parts(self, pose=None):
        """
        Build Lara as list of (mesh, color) tuples for colored rendering.
        """
        transforms = self._compute_transforms(pose)
        parts = []

        # Main body parts
        for part_name in self.SKELETON:
            T = transforms[part_name]
            mesh = self._make_part_mesh(part_name)
            if part_name in self.MESH_OFFSETS:
                offset = np.array(self.MESH_OFFSETS[part_name]) * self.scale
                mesh.vertices += offset
            mesh.apply_transform(T)
            color = PART_COLORS.get(part_name, LARA_COLORS["skin"])
            parts.append((mesh, color))

        # Hair cap (dark sphere on head, slightly larger)
        head_T = transforms["head"]
        hair = make_low_poly_sphere(0.098 * self.scale, subdivisions=1)
        # Clip to top half for a cap effect
        hair.vertices[:, 2] = np.maximum(hair.vertices[:, 2], -0.01 * self.scale)
        hair_offset = np.array([0, -0.01, 0.02]) * self.scale
        hair.vertices += hair_offset
        hair.apply_transform(head_T)
        parts.append((hair, LARA_COLORS["hair"]))

        # Holsters (small boxes on thighs)
        for side, thigh_name in [(-1, "l_thigh"), (1, "r_thigh")]:
            holster = make_box(0.06 * self.scale, 0.07 * self.scale, 0.10 * self.scale)
            T = transforms[thigh_name]
            holster_offset = np.array([side * 0.08, 0.04, -0.08]) * self.scale
            holster.vertices += holster_offset
            holster.apply_transform(T)
            parts.append((holster, LARA_COLORS["holster"]))

        # Belt
        belt = make_box(0.34 * self.scale, 0.19 * self.scale, 0.03 * self.scale)
        pelvis_T = transforms["pelvis"]
        belt_offset = np.array([0, 0, 0.05]) * self.scale
        belt.vertices += belt_offset
        belt.apply_transform(pelvis_T)
        parts.append((belt, LARA_COLORS["belt"]))

        # Triangle breasts (the star of the show)
        if self.triangles:
            chest_T = transforms["chest"]
            for side in [-1, 1]:
                cone = make_cone(
                    radius=self.TRIANGLE_RADIUS * self.scale,
                    height=self.TRIANGLE_HEIGHT * self.scale,
                    sections=4,  # 4 sides = pyramid = maximum PS1 energy
                )
                # Rotate to point forward (+Y)
                R = rotation_matrix_x(90)
                T_rot = np.eye(4)
                T_rot[:3, :3] = R
                cone.apply_transform(T_rot)
                local_offset = np.array([
                    side * self.TRIANGLE_SPREAD * self.scale,
                    0.10 * self.scale,
                    0.02 * self.scale,
                ])
                cone.vertices += local_offset
                cone.apply_transform(chest_T)
                label = "l_triangle" if side == -1 else "r_triangle"
                parts.append((cone, PART_COLORS[label]))

        return parts


class SmoothLaraRig(PS1LaraRig):
    """Modern Lara: same proportions but smooth geometry, no triangles."""

    def __init__(self, scale=1.0):
        super().__init__(scale=scale, triangles=False)

    def build_parts(self, pose=None):
        """Build with smooth hemispheres instead of triangles."""
        parts = PS1LaraRig.build_parts(self, pose)

        transforms = self._compute_transforms(pose)
        chest_T = transforms["chest"]

        for side in [-1, 1]:
            sphere = make_ellipsoid(
                0.06 * self.scale,
                0.06 * self.scale,
                0.05 * self.scale,
            )
            local_offset = np.array([
                side * self.TRIANGLE_SPREAD * self.scale,
                0.10 * self.scale,
                0.02 * self.scale,
            ])
            sphere.vertices += local_offset
            sphere.apply_transform(chest_T)
            parts.append((sphere, LARA_COLORS["top"]))

        return parts


def build_lara(pose_name, triangles=True, scale=1.0):
    """Build PS1 Lara (triangles=True) or Modern Lara (triangles=False)."""
    if triangles:
        rig = PS1LaraRig(scale=scale, triangles=True)
    else:
        rig = SmoothLaraRig(scale=scale)
    pose = LARA_POSES.get(pose_name, POSES_3D.get(pose_name, {}))
    return rig.build(pose=pose)


def build_lara_parts(pose_name, triangles=True, scale=1.0):
    """Build PS1 Lara as colored parts list."""
    if triangles:
        rig = PS1LaraRig(scale=scale, triangles=True)
    else:
        rig = SmoothLaraRig(scale=scale)
    pose = LARA_POSES.get(pose_name, POSES_3D.get(pose_name, {}))
    return rig.build_parts(pose=pose)


# ============================================================
# Colored Rendering
# ============================================================

def render_lara(parts, filename, title="", show_bb=True,
                window_size=(900, 900), camera_position=None):
    """Render a multi-colored Lara model from a front 3/4 view."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('#F0EDE8')

    combined = trimesh.util.concatenate([m for m, _ in parts])

    for mesh, color in parts:
        pv_mesh = trimesh_to_pyvista(mesh)
        pl.add_mesh(pv_mesh, color=color, opacity=1.0, smooth_shading=False,
                    show_edges=False, specular=0.3, ambient=0.25)

    if show_bb:
        bounds = combined.bounds
        bb = pv.Box(bounds=[bounds[0][0], bounds[1][0],
                            bounds[0][1], bounds[1][1],
                            bounds[0][2], bounds[1][2]])
        pl.add_mesh(bb, color='#CC4444', style='wireframe', line_width=2.0, opacity=0.4)

    if title:
        pl.add_title(title, font_size=11, color='#333333')

    if camera_position:
        pl.camera_position = camera_position
    else:
        # Side-front 3/4 view to show the triangle profile
        # Z is up, Y is forward in our coordinate system
        center = combined.centroid
        # Camera from right-side with slight front angle
        # This shows the iconic triangle silhouette
        cam_dist = 4.0
        cam_x = center[0] + cam_dist * 0.55  # to the right
        cam_y = center[1] + cam_dist * 0.65  # in front (shows triangle profile)
        cam_z = center[2] + cam_dist * 0.2   # slightly above
        pl.camera_position = [
            (cam_x, cam_y, cam_z),         # camera position
            tuple(center),                  # focal point
            (0, 0, 1),                      # view up = Z
        ]

    pl.add_light(pv.Light(position=(3, 3, 5), intensity=0.8))
    pl.add_light(pv.Light(position=(-2, -2, 3), intensity=0.3))

    pl.screenshot(filename)
    pl.close()


def render_lara_packing_scene(packed_parts_list, container_dims, filename,
                               title="", window_size=(1400, 900), max_render=60):
    """
    Render a packing scene with colored Lara models.
    packed_parts_list: list of (parts_list, offset) where parts_list is [(mesh, color), ...]
    """
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('#F0EDE8')

    n = min(len(packed_parts_list), max_render)

    for i, (parts, offset) in enumerate(packed_parts_list[:n]):
        # Slight tint variation per instance for visual distinction
        tint = 0.92 + 0.08 * (i % 5) / 4.0
        for mesh, color in parts:
            shifted = mesh.copy()
            shifted.vertices += offset
            pv_mesh = trimesh_to_pyvista(shifted)
            # Apply slight brightness variation
            r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
            r = min(255, int(r * tint))
            g = min(255, int(g * tint))
            b = min(255, int(b * tint))
            tinted = f"#{r:02x}{g:02x}{b:02x}"
            pl.add_mesh(pv_mesh, color=tinted, opacity=0.88, smooth_shading=False,
                        show_edges=False, specular=0.25, ambient=0.2)

    # Container wireframe
    Lx, Ly, Lz = container_dims
    container = pv.Box(bounds=[0, Lx, 0, Ly, 0, Lz])
    pl.add_mesh(container, color='#333333', style='wireframe', line_width=3.0, opacity=0.9)

    if title:
        pl.add_title(title, font_size=12, color='#333333')

    # 3/4 isometric view (Z is up)
    pl.camera_position = 'xz'
    pl.camera.azimuth = 35
    pl.camera.elevation = 25
    pl.camera.zoom(0.80)
    pl.add_light(pv.Light(position=(Lx*2, Ly*2, Lz*3), intensity=0.7))
    pl.add_light(pv.Light(position=(-Lx, -Ly, Lz*2), intensity=0.3))

    pl.screenshot(filename)
    pl.close()


# ============================================================
# Lara-specific poses
# ============================================================

LARA_POSES = {
    "Standing (arms at sides)": POSES_3D["Standing (arms at sides)"],
    "T-Pose": POSES_3D["T-Pose"],
    "Dual Pistols": {
        "l_upper_arm": (70, 30, 0),
        "r_upper_arm": (70, -30, 0),
        "l_forearm":   (-20, 0, 0),
        "r_forearm":   (-20, 0, 0),
    },
    "Handstand": {
        "pelvis":      (180, 0, 0),
    },
    "Crouch": {
        "l_thigh":     (90, 10, 0),
        "r_thigh":     (90, -10, 0),
        "l_shin":      (-80, 0, 0),
        "r_shin":      (-80, 0, 0),
        "spine":       (20, 0, 0),
        "chest":       (10, 0, 0),
    },
    "Swan Dive": {
        "pelvis":      (70, 0, 0),
        "l_upper_arm": (-40, 60, 0),
        "r_upper_arm": (-40, -60, 0),
    },
    "Planking": POSES_3D["Planking"],
    "Coffin Dance": POSES_3D["Coffin Dance"],
    "Fetal Position": POSES_3D["Fetal Position"],
}


# ============================================================
# Venues
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
    "Tomb Corridor":      (12.0, 1.2, 2.5),
    "Sarcophagus":        (2.1, 0.7, 0.8),
}


# ============================================================
# Experiments
# ============================================================

def experiment_lara_gallery():
    """Render PS1 Lara in key poses with proper coloring."""
    print("=" * 60)
    print("LARA EXPERIMENT 1: PS1-Era Pose Gallery (Angular Model)")
    print("=" * 60)

    for name in LARA_POSES:
        # Triangle version (PS1)
        parts_tri = build_lara_parts(name, triangles=True)
        mesh_tri = trimesh.util.concatenate([m for m, _ in parts_tri])

        # Smooth version (Modern)
        parts_smooth = build_lara_parts(name, triangles=False)
        mesh_smooth = trimesh.util.concatenate([m for m, _ in parts_smooth])

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

        # Render PS1 version with colors
        render_lara(parts_tri, f"{OUTPUT_DIR}/lara_ps1_{safe}.png",
                    title=f"PS1 Lara: {name}")

        # Render smooth version with colors
        render_lara(parts_smooth, f"{OUTPUT_DIR}/lara_modern_{safe}.png",
                    title=f"Modern Lara: {name}")


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

    # Triangle impact summary
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
        print(f"\n  Found {len(triangle_effects)} venue/pose combos where triangles matter:\n")
        for effect in sorted(triangle_effects, key=lambda e: abs(e["delta"]), reverse=True):
            direction = "BETTER" if effect["delta"] > 0 else "WORSE"
            print(f"  {effect['venue']:<22} {effect['pose']:<28} "
                  f"Tri={effect['tri']:>4}  Smooth={effect['smooth']:>4}  "
                  f"Delta={effect['delta']:+d} ({direction})")
    else:
        print("\n  NO DIFFERENCES - triangles are purely cosmetic for AABB packing!")

    with open(f"{OUTPUT_DIR}/lara_packing.json", "w") as f:
        json.dump(results, f, indent=2)

    return results, triangle_effects


def experiment_triangle_rotation_analysis():
    """Rotation-angle analysis of triangle vs smooth BB differences."""
    print("\n" + "=" * 60)
    print("LARA EXPERIMENT 3: Rotation-Angle BB Analysis")
    print("=" * 60)

    from experiment3d import rotated_bb

    pose_name = "Standing (arms at sides)"
    mesh_tri = build_lara(pose_name, triangles=True)
    mesh_smooth = build_lara(pose_name, triangles=False)

    angles = np.linspace(0, 180, 36, endpoint=False)

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

    print(f"  Max BB vol diff: {max_diff:.6f} m3 at rx={max_diff_angle[0]:.0f}, ry={max_diff_angle[1]:.0f}")
    print(f"  (Triangle {'larger' if max_diff > 0 else 'smaller'} than smooth)")

    better_angles = [d for d in bb_diffs if d["diff"] < -0.001]
    if better_angles:
        print(f"\n  Found {len(better_angles)} rotation angles where triangles produce SMALLER BB!")
        best = min(better_angles, key=lambda d: d["diff"])
        print(f"  Best: rx={best['rx']:.0f}, ry={best['ry']:.0f}: "
              f"tri={best['vol_tri']:.4f} vs smooth={best['vol_smooth']:.4f}")

    # Heatmap
    n = len(angles)
    diff_matrix = np.zeros((n, n))
    for d in bb_diffs:
        ix = int(d["rx"] / 5)
        iy = int(d["ry"] / 5)
        if ix < n and iy < n:
            diff_matrix[ix, iy] = d["diff"]

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(diff_matrix.min()), abs(diff_matrix.max()))
    if vmax == 0:
        vmax = 0.001
    im = ax.imshow(diff_matrix * 1000, cmap='RdBu_r', aspect='auto',
                   extent=[0, 180, 180, 0], vmin=-vmax*1000, vmax=vmax*1000)
    ax.set_xlabel("Ry (yaw) degrees", fontsize=12)
    ax.set_ylabel("Rx (pitch) degrees", fontsize=12)
    ax.set_title("BB Volume Difference: Triangle - Smooth (mL)\n"
                 "Blue = triangles produce smaller BB",
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Volume difference (mL)", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lara_rotation_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved lara_rotation_analysis.png")

    return bb_diffs


def experiment_triangle_visualization():
    """Render comparison scenes with colored Lara models."""
    print("\n" + "=" * 60)
    print("LARA EXPERIMENT 4: Visual Comparison (Colored)")
    print("=" * 60)

    from PIL import Image

    key_poses = ["Standing (arms at sides)", "Dual Pistols", "Crouch", "Swan Dive"]

    # Make a 2x4 grid: top row PS1, bottom row Modern
    cols = 4
    cell_w, cell_h = 600, 600
    grid = Image.new('RGB', (cols * cell_w, 2 * cell_h), (240, 237, 232))

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
    pose_name = "Standing (arms at sides)"
    mesh = build_lara(pose_name, triangles=True)
    count_basic, offsets_basic, _ = pack_3d_grid(mesh, dims)
    count_rot, offsets_rot, _, R_rot = pack_3d_rotation_search(mesh, dims, angle_steps=12)

    if count_rot > count_basic:
        count, offsets = count_rot, offsets_rot
        # Build colored parts for the rotated version
        parts_template = build_lara_parts(pose_name, triangles=True)
        rotated_parts = []
        for m, c in parts_template:
            mr = m.copy()
            mr.vertices = mr.vertices @ R_rot.T
            mr.vertices -= mr.vertices.min(axis=0)
            rotated_parts.append((mr, c))
        # Normalize all parts together
        all_verts = np.vstack([m.vertices for m, _ in rotated_parts])
        global_min = all_verts.min(axis=0)
        rotated_parts = [(trimesh.Trimesh(vertices=m.vertices - global_min, faces=m.faces), c)
                         for m, c in rotated_parts]
    else:
        count, offsets = count_basic, offsets_basic
        parts_template = build_lara_parts(pose_name, triangles=True)
        all_verts = np.vstack([m.vertices for m, _ in parts_template])
        global_min = all_verts.min(axis=0)
        rotated_parts = [(trimesh.Trimesh(vertices=m.vertices - global_min, faces=m.faces), c)
                         for m, c in parts_template]

    if count > 0:
        packed = [(rotated_parts, o) for o in offsets]
        render_lara_packing_scene(
            packed, dims,
            f"{OUTPUT_DIR}/lara_tomb_corridor_packing.png",
            title=f"Tomb Corridor: {count} PS1 Laras packed",
        )
        print(f"  Saved tomb corridor scene ({count} Laras)")

    # School Bus packing with colored Laras
    dims = VENUES["School Bus"]
    for pose_label, pose_name in [("Standing", "Standing (arms at sides)"),
                                   ("Fetal", "Fetal Position")]:
        mesh = build_lara(pose_name, triangles=True)
        c1, off1, _ = pack_3d_grid(mesh, dims)
        c2, off2, _, R2 = pack_3d_rotation_search(mesh, dims, angle_steps=12)

        if c2 > c1:
            count, offsets, R_use = c2, off2, R2
        else:
            count, offsets, R_use = c1, off1, np.eye(3)

        if count > 0:
            parts_template = build_lara_parts(pose_name, triangles=True)
            all_verts = np.vstack([m.vertices for m, _ in parts_template])
            if not np.allclose(R_use, np.eye(3)):
                rotated_parts = []
                for m, c in parts_template:
                    mr = m.copy()
                    mr.vertices = mr.vertices @ R_use.T
                    rotated_parts.append((mr, c))
                all_verts_r = np.vstack([m.vertices for m, _ in rotated_parts])
                global_min = all_verts_r.min(axis=0)
                rotated_parts = [(trimesh.Trimesh(vertices=m.vertices - global_min, faces=m.faces), c)
                                 for m, c in rotated_parts]
            else:
                global_min = all_verts.min(axis=0)
                rotated_parts = [(trimesh.Trimesh(vertices=m.vertices - global_min, faces=m.faces), c)
                                 for m, c in parts_template]

            packed = [(rotated_parts, o) for o in offsets]
            safe = f"School_Bus_{pose_label}"
            render_lara_packing_scene(
                packed, dims,
                f"{OUTPUT_DIR}/lara_{safe}_packing.png",
                title=f"School Bus: {count} PS1 Laras ({pose_label})",
            )
            print(f"  Saved School Bus {pose_label} ({count} Laras)")

    # Sarcophagus packing
    dims = VENUES["Sarcophagus"]
    for label, tri in [("PS1", True), ("Modern", False)]:
        mesh = build_lara("Coffin Dance", triangles=tri)
        c1, _, _ = pack_3d_grid(mesh, dims)
        c2, off2, _, R2 = pack_3d_rotation_search(mesh, dims, angle_steps=12)
        count = max(c1, c2)
        print(f"  Sarcophagus ({label}): {count} Laras in Coffin Dance pose")


def make_summary_chart(results, triangle_effects):
    """Create summary visualization of triangle impact."""
    print("\n  Creating summary chart...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: BB volume comparison
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

    for i, (tv, sv) in enumerate(zip(tri_vols, smooth_vols)):
        if tv != sv:
            diff_pct = (tv - sv) / sv * 100
            ax1.text(max(tv, sv) + 0.005, i,
                     f'{diff_pct:+.1f}%', va='center', fontsize=7, color='#555555')

    # Right: packing count comparison for Shipping Container
    ax2 = axes[1]
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
    print("  PS1-Era Angular Geometry Analysis")
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
