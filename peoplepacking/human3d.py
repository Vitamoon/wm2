"""
3D Articulated Human Model for Packing Experiments.

Skeleton-based rig using capsule primitives for each body part.
Supports arbitrary posing via joint angles.

Coordinate system:
  X = left/right (+ = right)
  Y = front/back (+ = forward)
  Z = up/down (+ = up)

Rotation convention (in parent's frame):
  rx = pitch (forward/backward tilt) - positive = forward
  ry = yaw (left/right swing) - positive = swing toward +X
  rz = roll (twist around limb axis)

Applied as: R = Rz @ Ry @ Rx (intrinsic ZYX)
"""

import numpy as np
import trimesh
from trimesh.creation import capsule, box as tmbox, icosphere


def rotation_matrix_x(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rotation_matrix_y(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rotation_matrix_z(angle_deg):
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def make_capsule(radius, height):
    return capsule(height=height, radius=radius, count=[8, 8])

def make_ellipsoid(rx, ry, rz):
    sphere = icosphere(subdivisions=2, radius=1.0)
    verts = sphere.vertices.copy()
    verts[:, 0] *= rx
    verts[:, 1] *= ry
    verts[:, 2] *= rz
    return trimesh.Trimesh(vertices=verts, faces=sphere.faces)


class HumanRig:
    """
    Articulated 3D human model.

    Rest pose = arms at sides, legs straight down (anatomical position).
    Capsules are created along Z-axis by trimesh.

    For arms/legs (which hang down in -Z):
      - ry rotation = abduction (swing out to the side, T-pose direction)
      - rx rotation = flexion (swing forward, raising limbs in front)
    """

    # Anthropometric proportions (avg adult, ~1.75m)
    # Upper arm ~0.25m, forearm ~0.22m, hand ~0.08m
    # Thigh ~0.40m, shin ~0.38m, foot ~0.18m
    BODY_PARTS = {
        "pelvis":       {"type": "ellipsoid", "rx": 0.16, "ry": 0.12, "rz": 0.10},
        "spine":        {"type": "capsule", "radius": 0.13, "height": 0.18},
        "chest":        {"type": "ellipsoid", "rx": 0.20, "ry": 0.13, "rz": 0.16},
        "neck":         {"type": "capsule", "radius": 0.05, "height": 0.10},
        "head":         {"type": "ellipsoid", "rx": 0.09, "ry": 0.10, "rz": 0.10},
        "l_upper_arm":  {"type": "capsule", "radius": 0.04, "height": 0.20},
        "l_forearm":    {"type": "capsule", "radius": 0.033, "height": 0.22},
        "l_hand":       {"type": "ellipsoid", "rx": 0.04, "ry": 0.03, "rz": 0.06},
        "r_upper_arm":  {"type": "capsule", "radius": 0.04, "height": 0.20},
        "r_forearm":    {"type": "capsule", "radius": 0.033, "height": 0.22},
        "r_hand":       {"type": "ellipsoid", "rx": 0.04, "ry": 0.03, "rz": 0.06},
        "l_thigh":      {"type": "capsule", "radius": 0.07, "height": 0.36},
        "l_shin":       {"type": "capsule", "radius": 0.05, "height": 0.38},
        "l_foot":       {"type": "capsule", "radius": 0.035, "height": 0.18},
        "r_thigh":      {"type": "capsule", "radius": 0.07, "height": 0.36},
        "r_shin":       {"type": "capsule", "radius": 0.05, "height": 0.38},
        "r_foot":       {"type": "capsule", "radius": 0.035, "height": 0.18},
    }

    # Local mesh offsets to shift capsules along bone direction,
    # closing visual gaps between limb segments and their children.
    # Applied in the part's local frame before the global transform.
    MESH_OFFSETS = {
        "l_upper_arm":  [0, 0, -0.05],
        "r_upper_arm":  [0, 0, -0.05],
        "l_forearm":    [0, 0, -0.06],
        "r_forearm":    [0, 0, -0.06],
        "l_thigh":      [0, 0, -0.05],
        "r_thigh":      [0, 0, -0.05],
        "l_shin":       [0, 0, -0.06],
        "r_shin":       [0, 0, -0.06],
    }

    # Offsets in rest pose (arms at sides, standing straight)
    # Lengths here = bone lengths between joints (not capsule visual size)
    SKELETON = {
        "pelvis":       {"parent": None,          "offset": [0, 0, 0.93]},
        "spine":        {"parent": "pelvis",      "offset": [0, 0, 0.10]},
        "chest":        {"parent": "spine",       "offset": [0, 0, 0.20]},
        "neck":         {"parent": "chest",       "offset": [0, 0, 0.16]},
        "head":         {"parent": "neck",        "offset": [0, 0, 0.12]},
        # Arms: shoulder to elbow ~0.25m, elbow to wrist ~0.22m
        "l_upper_arm":  {"parent": "chest",       "offset": [-0.22, 0, 0.06]},
        "l_forearm":    {"parent": "l_upper_arm", "offset": [0, 0, -0.25]},
        "l_hand":       {"parent": "l_forearm",   "offset": [0, 0, -0.22]},
        "r_upper_arm":  {"parent": "chest",       "offset": [0.22, 0, 0.06]},
        "r_forearm":    {"parent": "r_upper_arm", "offset": [0, 0, -0.25]},
        "r_hand":       {"parent": "r_forearm",   "offset": [0, 0, -0.22]},
        # Legs: hip to knee ~0.40m, knee to ankle ~0.38m
        "l_thigh":      {"parent": "pelvis",      "offset": [-0.10, 0, -0.06]},
        "l_shin":       {"parent": "l_thigh",     "offset": [0, 0, -0.40]},
        "l_foot":       {"parent": "l_shin",      "offset": [0, 0.08, -0.38]},
        "r_thigh":      {"parent": "pelvis",      "offset": [0.10, 0, -0.06]},
        "r_shin":       {"parent": "r_thigh",     "offset": [0, 0, -0.40]},
        "r_foot":       {"parent": "r_shin",      "offset": [0, 0.08, -0.38]},
    }

    def __init__(self, scale=1.0):
        self.scale = scale

    def _make_part_mesh(self, part_name):
        info = self.BODY_PARTS[part_name]
        if info["type"] == "capsule":
            return make_capsule(info["radius"] * self.scale,
                                info["height"] * self.scale)
        elif info["type"] == "ellipsoid":
            return make_ellipsoid(info["rx"] * self.scale,
                                  info["ry"] * self.scale,
                                  info["rz"] * self.scale)

    def build(self, pose=None):
        """
        Build the full human mesh in a given pose.
        pose: dict of {joint_name: (rx, ry, rz)} rotation angles in degrees.
        Returns: trimesh.Trimesh
        """
        if pose is None:
            pose = {}

        transforms = {}
        meshes = []

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

        for part_name in self.SKELETON:
            T = compute_transform(part_name)
            mesh = self._make_part_mesh(part_name)
            # Shift limb capsules along bone to close gaps between segments
            if part_name in self.MESH_OFFSETS:
                offset = np.array(self.MESH_OFFSETS[part_name]) * self.scale
                mesh.vertices += offset
            mesh.apply_transform(T)
            meshes.append(mesh)

        return trimesh.util.concatenate(meshes)


# ============================================================
# POSE LIBRARY
# ============================================================
# Convention: (rx, ry, rz)
#   rx = pitch (forward +, backward -)
#   ry = yaw   (for arms: outward toward T-pose; for left arm +ry = toward -X)
#   rz = roll  (twist)
#
# Arms hang in -Z at rest. To T-pose: rotate ry.
#   Left arm:  ry = +90  -> arm extends to -X (left)
#   Right arm: ry = -90  -> arm extends to +X (right)
# To raise arms forward: rx = +90
# Legs same: ry = spread, rx = forward kick

POSES_3D = {
    # --- BASIC POSES ---
    "Standing (arms at sides)": {
        # Rest pose, no changes
    },

    "T-Pose": {
        "l_upper_arm": (0, 90, 0),
        "l_forearm":   (0, 0, 0),
        "r_upper_arm": (0, -90, 0),
        "r_forearm":   (0, 0, 0),
    },

    "Fetal Position": {
        # Lying on side (roll pelvis 90 around Y to lie on side)
        "pelvis":      (0, 0, 90),       # Roll to lie on right side
        "spine":       (40, 0, 0),        # Curl forward
        "chest":       (30, 0, 0),
        "neck":        (30, 0, 0),
        "head":        (20, 0, 0),
        "l_thigh":     (120, 10, 0),      # Legs pulled up to chest
        "r_thigh":     (120, -10, 0),
        "l_shin":      (-120, 0, 0),      # Knees fully bent
        "r_shin":      (-120, 0, 0),
        "l_upper_arm": (80, 30, 0),       # Arms curled in
        "r_upper_arm": (80, -30, 0),
        "l_forearm":   (-100, 0, 0),
        "r_forearm":   (-100, 0, 0),
    },

    "Star / Spread Eagle": {
        "l_upper_arm": (0, 135, 0),       # Arms up-diagonal
        "r_upper_arm": (0, -135, 0),
        "l_thigh":     (0, 25, 0),        # Legs spread
        "r_thigh":     (0, -25, 0),
    },

    "Pike (folded)": {
        # Standing but folded forward at the waist, touching toes
        "spine":       (50, 0, 0),
        "chest":       (50, 0, 0),
        "neck":        (20, 0, 0),
        "head":        (10, 0, 0),
        "l_upper_arm": (90, 0, 0),        # Arms reaching down
        "r_upper_arm": (90, 0, 0),
        "l_forearm":   (0, 0, 0),
        "r_forearm":   (0, 0, 0),
    },

    "Superman (arms up)": {
        "l_upper_arm": (0, 180, 0),       # Arms straight up
        "r_upper_arm": (0, -180, 0),
    },

    # --- FAMOUS POSES ---

    "Da Vinci Vitruvian": {
        "l_upper_arm": (0, 90, 0),        # Arms horizontal
        "r_upper_arm": (0, -90, 0),
        "l_thigh":     (0, 22, 0),        # Legs slightly spread
        "r_thigh":     (0, -22, 0),
    },

    "Usain Bolt Lightning": {
        "r_upper_arm": (0, -150, 0),      # Right arm pointing up-right
        "r_forearm":   (-20, 0, 0),
        "l_upper_arm": (30, 40, 0),       # Left arm back
        "l_forearm":   (-60, 0, 0),
        "l_thigh":     (30, 0, 0),        # Mid-stride
        "r_thigh":     (-20, 0, 0),
        "l_shin":      (-20, 0, 0),
        "r_shin":      (-50, 0, 0),
        "spine":       (0, -10, 0),       # Slight lean
    },

    "John Travolta Disco": {
        "r_upper_arm": (0, -150, 0),      # Right arm up (Saturday Night Fever)
        "r_forearm":   (0, 0, 0),
        "l_upper_arm": (40, 40, 0),       # Left arm on hip area
        "l_forearm":   (-90, 0, 0),
        "l_thigh":     (0, 15, 0),        # Slight stance
        "r_thigh":     (0, -5, 0),
    },

    "Titanic Jack & Rose": {
        "l_upper_arm": (0, 95, 0),        # Arms spread wide
        "r_upper_arm": (0, -95, 0),
        "spine":       (15, 0, 0),        # Leaning forward
        "chest":       (10, 0, 0),
    },

    "Heisman Trophy": {
        "r_upper_arm": (0, -90, 0),       # Stiff arm extended
        "r_forearm":   (0, 0, 0),
        "l_upper_arm": (40, 40, 0),       # Ball tucked
        "l_forearm":   (-110, 0, 0),
        "l_thigh":     (90, 0, 0),        # Left knee up high
        "l_shin":      (-90, 0, 0),
        "r_thigh":     (-15, 0, 0),       # Back leg
        "r_shin":      (-20, 0, 0),
        "spine":       (-5, 0, 0),
    },

    "Thinking Man (Rodin)": {
        # Seated, hunched, chin on fist
        "l_thigh":     (90, 10, 0),       # Seated
        "r_thigh":     (90, -10, 0),
        "l_shin":      (-90, 0, 0),
        "r_shin":      (-90, 0, 0),
        "spine":       (35, 0, 0),        # Hunched
        "chest":       (25, 0, 0),
        "neck":        (20, 0, 0),
        "head":        (15, 0, 0),
        "r_upper_arm": (60, -20, 0),      # Right arm to chin
        "r_forearm":   (-110, 0, 0),
        "l_upper_arm": (20, 20, 0),       # Left arm on knee
        "l_forearm":   (-40, 0, 0),
    },

    "Naruto Run": {
        "spine":       (30, 0, 0),        # Leaning forward
        "chest":       (20, 0, 0),
        "l_upper_arm": (-50, 0, 0),       # Arms swept back
        "r_upper_arm": (-50, 0, 0),
        "l_forearm":   (0, 0, 0),
        "r_forearm":   (0, 0, 0),
        "l_thigh":     (50, 0, 0),        # Mid-run
        "r_thigh":     (-20, 0, 0),
        "l_shin":      (-30, 0, 0),
        "r_shin":      (-50, 0, 0),
    },

    "Dab": {
        "l_upper_arm": (30, 120, 0),      # Left arm across face (up)
        "l_forearm":   (-40, 0, 0),
        "r_upper_arm": (0, -110, 0),      # Right arm extended out
        "r_forearm":   (0, 0, 0),
        "spine":       (0, 10, 0),
        "neck":        (20, 20, 0),
        "head":        (20, 10, 0),
    },

    "Planking": {
        # Lying face down, rigid
        "pelvis":      (90, 0, 0),        # Tip forward to horizontal
        "l_upper_arm": (0, 0, 0),
        "r_upper_arm": (0, 0, 0),
    },

    "Downward Dog": {
        # Inverted V yoga pose
        "spine":       (60, 0, 0),
        "chest":       (40, 0, 0),
        "neck":        (30, 0, 0),
        "head":        (20, 0, 0),
        "l_upper_arm": (0, 180, 0),       # Arms straight up (extending toward floor)
        "r_upper_arm": (0, -180, 0),
        "l_thigh":     (10, 5, 0),
        "r_thigh":     (10, -5, 0),
    },

    # --- EXTRA MEME POSES ---

    "A-Pose (Game Dev)": {
        # Industry standard rest pose - arms 45 degrees
        "l_upper_arm": (0, 45, 0),
        "r_upper_arm": (0, -45, 0),
    },

    "Crucifixion": {
        # Arms out, slight droop
        "l_upper_arm": (0, 80, 0),
        "r_upper_arm": (0, -80, 0),
        "l_forearm":   (10, 0, 0),
        "r_forearm":   (10, 0, 0),
        "head":        (30, 0, 0),        # Head drooped
        "l_thigh":     (0, 0, 0),
        "r_thigh":     (0, 0, 0),
        "l_shin":      (-20, 0, 0),       # Slightly bent
        "r_shin":      (-20, 0, 0),
    },

    "Squat (Slav)": {
        "l_thigh":     (100, 15, 0),
        "r_thigh":     (100, -15, 0),
        "l_shin":      (-130, 0, 0),
        "r_shin":      (-130, 0, 0),
        "spine":       (10, 0, 0),
        "l_upper_arm": (30, 20, 0),
        "r_upper_arm": (30, -20, 0),
        "l_forearm":   (-60, 0, 0),
        "r_forearm":   (-60, 0, 0),
    },

    "Coffin Dance": {
        # Lying flat on back
        "pelvis":      (-90, 0, 0),       # Tipped backward to horizontal
        "l_upper_arm": (0, 0, 0),
        "r_upper_arm": (0, 0, 0),
    },
}


def build_posed_human(pose_name, scale=1.0):
    """Build a 3D human mesh in the named pose."""
    rig = HumanRig(scale=scale)
    pose = POSES_3D.get(pose_name, {})
    return rig.build(pose=pose)


def get_bounding_box(mesh):
    """Get axis-aligned bounding box dimensions."""
    dims = mesh.bounds[1] - mesh.bounds[0]
    return dims

def get_bb_volume(mesh):
    dims = get_bounding_box(mesh)
    return dims[0] * dims[1] * dims[2]

def packing_efficiency_3d(mesh):
    """Ratio of convex hull volume to bounding box volume."""
    bb_vol = get_bb_volume(mesh)
    if bb_vol == 0:
        return 0
    return mesh.convex_hull.volume / bb_vol
