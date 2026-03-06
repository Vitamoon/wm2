"""
SPHERICAL HUMAN PACKING:
An Optimistic First-Order Approximation

"Consider a spherical human of uniform density..."
In the grand tradition of physics, we approximate the human body as a sphere.
This dramatically simplifies the packing problem to the well-studied
sphere packing problem (Kepler conjecture, proved by Hales 2005).

We investigate: given that humans are spheres, how many fit in various venues?
We compare random packing, FCC lattice packing, and the theoretical maximum.
"""

import numpy as np
import trimesh
import json
import os

import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl

OUTPUT_DIR = "results_spherical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

pv.OFF_SCREEN = True

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'savefig.facecolor': '#FAFAFA',
})


# ============================================================
# The Spherical Human Model
# ============================================================

# Average human: 1.75m tall, ~0.40m wide, ~0.28m deep
# Volume of average human body: ~0.066 m^3
# We want a sphere with equivalent volume:
#   V = 4/3 * pi * r^3 = 0.066 => r = 0.250 m
# Alternatively, the "bounding sphere" approach:
#   radius = half the height = 0.875m (very generous)
# We'll use multiple approximations for comedy.

HUMAN_VOLUME = 0.066  # m^3, average human body volume
HUMAN_HEIGHT = 1.75   # m

# Different spherical approximations
SPHERE_MODELS = {
    "Volume-Equivalent Sphere": {
        "radius": (3 * HUMAN_VOLUME / (4 * np.pi)) ** (1/3),
        "description": "Same volume as human body (r=0.25m). Extremely dense bowling ball human.",
        "color": "#E74C3C",
    },
    "Shoulder-Width Sphere": {
        "radius": 0.22,  # half of average shoulder width ~0.44m
        "description": "Diameter equals shoulder width (r=0.22m). A compact orb person.",
        "color": "#3498DB",
    },
    "Waist-Circumference Sphere": {
        "radius": 0.86 / (2 * np.pi),  # avg waist circ ~86cm
        "description": "Circumference equals waist (r=0.14m). A very small sphere.",
        "color": "#27AE60",
    },
    "Height-Bounding Sphere": {
        "radius": HUMAN_HEIGHT / 2,
        "description": "Diameter equals height (r=0.875m). Giant hamster ball human.",
        "color": "#F39C12",
    },
    "RMS Sphere (Compromise)": {
        "radius": np.sqrt((0.44**2 + 0.28**2 + 1.75**2) / 3) / 2,
        "description": "RMS of all body dimensions (r=0.53m). Statistically dubious.",
        "color": "#9B59B6",
    },
    "BMI-Adjusted Sphere": {
        # For BMI 25, mass ~76kg, density ~985 kg/m^3
        # V = 76/985 = 0.0771 m^3, r = (3V/4pi)^(1/3)
        "radius": (3 * (76/985) / (4 * np.pi)) ** (1/3),
        "description": "Mass/density derived (BMI 25). A meatball of precisely 76kg.",
        "color": "#E67E22",
    },
}


# ============================================================
# Sphere Packing Algorithms
# ============================================================

# Kepler conjecture: densest sphere packing = FCC = pi/(3*sqrt(2)) ~ 74.05%
KEPLER_DENSITY = np.pi / (3 * np.sqrt(2))

# Random close packing ~ 64%
RANDOM_CLOSE_DENSITY = 0.64

# Random loose packing ~ 60%
RANDOM_LOOSE_DENSITY = 0.60


def pack_spheres_simple(radius, container_dims):
    """Simple cubic packing (worst regular packing)."""
    Lx, Ly, Lz = container_dims
    d = 2 * radius
    nx = int(Lx // d)
    ny = int(Ly // d)
    nz = int(Lz // d)
    centers = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                centers.append([radius + ix * d, radius + iy * d, radius + iz * d])
    return len(centers), np.array(centers) if centers else np.empty((0, 3))


def pack_spheres_fcc(radius, container_dims):
    """Face-centered cubic packing (densest known regular packing)."""
    Lx, Ly, Lz = container_dims
    d = 2 * radius
    layer_height = d * np.sqrt(2/3)
    row_offset = d * np.sqrt(3) / 2

    centers = []
    iz = 0
    z = radius
    while z + radius <= Lz:
        iy = 0
        y_start = radius + (iz % 2) * (row_offset / 3)
        y = y_start
        while y + radius <= Ly:
            x_start = radius + ((iy + iz) % 2) * radius
            x = x_start
            while x + radius <= Lx:
                centers.append([x, y, z])
                x += d
            iy += 1
            y += row_offset
        iz += 1
        z += layer_height

    return len(centers), np.array(centers) if centers else np.empty((0, 3))


def pack_spheres_theoretical(radius, container_dims):
    """Theoretical maximum using Kepler density."""
    Lx, Ly, Lz = container_dims
    container_vol = Lx * Ly * Lz
    sphere_vol = (4/3) * np.pi * radius**3
    return int(container_vol * KEPLER_DENSITY / sphere_vol)


def pack_spheres_random_estimate(radius, container_dims, density=RANDOM_CLOSE_DENSITY):
    """Estimate for random close packing."""
    Lx, Ly, Lz = container_dims
    container_vol = Lx * Ly * Lz
    sphere_vol = (4/3) * np.pi * radius**3
    return int(container_vol * density / sphere_vol)


# ============================================================
# Venues (same as peoplepacking for comparison)
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
}


# ============================================================
# Rendering
# ============================================================

def render_sphere_packing(centers, radius, container_dims, filename, title="",
                           color="#D4956A", window_size=(1400, 900), max_render=300):
    """Render spheres packed in a container."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('#F0EDE8')

    n_total = len(centers)
    if n_total <= max_render:
        indices = range(n_total)
    else:
        indices = np.linspace(0, n_total - 1, max_render, dtype=int)

    # Skin tone palette for spherical humans
    palette = [
        '#D4956A', '#E8C4A8', '#C68642', '#FFDBAC', '#8D5524',
        '#F1C27D', '#A0785A', '#E8B89D', '#C4867A', '#D2A679',
    ]

    for i, idx in enumerate(indices):
        sphere = pv.Sphere(radius=radius, center=centers[idx], theta_resolution=16, phi_resolution=16)
        c = palette[i % len(palette)]
        pl.add_mesh(sphere, color=c, smooth_shading=True, specular=0.5, ambient=0.2)

    # Container wireframe
    Lx, Ly, Lz = container_dims
    container = pv.Box(bounds=[0, Lx, 0, Ly, 0, Lz])
    pl.add_mesh(container, color='#333333', style='wireframe', line_width=3.0, opacity=0.9)

    if title:
        pl.add_title(title, font_size=12, color='#333333')

    aspect = max(Lx, Ly) / max(min(Lx, Ly), 0.1)
    if aspect > 5:
        pl.camera.focal_point = (Lx/2, Ly/2, Lz/2)
        pl.camera.position = (Lx * 1.8, -Ly * 2.5, Lz * 2.5)
        pl.camera.up = (0, 0, 1)
        pl.reset_camera()
        pl.camera.zoom(0.9)
    else:
        pl.reset_camera()
        pl.camera.azimuth = 35
        pl.camera.elevation = 25
        pl.camera.zoom(0.85)

    pl.add_light(pv.Light(position=(Lx*2, Ly*2, Lz*3), intensity=0.7))
    pl.add_light(pv.Light(position=(-Lx, -Ly, Lz*2), intensity=0.3))

    pl.screenshot(filename)
    pl.close()


# ============================================================
# Experiments
# ============================================================

def experiment_1_sphere_comparison():
    """Compare all spherical human approximations."""
    print("=" * 60)
    print("EXPERIMENT 1: The Many Spheres of Man")
    print("=" * 60)

    results = {}
    for name, model in SPHERE_MODELS.items():
        r = model["radius"]
        vol = (4/3) * np.pi * r**3
        print(f"\n  {name}: r={r:.3f}m, vol={vol:.4f}m3")
        print(f"    {model['description']}")
        results[name] = {"radius": r, "volume": vol}

    # Render each sphere model at scale
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for i, (name, model) in enumerate(SPHERE_MODELS.items()):
        ax = axes[i]
        r = model["radius"]
        circle = plt.Circle((0.5, r), r, color=model["color"], alpha=0.7)
        ax.add_patch(circle)

        # Draw human outline for reference (simplified)
        human_x = [0.5 - 0.22, 0.5 - 0.22, 0.5 - 0.1, 0.5 - 0.1,
                    0.5 + 0.1, 0.5 + 0.1, 0.5 + 0.22, 0.5 + 0.22]
        human_y = [0, 0.9, 0.9, 1.55, 1.55, 0.9, 0.9, 0]
        ax.plot(human_x, human_y, 'k--', alpha=0.3, linewidth=1.5)
        # Head
        head = plt.Circle((0.5, 1.65), 0.1, fill=False, color='black', alpha=0.3, linewidth=1.5)
        ax.add_patch(head)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.2, 2.0)
        ax.set_aspect('equal')
        ax.set_title(f"{name}\nr={r:.3f}m", fontsize=10, fontweight='bold')
        ax.axhline(y=0, color='#333', linewidth=0.5)
        ax.grid(True, alpha=0.2)

    plt.suptitle("Spherical Human Approximations\n(dashed = actual human outline, solid = spherical approximation)",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sphere_models.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved sphere_models.png")

    return results


def experiment_2_packing_all_venues():
    """Pack spherical humans into all venues with all approximations."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Spherical Human Packing Across Venues")
    print("=" * 60)

    all_results = {}

    for venue_name, dims in VENUES.items():
        vol = dims[0] * dims[1] * dims[2]
        print(f"\n  {venue_name}: {dims[0]}x{dims[1]}x{dims[2]}m (vol={vol:.1f}m3)")

        venue_results = {}
        for model_name, model in SPHERE_MODELS.items():
            r = model["radius"]
            count_cubic, _ = pack_spheres_simple(r, dims)
            count_fcc, _ = pack_spheres_fcc(r, dims)
            count_theory = pack_spheres_theoretical(r, dims)
            count_random = pack_spheres_random_estimate(r, dims)

            best_actual = max(count_cubic, count_fcc)
            venue_results[model_name] = {
                "cubic": count_cubic,
                "fcc": count_fcc,
                "theoretical_max": count_theory,
                "random_close": count_random,
                "best_actual": best_actual,
            }
            print(f"    {model_name}: cubic={count_cubic}, fcc={count_fcc}, "
                  f"theory={count_theory}, random~{count_random}")

        all_results[venue_name] = venue_results

    # Giant comparison chart
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.ravel()

    for i, (venue_name, venue_data) in enumerate(all_results.items()):
        ax = axes[i]
        models = list(venue_data.keys())
        short_names = [m.split("(")[0].strip().replace("Sphere", "").strip() for m in models]

        cubic_counts = [venue_data[m]["cubic"] for m in models]
        fcc_counts = [venue_data[m]["fcc"] for m in models]
        theory_counts = [venue_data[m]["theoretical_max"] for m in models]

        x = np.arange(len(models))
        w = 0.25

        ax.barh(x - w, cubic_counts, w, label='Simple Cubic', color='#E74C3C', alpha=0.8)
        ax.barh(x, fcc_counts, w, label='FCC', color='#3498DB', alpha=0.8)
        ax.barh(x + w, theory_counts, w, label='Kepler Max', color='#27AE60', alpha=0.8)

        ax.set_yticks(x)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_title(venue_name, fontsize=12, fontweight='bold')
        ax.set_xscale('symlog', linthresh=1)
        if i == 0:
            ax.legend(fontsize=7, loc='lower right')

    plt.suptitle("Spherical Human Packing: All Venues x All Approximations\n"
                 '"Consider a spherical cow human..."',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/venue_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved venue_comparison.png")

    with open(f"{OUTPUT_DIR}/packing_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


def experiment_3_articulated_vs_spherical():
    """Compare spherical packing to articulated results from peoplepacking."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Spherical vs Articulated Human Comparison")
    print("(The Cost of Anatomical Realism)")
    print("=" * 60)

    # Load articulated results if available
    articulated_file = "../peoplepacking/results3d/venue_packing_3d.json"
    if os.path.exists(articulated_file):
        with open(articulated_file) as f:
            articulated_data = json.load(f)
    else:
        print("  WARNING: No articulated results found. Using hardcoded values.")
        articulated_data = {
            "Elevator": {"Standing (arms at sides)": 21},
            "Phone Booth": {"Standing (arms at sides)": 4},
            "Shipping Container": {"Standing (arms at sides)": 108},
            "Minivan": {"Standing (arms at sides)": 10},
            "Boeing 737 Cabin": {"Standing (arms at sides)": 816},
            "Subway Car": {"Standing (arms at sides)": 288},
            "School Bus": {"Standing (arms at sides)": 105},
            "Hot Tub": {"Standing (arms at sides)": 12},
        }

    comparisons = {}
    print(f"\n  {'Venue':<22} {'Articulated':>12} {'Vol-Eq Sphere':>14} {'Height Sphere':>14} {'Ratio (Vol)':>12}")
    print("  " + "-" * 78)

    for venue_name, dims in VENUES.items():
        # Best articulated count
        if venue_name in articulated_data:
            art_best = max(articulated_data[venue_name].values())
        else:
            art_best = 0

        # Volume-equivalent sphere
        r_vol = SPHERE_MODELS["Volume-Equivalent Sphere"]["radius"]
        count_vol, _ = pack_spheres_fcc(r_vol, dims)

        # Height-bounding sphere
        r_height = SPHERE_MODELS["Height-Bounding Sphere"]["radius"]
        count_height, _ = pack_spheres_fcc(r_height, dims)

        ratio = count_vol / art_best if art_best > 0 else float('inf')

        comparisons[venue_name] = {
            "articulated": art_best,
            "sphere_volume_eq": count_vol,
            "sphere_height": count_height,
            "ratio_vol": ratio,
        }

        print(f"  {venue_name:<22} {art_best:>12} {count_vol:>14} {count_height:>14} {ratio:>11.1f}x")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    venues = list(comparisons.keys())
    art_counts = [comparisons[v]["articulated"] for v in venues]
    vol_counts = [comparisons[v]["sphere_volume_eq"] for v in venues]
    height_counts = [comparisons[v]["sphere_height"] for v in venues]

    x = np.arange(len(venues))
    w = 0.25

    ax1.bar(x - w, art_counts, w, label='Articulated (20 poses)', color='#E74C3C', edgecolor='white')
    ax1.bar(x, vol_counts, w, label='Volume-Eq Sphere (FCC)', color='#3498DB', edgecolor='white')
    ax1.bar(x + w, height_counts, w, label='Height-Bounding Sphere (FCC)', color='#F39C12', edgecolor='white')

    ax1.set_xticks(x)
    ax1.set_xticklabels(venues, rotation=35, ha='right', fontsize=9)
    ax1.set_ylabel("Humans Packed")
    ax1.set_yscale('symlog', linthresh=1)
    ax1.set_title("Articulated vs Spherical Human Packing", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Ratio plot
    ratios_vol = [comparisons[v]["ratio_vol"] for v in venues]
    colors = ['#27AE60' if r > 1 else '#E74C3C' for r in ratios_vol]
    ax2.barh(venues, ratios_vol, color=colors, edgecolor='white', height=0.6)
    ax2.axvline(x=1.0, color='#333', linestyle='--', linewidth=1.5, alpha=0.7, label='Parity')
    ax2.set_xlabel("Ratio: Volume-Eq Sphere / Articulated")
    ax2.set_title("The Spherical Approximation Factor\n(>1 = sphere packs more, <1 = articulated wins)",
                  fontsize=14, fontweight='bold')
    ax2.legend()
    for i, v in enumerate(ratios_vol):
        ax2.text(v + 0.1, i, f'{v:.1f}x', va='center', fontsize=9)

    plt.tight_layout(w_pad=3)
    plt.savefig(f"{OUTPUT_DIR}/spherical_vs_articulated.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved spherical_vs_articulated.png")

    return comparisons


def experiment_4_packing_visualizations():
    """Render 3D sphere packing scenes."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Spherical Human Packing Visualizations")
    print("=" * 60)

    # Use volume-equivalent sphere for most scenes
    r = SPHERE_MODELS["Volume-Equivalent Sphere"]["radius"]

    SCENES = [
        ("Elevator",           (2.0, 1.5, 2.4)),
        ("Phone Booth",        (0.9, 0.9, 2.3)),
        ("Shipping Container", (5.9, 2.35, 2.39)),
        ("School Bus",         (7.3, 2.3, 1.8)),
        ("Hot Tub",            (2.1, 2.1, 0.9)),
        ("Boeing 737 Cabin",   (28.0, 3.54, 2.2)),
    ]

    for venue_name, dims in SCENES:
        count_fcc, centers = pack_spheres_fcc(r, dims)
        count_cubic, centers_cubic = pack_spheres_simple(r, dims)

        # Use whichever has more
        if count_fcc >= count_cubic:
            count, ctrs = count_fcc, centers
            method = "FCC"
        else:
            count, ctrs = count_cubic, centers_cubic
            method = "Cubic"

        if count == 0:
            print(f"  {venue_name}: 0 spheres fit")
            continue

        safe = venue_name.replace(" ", "_")
        render_sphere_packing(
            ctrs, r, dims,
            f"{OUTPUT_DIR}/scene_{safe}.png",
            title=f"{venue_name}: {count} spherical humans ({method}, r={r:.3f}m)",
        )
        print(f"  {venue_name}: {count} spheres ({method})")

    # Also render height-bounding sphere for comparison
    r_big = SPHERE_MODELS["Height-Bounding Sphere"]["radius"]
    for venue_name, dims in [("Elevator", (2.0, 1.5, 2.4)),
                               ("Shipping Container", (5.9, 2.35, 2.39))]:
        count, centers = pack_spheres_fcc(r_big, dims)
        if count == 0:
            count, centers = pack_spheres_simple(r_big, dims)
        if count == 0:
            print(f"  {venue_name} (height sphere): 0 fit")
            continue
        safe = venue_name.replace(" ", "_")
        render_sphere_packing(
            centers, r_big, dims,
            f"{OUTPUT_DIR}/scene_{safe}_height_sphere.png",
            title=f"{venue_name}: {count} hamster-ball humans (r={r_big:.3f}m)",
            color="#F39C12",
        )
        print(f"  {venue_name} (height sphere): {count}")


def experiment_5_kepler_analysis():
    """Analyze the gap between our FCC packing and the Kepler limit."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Approaching the Kepler Limit")
    print("(How close can we get to 74.05%?)")
    print("=" * 60)

    r = SPHERE_MODELS["Volume-Equivalent Sphere"]["radius"]
    sphere_vol = (4/3) * np.pi * r**3

    results = []
    for venue_name, dims in VENUES.items():
        container_vol = dims[0] * dims[1] * dims[2]
        count_fcc, _ = pack_spheres_fcc(r, dims)
        count_cubic, _ = pack_spheres_simple(r, dims)
        best = max(count_fcc, count_cubic)

        actual_density = best * sphere_vol / container_vol
        kepler_ratio = actual_density / KEPLER_DENSITY
        theoretical = pack_spheres_theoretical(r, dims)

        results.append({
            "venue": venue_name,
            "actual_count": best,
            "theoretical_max": theoretical,
            "actual_density": actual_density,
            "kepler_ratio": kepler_ratio,
            "wasted_space": 1 - actual_density,
        })

        print(f"  {venue_name:<22} actual={best:>6}  theory={theoretical:>6}  "
              f"density={actual_density:.1%}  ({kepler_ratio:.1%} of Kepler)")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    venues = [r["venue"] for r in results]
    densities = [r["actual_density"] * 100 for r in results]
    kepler_ratios = [r["kepler_ratio"] * 100 for r in results]

    colors = plt.cm.RdYlGn([kr/100 for kr in kepler_ratios])

    ax1.barh(venues, densities, color=colors, edgecolor='white', height=0.6)
    ax1.axvline(x=KEPLER_DENSITY * 100, color='#E74C3C', linestyle='--', linewidth=2,
                label=f'Kepler limit ({KEPLER_DENSITY:.1%})')
    ax1.axvline(x=RANDOM_CLOSE_DENSITY * 100, color='#F39C12', linestyle='--', linewidth=1.5,
                label=f'Random close ({RANDOM_CLOSE_DENSITY:.0%})')
    ax1.axvline(x=52.36, color='#3498DB', linestyle=':', linewidth=1.5,
                label='Simple cubic (52.4%)')
    ax1.set_xlabel("Packing Density (%)")
    ax1.set_title("Actual Packing Density vs Theoretical Limits", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower right')

    # Efficiency chart
    actual_counts = [r["actual_count"] for r in results]
    theory_counts = [r["theoretical_max"] for r in results]
    efficiency = [a/t * 100 if t > 0 else 0 for a, t in zip(actual_counts, theory_counts)]

    ax2.barh(venues, efficiency, color='#3498DB', edgecolor='white', height=0.6, alpha=0.8)
    ax2.axvline(x=100, color='#E74C3C', linestyle='--', linewidth=2, label='Kepler limit')
    ax2.set_xlabel("% of Kepler Maximum Achieved")
    ax2.set_title("FCC Packing Efficiency\n(wall effects reduce density in small containers)",
                  fontsize=13, fontweight='bold')
    ax2.legend()
    for i, e in enumerate(efficiency):
        ax2.text(e + 1, i, f'{e:.0f}%', va='center', fontsize=9)

    plt.tight_layout(w_pad=3)
    plt.savefig(f"{OUTPUT_DIR}/kepler_analysis.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved kepler_analysis.png")

    return results


def experiment_6_radius_sweep():
    """How does sphere radius affect packing? Sweep from tiny to huge."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: The Radius-Packing Relationship")
    print("(From marble-humans to beach-ball-humans)")
    print("=" * 60)

    CONTAINER = (5.9, 2.35, 2.39)  # Shipping container
    container_vol = CONTAINER[0] * CONTAINER[1] * CONTAINER[2]

    radii = np.linspace(0.05, 1.2, 50)
    fcc_counts = []
    cubic_counts = []
    theory_counts = []

    for r in radii:
        fcc_c, _ = pack_spheres_fcc(r, CONTAINER)
        cubic_c, _ = pack_spheres_simple(r, CONTAINER)
        theory_c = pack_spheres_theoretical(r, CONTAINER)
        fcc_counts.append(fcc_c)
        cubic_counts.append(cubic_c)
        theory_counts.append(theory_c)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(radii, theory_counts, 'r--', linewidth=2, label='Kepler theoretical', alpha=0.7)
    ax1.plot(radii, fcc_counts, 'b-', linewidth=2.5, label='FCC lattice', marker='o', markersize=3)
    ax1.plot(radii, cubic_counts, 'g-', linewidth=1.5, label='Simple cubic', alpha=0.7)
    ax1.set_xlabel("Sphere Radius (m)")
    ax1.set_ylabel("Spheres per Shipping Container")
    ax1.set_yscale('symlog', linthresh=1)
    ax1.set_title("Packing Count vs Sphere Radius\n(20ft Shipping Container)", fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mark the different models
    for name, model in SPHERE_MODELS.items():
        r = model["radius"]
        if 0.05 <= r <= 1.2:
            short = name.split("(")[0].strip().replace("Sphere", "").strip()
            c, _ = pack_spheres_fcc(r, CONTAINER)
            ax1.annotate(short, (r, max(c, 1)), fontsize=7, rotation=45,
                        ha='left', va='bottom', color=model["color"],
                        fontweight='bold')
            ax1.plot(r, c, 'o', color=model["color"], markersize=8, zorder=5)

    # Density plot
    fcc_densities = [c * (4/3) * np.pi * r**3 / container_vol
                     for c, r in zip(fcc_counts, radii)]
    ax2.plot(radii, [d * 100 for d in fcc_densities], 'b-', linewidth=2.5)
    ax2.axhline(y=KEPLER_DENSITY * 100, color='r', linestyle='--', linewidth=2,
                label=f'Kepler limit ({KEPLER_DENSITY:.1%})', alpha=0.7)
    ax2.set_xlabel("Sphere Radius (m)")
    ax2.set_ylabel("Packing Density (%)")
    ax2.set_title("Packing Density vs Sphere Radius\n(approaches Kepler limit for small spheres)",
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 80)

    plt.tight_layout(w_pad=3)
    plt.savefig(f"{OUTPUT_DIR}/radius_sweep.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("\nSaved radius_sweep.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  SPHERICAL HUMAN PACKING")
    print('  "Consider a spherical human of uniform density..."')
    print("=" * 60)
    print()

    experiment_1_sphere_comparison()
    experiment_2_packing_all_venues()
    experiment_3_articulated_vs_spherical()
    experiment_4_packing_visualizations()
    experiment_5_kepler_analysis()
    experiment_6_radius_sweep()

    print("\n" + "=" * 60)
    print("All spherical human experiments complete.")
    print(f"Results saved to ./{OUTPUT_DIR}/")
    print("=" * 60)
