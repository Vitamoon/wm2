"""
Experiment 2: Recursive Autoencoding - The Telephone Game
==========================================================
What happens when you pass a signal through an autoencoder, then pass
the reconstruction through ANOTHER autoencoder, then ANOTHER...

This is the neural network equivalent of whispering a secret around
a circle and seeing what comes back.

Spoiler: it's not the secret.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Generate interesting 2D data distributions
# ============================================================

def generate_swiss_roll(n=3000):
    """A swiss roll - topologically interesting, visually pretty."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t)
    y = t * np.sin(t)
    data = np.stack([x, y], axis=1).astype(np.float32)
    data = (data - data.mean(0)) / data.std(0)
    return data


def generate_smiley(n=3000):
    """A smiley face distribution. Because science."""
    points = []

    # Face outline
    for _ in range(n // 3):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2.0 + np.random.normal(0, 0.05)
        points.append([r * np.cos(angle), r * np.sin(angle)])

    # Eyes
    for _ in range(n // 6):
        points.append([np.random.normal(-0.7, 0.15), np.random.normal(0.7, 0.15)])
        points.append([np.random.normal(0.7, 0.15), np.random.normal(0.7, 0.15)])

    # Smile (arc)
    for _ in range(n // 3):
        angle = np.random.uniform(-2.3, -0.8)
        r = 1.2 + np.random.normal(0, 0.05)
        points.append([r * np.cos(angle), r * np.sin(angle)])

    data = np.array(points[:n], dtype=np.float32)
    data = (data - data.mean(0)) / data.std(0)
    return data


# ============================================================
# Autoencoder
# ============================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, bottleneck_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(data, epochs=300, lr=1e-3, hidden_dim=32, bottleneck_dim=2):
    """Train a single autoencoder on the given data."""
    dataset = TensorDataset(torch.FloatTensor(data).to(DEVICE))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = Autoencoder(
        input_dim=data.shape[1],
        hidden_dim=hidden_dim,
        bottleneck_dim=bottleneck_dim
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for (xb,) in loader:
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            loss.backward()
            optimizer.step()

    return model


def reconstruct(model, data):
    """Pass data through autoencoder and return reconstruction."""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(data).to(DEVICE)
        recon, z = model(x)
    return recon.cpu().numpy(), z.cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def compute_distribution_stats(original, reconstructed):
    """Compare distributions via various metrics."""
    mse = np.mean((original - reconstructed) ** 2)

    # Variance ratio: how much variance is preserved?
    orig_var = np.var(original, axis=0).sum()
    recon_var = np.var(reconstructed, axis=0).sum()
    var_ratio = recon_var / (orig_var + 1e-10)

    # Correlation between dimensions
    if original.shape[1] >= 2:
        orig_corr = np.corrcoef(original[:, 0], original[:, 1])[0, 1]
        recon_corr = np.corrcoef(reconstructed[:, 0], reconstructed[:, 1])[0, 1]
        corr_diff = abs(orig_corr - recon_corr)
    else:
        corr_diff = 0.0

    return {
        "mse": float(mse),
        "variance_ratio": float(var_ratio),
        "correlation_shift": float(corr_diff),
        "mean_shift": float(np.linalg.norm(original.mean(0) - reconstructed.mean(0))),
    }


# ============================================================
# Main Experiment
# ============================================================

def run_telephone_game(data, max_depth=10, epochs=300, name="distribution", bottleneck_dim=2):
    """
    The Neural Telephone Game:
    1. Train autoencoder on original data
    2. Reconstruct data
    3. Train NEW autoencoder on reconstructed data
    4. Reconstruct again
    5. Repeat until the data is unrecognizable
    """
    print(f"\n{'='*60}")
    print(f"THE NEURAL TELEPHONE GAME: {name}")
    print(f"{'='*60}")

    results = {
        "depths": [],
        "mse_to_original": [],
        "variance_ratio": [],
        "correlation_shift": [],
        "mean_shift": [],
    }
    reconstructions = [data.copy()]
    current_data = data.copy()

    for depth in range(max_depth):
        print(f"\n  Depth {depth}: Training autoencoder on {'original' if depth == 0 else 'reconstructed'} data...")

        model = train_autoencoder(current_data, epochs=epochs, bottleneck_dim=bottleneck_dim)
        recon, latent = reconstruct(model, current_data)
        reconstructions.append(recon.copy())

        # Always compare to ORIGINAL data
        stats = compute_distribution_stats(data, recon)
        print(f"    MSE to original: {stats['mse']:.6f}")
        print(f"    Variance ratio:  {stats['variance_ratio']:.4f}")
        print(f"    Corr shift:      {stats['correlation_shift']:.4f}")

        results["depths"].append(depth)
        results["mse_to_original"].append(stats["mse"])
        results["variance_ratio"].append(stats["variance_ratio"])
        results["correlation_shift"].append(stats["correlation_shift"])
        results["mean_shift"].append(stats["mean_shift"])

        # The reconstructed data becomes the training data for the next autoencoder
        current_data = recon

    return results, reconstructions


def plot_telephone_game(reconstructions, name="distribution"):
    """
    The money shot: show the data degrading through recursive autoencoding.
    """
    n_show = min(len(reconstructions), 8)
    indices = np.linspace(0, len(reconstructions) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))

    captions = [
        "Original\n(blissfully unaware)",
        "Depth 1\n(mild concern)",
        "Depth 2\n(something is wrong)",
        "Depth 3\n(I don't feel so good)",
        "Depth 4\n(who am I?)",
        "Depth 5\n(the void stares back)",
        "Depth 6\n(acceptance)",
        "Depth 7\n(we are sorry)",
        "Depth 8\n(heat death)",
        "Depth 9\n(...))",
    ]

    for i, idx in enumerate(indices):
        ax = axes[i]
        data = reconstructions[idx]
        ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.5, c='#3498db')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        caption = captions[idx] if idx < len(captions) else f"Depth {idx}\n(beyond help)"
        ax.set_title(caption, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"The Neural Telephone Game: {name}\n"
        f"(Each autoencoder trains on the previous one's reconstruction)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"exp2_telephone_{name}.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / f"exp2_telephone_{name}.pdf", bbox_inches='tight')
    print(f"  Figure saved: exp2_telephone_{name}.png")


def plot_metrics(all_results):
    """Plot degradation metrics for all distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for name, results in all_results.items():
        axes[0].plot(results["depths"], results["mse_to_original"], 'o-', label=name, markersize=5)
        axes[1].plot(results["depths"], results["variance_ratio"], 's-', label=name, markersize=5)
        axes[2].plot(results["depths"], results["correlation_shift"], '^-', label=name, markersize=5)

    axes[0].set_xlabel("Recursion Depth")
    axes[0].set_ylabel("MSE to Original")
    axes[0].set_title("Reconstruction Error\n(how far from the truth)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Recursion Depth")
    axes[1].set_ylabel("Variance Ratio")
    axes[1].set_title("Variance Preservation\n(how much life remains)")
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Recursion Depth")
    axes[2].set_ylabel("|Correlation Shift|")
    axes[2].set_title("Structural Distortion\n(how wrong the vibes are)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Quantitative Degradation Across Recursive Autoencoding Depths",
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp2_metrics.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "exp2_metrics.pdf", bbox_inches='tight')
    print(f"\nMetrics figure saved: exp2_metrics.png")


if __name__ == "__main__":
    all_results = {}

    # Swiss roll - 1D bottleneck (lossy compression forces visible degradation)
    print("Generating swiss roll data...")
    swiss_data = generate_swiss_roll(3000)
    results_swiss, recons_swiss = run_telephone_game(
        swiss_data, max_depth=10, name="Swiss Roll (1D bottleneck)",
        epochs=300, bottleneck_dim=1
    )
    plot_telephone_game(recons_swiss, name="Swiss Roll (1D bottleneck)")
    all_results["Swiss Roll (1D)"] = results_swiss

    # Smiley face - 1D bottleneck
    print("\nGenerating smiley face data...")
    smiley_data = generate_smiley(3000)
    results_smiley, recons_smiley = run_telephone_game(
        smiley_data, max_depth=10, name="Smiley Face (1D bottleneck)",
        epochs=300, bottleneck_dim=1
    )
    plot_telephone_game(recons_smiley, name="Smiley Face (1D bottleneck)")
    all_results["Smiley Face (1D)"] = results_smiley

    # Also run with 2D bottleneck (full capacity) for comparison
    print("\nGenerating smiley face data (2D bottleneck - control)...")
    results_smiley2d, recons_smiley2d = run_telephone_game(
        smiley_data, max_depth=10, name="Smiley Face (2D bottleneck)",
        epochs=300, bottleneck_dim=2
    )
    plot_telephone_game(recons_smiley2d, name="Smiley Face (2D bottleneck)")
    all_results["Smiley Face (2D)"] = results_smiley2d

    plot_metrics(all_results)

    # Save results
    with open("exp2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll results saved to exp2_results.json")
