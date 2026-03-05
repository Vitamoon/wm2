"""
Experiment 1: Recursive Neural Network World Models
=====================================================
Train MLPs in a recursive hierarchy where each model tries to predict
the internal representations of the model below it.

W0: Predicts environment dynamics (ground truth world model)
W1: Predicts W0's hidden representations
W2: Predicts W1's hidden representations
...
Wn: Has no idea what's going on

Metrics: MSE, representation entropy, CKA similarity, prediction R²
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy
from pathlib import Path
import json

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Environment: Chaotic Lorenz System
# ============================================================

def generate_lorenz_data(n_steps=10000, dt=0.01, sigma=10.0, rho=28.0, beta=8/3):
    """Generate trajectories from the Lorenz attractor - a real chaotic system."""
    states = np.zeros((n_steps, 3))
    states[0] = [1.0, 1.0, 1.0]  # initial condition

    for i in range(1, n_steps):
        x, y, z = states[i-1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        states[i] = states[i-1] + np.array([dx, dy, dz]) * dt

    # Normalize to zero mean, unit variance
    mean = states.mean(axis=0)
    std = states.std(axis=0)
    states = (states - mean) / std

    # Create input/target pairs: predict next state from current
    X = states[:-1]
    Y = states[1:]
    return X.astype(np.float32), Y.astype(np.float32)


# ============================================================
# World Model Architecture
# ============================================================

class WorldModel(nn.Module):
    """Simple MLP world model with accessible hidden representations."""

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim = hidden_dim

    def get_hidden(self, x):
        """Return hidden representation (the thing the next model tries to predict)."""
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        return self.decoder(h), h


# ============================================================
# Metrics
# ============================================================

def compute_cka(X, Y):
    """
    Centered Kernel Alignment - measures representational similarity.
    CKA = 1.0 means identical representations, 0.0 means unrelated.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    XX = X @ X.T
    YY = Y @ Y.T

    hsic_xy = (XX * YY).sum()
    hsic_xx = (XX * XX).sum()
    hsic_yy = (YY * YY).sum()

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return float(hsic_xy / denom)


def compute_representation_entropy(H, n_bins=50):
    """Estimate entropy of hidden representations via histogram binning."""
    H_np = H.detach().cpu().numpy()
    entropies = []
    for dim in range(H_np.shape[1]):
        counts, _ = np.histogram(H_np[:, dim], bins=n_bins, density=False)
        counts = counts + 1e-10  # avoid log(0)
        entropies.append(scipy_entropy(counts / counts.sum()))
    return float(np.mean(entropies))


def compute_r_squared(predictions, targets):
    """R² score - 1.0 is perfect, 0.0 is mean-baseline, negative is worse than guessing."""
    ss_res = ((predictions - targets) ** 2).sum()
    ss_tot = ((targets - targets.mean(0)) ** 2).sum()
    if ss_tot < 1e-10:
        return 0.0
    return float(1 - ss_res / ss_tot)


# ============================================================
# Training
# ============================================================

def train_model(model, X_train, Y_train, epochs=200, lr=1e-3, batch_size=256):
    """Train a world model, return training losses."""
    dataset = TensorDataset(
        torch.FloatTensor(X_train).to(DEVICE),
        torch.FloatTensor(Y_train).to(DEVICE)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    return losses


def extract_hidden_representations(model, X):
    """Get hidden activations from a trained model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        H = model.get_hidden(X_tensor)
    return H.cpu().numpy()


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(max_depth=7, hidden_dim=64, epochs=200):
    """Run the recursive world model experiment."""
    print("=" * 60)
    print("EXPERIMENT 1: RECURSIVE NEURAL WORLD MODELS")
    print("=" * 60)
    print(f"Environment: Lorenz attractor (chaotic)")
    print(f"Max depth: {max_depth}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Training epochs per model: {epochs}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Generate environment data
    print("\nGenerating Lorenz attractor data...")
    X_env, Y_env = generate_lorenz_data(n_steps=12000)
    X_train, X_test = X_env[:10000], X_env[10000:]
    Y_train, Y_test = Y_env[:10000], Y_env[10000:]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    results = {
        "depths": [],
        "mse": [],
        "r_squared": [],
        "entropy": [],
        "cka_to_ground": [],
        "cka_to_parent": [],
        "training_losses": [],
    }

    models = []
    hidden_reps = {}

    # Collect ground truth hidden reps from depth 0
    prev_X_train = X_train
    prev_Y_train = Y_train
    prev_X_test = X_test
    prev_Y_test = Y_test
    input_dim = 3  # Lorenz system is 3D

    for depth in range(max_depth):
        print(f"\n--- Depth {depth}: W{depth} ---")

        if depth == 0:
            in_dim = input_dim
            out_dim = input_dim
        else:
            in_dim = hidden_dim
            out_dim = hidden_dim

        model = WorldModel(in_dim, out_dim, hidden_dim=hidden_dim).to(DEVICE)

        # Train
        print(f"  Training W{depth} ({in_dim} -> {hidden_dim} -> {out_dim})...")
        losses = train_model(model, prev_X_train, prev_Y_train, epochs=epochs)
        models.append(model)

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(prev_X_test).to(DEVICE)
            Y_t = torch.FloatTensor(prev_Y_test).to(DEVICE)
            pred, h = model(X_t)
            mse = nn.MSELoss()(pred, Y_t).item()

        pred_np = pred.cpu().numpy()
        h_np = h.cpu().numpy()
        r2 = compute_r_squared(pred_np, prev_Y_test)
        ent = compute_representation_entropy(h)

        # Store hidden representations
        hidden_reps[depth] = h_np

        # CKA to ground truth (depth 0's hidden reps)
        if depth == 0:
            ground_hidden = extract_hidden_representations(model, X_test)
            cka_ground = 1.0
        else:
            cka_ground = compute_cka(
                h_np[:min(1000, len(h_np))],
                ground_hidden[:min(1000, len(ground_hidden))]
            )

        # CKA to parent
        if depth == 0:
            cka_parent = 1.0
        else:
            cka_parent = compute_cka(
                h_np[:min(1000, len(h_np))],
                hidden_reps[depth - 1][:min(1000, len(hidden_reps[depth - 1]))]
            )

        print(f"  MSE: {mse:.6f}")
        print(f"  R^2: {r2:.4f}")
        print(f"  Entropy: {ent:.4f}")
        print(f"  CKA to ground: {cka_ground:.4f}")
        print(f"  CKA to parent: {cka_parent:.4f}")
        print(f"  Final train loss: {losses[-1]:.6f}")

        results["depths"].append(depth)
        results["mse"].append(mse)
        results["r_squared"].append(r2)
        results["entropy"].append(ent)
        results["cka_to_ground"].append(cka_ground)
        results["cka_to_parent"].append(cka_parent)
        results["training_losses"].append(losses)

        # Prepare data for next depth: next model predicts THIS model's hidden reps
        h_train = extract_hidden_representations(model, prev_X_train)
        h_test = extract_hidden_representations(model, prev_X_test)

        # The target for the next model is predicting THIS model's hidden reps
        # from the SAME input (creating the recursive prediction chain)
        prev_X_train = h_train
        prev_Y_train = extract_hidden_representations(model, Y_train if depth == 0 else prev_Y_train)
        prev_X_test = h_test
        prev_Y_test = extract_hidden_representations(model, Y_test if depth == 0 else prev_Y_test)

        # Wait actually, for the recursive chain to make sense:
        # W_{n+1} takes W_n's hidden(x_t) and tries to predict W_n's hidden(x_{t+1})
        # i.e., it's trying to learn the dynamics in W_n's representation space

    return results, models


# ============================================================
# Plotting
# ============================================================

def plot_results(results):
    """Generate publication-quality figures."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        "Recursive World Model Degradation\n"
        "(Each model predicts the internal dynamics of the model below it)",
        fontsize=14, fontweight='bold'
    )

    depths = results["depths"]

    # 1. MSE by depth
    ax = axes[0, 0]
    ax.plot(depths, results["mse"], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("Test MSE")
    ax.set_title("Prediction Error vs. Depth")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2. R² by depth
    ax = axes[0, 1]
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in results["r_squared"]]
    ax.bar(depths, results["r_squared"], color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Chance level')
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("R² Score")
    ax.set_title("Prediction Quality vs. Depth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Representation entropy
    ax = axes[0, 2]
    ax.plot(depths, results["entropy"], 's-', color='#9b59b6', linewidth=2, markersize=8)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("Mean Entropy (nats)")
    ax.set_title("Representation Entropy vs. Depth")
    ax.grid(True, alpha=0.3)

    # 4. CKA to ground truth
    ax = axes[1, 0]
    ax.plot(depths, results["cka_to_ground"], 'D-', color='#3498db', linewidth=2, markersize=8)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("CKA to Ground Truth (W0) vs. Depth")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 5. CKA to parent
    ax = axes[1, 1]
    ax.plot(depths, results["cka_to_parent"], '^-', color='#f39c12', linewidth=2, markersize=8)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("CKA to Parent Model vs. Depth")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 6. Training loss curves (all depths overlaid)
    ax = axes[1, 2]
    cmap = plt.cm.viridis
    for i, losses in enumerate(results["training_losses"]):
        color = cmap(i / max(1, len(results["training_losses"]) - 1))
        ax.plot(losses, color=color, alpha=0.8, label=f'W{i}')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Convergence by Depth")
    ax.set_yscale('log')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp1_recursive_worldmodels.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "exp1_recursive_worldmodels.pdf", bbox_inches='tight')
    print(f"\nFigures saved to {FIGURES_DIR}/exp1_recursive_worldmodels.png/.pdf")


def plot_summary_table(results):
    """Print and save a results summary table."""
    print("\n" + "=" * 70)
    print(f"{'Depth':<8}{'MSE':<12}{'R^2':<10}{'Entropy':<10}{'CKA(W0)':<10}{'CKA(parent)'}")
    print("-" * 70)
    for i in range(len(results["depths"])):
        print(
            f"W{results['depths'][i]:<7}"
            f"{results['mse'][i]:<12.6f}"
            f"{results['r_squared'][i]:<10.4f}"
            f"{results['entropy'][i]:<10.4f}"
            f"{results['cka_to_ground'][i]:<10.4f}"
            f"{results['cka_to_parent'][i]:.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    results, models = run_experiment(max_depth=7, hidden_dim=64, epochs=200)
    plot_results(results)
    plot_summary_table(results)

    # Save raw results
    save_results = {k: v for k, v in results.items() if k != "training_losses"}
    save_results["training_final_losses"] = [l[-1] for l in results["training_losses"]]
    with open("exp1_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print("\nResults saved to exp1_results.json")
