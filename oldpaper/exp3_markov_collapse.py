"""
Experiment 3: Recursive Model Collapse via Markov Chains
=========================================================
The simplest possible demonstration of recursive model collapse:

1. Start with a "true" distribution over sequences
2. Estimate a Markov chain from samples
3. Generate new samples from the estimated chain
4. Estimate a NEW Markov chain from those synthetic samples
5. Repeat until all information has been destroyed

This mirrors the real-world concern about training AI on AI-generated data,
but with the mathematical elegance of a system we can fully characterize.

We know the ground truth. We can measure exactly how much is lost.
The answer is: all of it.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

np.random.seed(42)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)


# ============================================================
# Markov Chain Operations
# ============================================================

def create_interesting_chain(n_states=8):
    """
    Create a Markov chain with interesting structure -
    two clusters of states with rare transitions between them.
    Like two friend groups that occasionally interact at parties.
    """
    T = np.zeros((n_states, n_states))

    half = n_states // 2

    # Cluster 1: states 0..half-1
    for i in range(half):
        for j in range(half):
            T[i, j] = np.random.exponential(1.0)
        # Rare transitions to cluster 2
        for j in range(half, n_states):
            T[i, j] = np.random.exponential(0.05)

    # Cluster 2: states half..n-1
    for i in range(half, n_states):
        for j in range(half, n_states):
            T[i, j] = np.random.exponential(1.0)
        # Rare transitions to cluster 1
        for j in range(half):
            T[i, j] = np.random.exponential(0.05)

    # Normalize rows
    T = T / T.sum(axis=1, keepdims=True)
    return T


def stationary_distribution(T):
    """Compute stationary distribution via eigendecomposition."""
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    return np.abs(pi)  # ensure non-negative


def generate_sequence(T, length=5000, start=0):
    """Generate a sequence from a Markov chain."""
    n_states = T.shape[0]
    seq = [start]
    for _ in range(length - 1):
        current = seq[-1]
        next_state = np.random.choice(n_states, p=T[current])
        seq.append(next_state)
    return np.array(seq)


def estimate_chain(sequence, n_states):
    """Estimate transition matrix from an observed sequence."""
    T = np.zeros((n_states, n_states))
    for i in range(len(sequence) - 1):
        T[sequence[i], sequence[i+1]] += 1

    # Laplace smoothing (avoid zero rows)
    T += 1e-6

    # Normalize
    T = T / T.sum(axis=1, keepdims=True)
    return T


def kl_divergence_chains(P, Q):
    """KL divergence between two Markov chains (averaged over rows)."""
    kl = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 1e-10:
                kl += P[i, j] * np.log(P[i, j] / max(Q[i, j], 1e-10))
    return kl / P.shape[0]


def total_variation_chains(P, Q):
    """Total variation distance between chains."""
    return 0.5 * np.abs(P - Q).sum() / P.shape[0]


def matrix_entropy(T):
    """Average entropy of transition distributions."""
    entropies = []
    for i in range(T.shape[0]):
        row = T[i]
        row = row[row > 1e-10]
        entropies.append(-np.sum(row * np.log(row)))
    return np.mean(entropies)


# ============================================================
# Main Experiment
# ============================================================

def run_collapse_experiment(n_states=8, seq_length=5000, max_depth=15):
    """
    The Recursive Collapse:
    At each step, we estimate a chain from data, generate new data,
    and estimate again. Like a game of telephone, but with probability.
    """
    print("=" * 60)
    print("EXPERIMENT 3: RECURSIVE MARKOV CHAIN COLLAPSE")
    print("=" * 60)
    print(f"States: {n_states}, Sequence length: {seq_length}")
    print(f"Max recursion depth: {max_depth}")

    # Ground truth
    T_true = create_interesting_chain(n_states)
    pi_true = stationary_distribution(T_true)
    entropy_true = matrix_entropy(T_true)

    print(f"\nGround truth chain entropy: {entropy_true:.4f}")
    print(f"Max entropy (uniform): {np.log(n_states):.4f}")
    print(f"True stationary dist: {np.round(pi_true, 3)}")

    results = {
        "depths": [],
        "kl_to_true": [],
        "tv_to_true": [],
        "entropy": [],
        "entropy_ratio": [],
        "stationary_kl": [],
        "max_transition_prob": [],
    }

    chains = [T_true]
    current_chain = T_true

    for depth in range(max_depth):
        # Generate data from current chain
        seq = generate_sequence(current_chain, length=seq_length)

        # Estimate new chain from generated data
        T_estimated = estimate_chain(seq, n_states)
        chains.append(T_estimated)

        # Metrics
        kl = kl_divergence_chains(T_true, T_estimated)
        tv = total_variation_chains(T_true, T_estimated)
        ent = matrix_entropy(T_estimated)
        pi_est = stationary_distribution(T_estimated)

        # KL between stationary distributions
        stat_kl = np.sum(pi_true * np.log(pi_true / np.maximum(pi_est, 1e-10)))

        # Maximum transition probability (measure of "peakiness")
        max_p = T_estimated.max()

        print(f"\n  Depth {depth}:")
        print(f"    KL(true || estimated): {kl:.6f}")
        print(f"    TV distance:           {tv:.6f}")
        print(f"    Chain entropy:         {ent:.4f} / {np.log(n_states):.4f}")
        print(f"    Stationary KL:         {stat_kl:.6f}")
        print(f"    Max transition prob:   {max_p:.4f}")

        results["depths"].append(depth)
        results["kl_to_true"].append(float(kl))
        results["tv_to_true"].append(float(tv))
        results["entropy"].append(float(ent))
        results["entropy_ratio"].append(float(ent / np.log(n_states)))
        results["stationary_kl"].append(float(stat_kl))
        results["max_transition_prob"].append(float(max_p))

        current_chain = T_estimated

    return results, chains, T_true


# ============================================================
# Plotting
# ============================================================

def plot_chain_evolution(chains, T_true, max_show=8):
    """Visualize how the transition matrix changes over recursive depths."""
    n_show = min(len(chains), max_show)
    indices = [0] + list(np.linspace(1, len(chains) - 1, n_show - 1, dtype=int))

    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))

    titles = [
        "Ground Truth\n(the real thing)",
        "Depth 1\n(pretty close)",
        "Depth 2\n(minor drift)",
        "Depth 4\n(losing structure)",
        "Depth 6\n(what clusters?)",
        "Depth 8\n(approaching heat death)",
        "Depth 10\n(maximum entropy)",
        "Depth 14\n(the void)",
    ]

    for i, idx in enumerate(indices):
        ax = axes[i]
        im = ax.imshow(chains[idx], cmap='viridis', vmin=0, vmax=0.5)
        title = titles[i] if i < len(titles) else f"Depth {idx}"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Transition Matrix Evolution Through Recursive Estimation\n"
        "(Watch the structure dissolve into uniform noise)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_chain_evolution.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "exp3_chain_evolution.pdf", bbox_inches='tight')
    print(f"\nChain evolution figure saved")


def plot_collapse_metrics(results):
    """Plot the metrics of collapse."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    depths = results["depths"]

    # KL divergence
    ax = axes[0, 0]
    ax.plot(depths, results["kl_to_true"], 'o-', color='#e74c3c', linewidth=2)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("KL Divergence from Truth")
    ax.set_title("KL Divergence to Ground Truth\n(how wrong we are)")
    ax.grid(True, alpha=0.3)

    # Entropy ratio
    ax = axes[0, 1]
    ax.plot(depths, results["entropy_ratio"], 's-', color='#9b59b6', linewidth=2)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Maximum entropy (heat death)')
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("Entropy / Max Entropy")
    ax.set_title("Entropy Ratio\n(1.0 = all information is gone)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # TV distance
    ax = axes[1, 0]
    ax.plot(depths, results["tv_to_true"], '^-', color='#3498db', linewidth=2)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("Total Variation Distance")
    ax.set_title("Total Variation from Ground Truth\n(the gap between reality and belief)")
    ax.grid(True, alpha=0.3)

    # Stationary distribution KL
    ax = axes[1, 1]
    ax.plot(depths, results["stationary_kl"], 'D-', color='#f39c12', linewidth=2)
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Stationary Distribution Divergence\n(long-run behavior drift)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Recursive Model Collapse in Markov Chains\n"
        "(Training on your own outputs: a cautionary tale)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_collapse_metrics.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "exp3_collapse_metrics.pdf", bbox_inches='tight')
    print(f"Collapse metrics figure saved")


def run_multiple_trials(n_trials=20, n_states=8, seq_length=5000, max_depth=15):
    """Run multiple trials to get error bars (because we're doing real science here)."""
    print(f"\n{'='*60}")
    print(f"RUNNING {n_trials} TRIALS FOR ERROR BARS")
    print(f"(Yes, we are putting error bars on recursive existential collapse)")
    print(f"{'='*60}")

    all_kl = []
    all_entropy = []

    for trial in range(n_trials):
        np.random.seed(trial * 100)
        T_true = create_interesting_chain(n_states)
        current_chain = T_true
        trial_kl = []
        trial_ent = []

        for depth in range(max_depth):
            seq = generate_sequence(current_chain, length=seq_length)
            T_est = estimate_chain(seq, n_states)
            trial_kl.append(kl_divergence_chains(T_true, T_est))
            trial_ent.append(matrix_entropy(T_est) / np.log(n_states))
            current_chain = T_est

        all_kl.append(trial_kl)
        all_entropy.append(trial_ent)

    all_kl = np.array(all_kl)
    all_entropy = np.array(all_entropy)

    # Plot with error bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    depths = range(max_depth)

    ax = axes[0]
    mean_kl = all_kl.mean(0)
    std_kl = all_kl.std(0)
    ax.plot(depths, mean_kl, 'o-', color='#e74c3c', linewidth=2)
    ax.fill_between(depths, mean_kl - std_kl, mean_kl + std_kl, alpha=0.2, color='#e74c3c')
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("KL Divergence from Truth")
    ax.set_title(f"Model Collapse (n={n_trials} trials)\nMean +/- 1 std")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    mean_ent = all_entropy.mean(0)
    std_ent = all_entropy.std(0)
    ax.plot(depths, mean_ent, 's-', color='#9b59b6', linewidth=2)
    ax.fill_between(depths, mean_ent - std_ent, mean_ent + std_ent, alpha=0.2, color='#9b59b6')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Heat death')
    ax.set_xlabel("Recursion Depth")
    ax.set_ylabel("Entropy / Max Entropy")
    ax.set_title(f"Entropy Convergence to Maximum\n(information heat death)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Recursive Model Collapse: Statistical Analysis\n"
        "(With error bars, because even the void deserves rigor)",
        fontsize=13, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_errorbars.png", dpi=200, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "exp3_errorbars.pdf", bbox_inches='tight')
    print(f"\nError bar figure saved")

    return all_kl, all_entropy


if __name__ == "__main__":
    # Main experiment
    results, chains, T_true = run_collapse_experiment(
        n_states=8, seq_length=5000, max_depth=15
    )
    plot_chain_evolution(chains, T_true)
    plot_collapse_metrics(results)

    # Multi-trial with error bars
    all_kl, all_entropy = run_multiple_trials(n_trials=20)

    # Save
    with open("exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to exp3_results.json")
