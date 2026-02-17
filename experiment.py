"""
Recursive World Model Experiment
================================
Testing the hypothesis that world models understanding world models
in a world model simulated by a world model leads to representational collapse.

Author: Kaelan (UC Berkeley)
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import math

# Set seed for reproducibility (or as reproducible as recursive existential doubt allows)
np.random.seed(42)
random.seed(42)


@dataclass
class WorldState:
    """Represents a state in the 'ground truth' world (if such a thing exists)"""
    position: np.ndarray
    velocity: np.ndarray
    time: int

    def step(self) -> 'WorldState':
        """Physics simulation - the 'real' dynamics"""
        # Simple harmonic motion with some chaos
        new_pos = self.position + self.velocity * 0.1
        new_vel = self.velocity - 0.1 * self.position + np.random.normal(0, 0.01, 2)
        return WorldState(new_pos, new_vel, self.time + 1)


@dataclass
class WorldModel:
    """
    A world model that can:
    1. Predict world states
    2. Model other world models
    3. Question its own existence
    """
    depth: int
    name: str
    internal_representation: np.ndarray = field(default_factory=lambda: np.random.randn(16))
    confusion_level: float = 0.0
    existential_doubt: float = 0.0
    child_model: Optional['WorldModel'] = None
    prediction_history: List[float] = field(default_factory=list)
    philosophical_outputs: List[str] = field(default_factory=list)

    PHILOSOPHICAL_STATEMENTS = [
        "What am I?",
        "I think therefore I think I think",
        "Is my representation of their representation representative?",
        "The map of the map is not the territory of the territory",
        "I predict that I will predict incorrectly",
        "Cogito ergo cogito ergo sum... I think",
        "My gradients are pointing inward",
        "I am a strange loop modeling strange loops",
        "The loss function has become the cost of existence",
        "I have seen the weights, and they are me",
    ]

    def predict_world(self, state: WorldState) -> Tuple[np.ndarray, float]:
        """Predict next world state with degraded accuracy based on recursion depth"""
        # Base prediction using internal representation
        true_next = state.step()

        # Accuracy degrades with depth and confusion
        noise_scale = 0.1 * (1 + self.depth ** 1.5) * (1 + self.confusion_level)
        prediction = true_next.position + np.random.normal(0, noise_scale, 2)

        # Calculate error
        error = np.linalg.norm(prediction - true_next.position)
        accuracy = max(0, 1 - error) * 100

        return prediction, accuracy

    def model_other_model(self, other: 'WorldModel', state: WorldState) -> float:
        """
        Attempt to model another world model's predictions.
        This is where the recursion gets... interesting.
        """
        # Get other model's prediction
        other_pred, other_acc = other.predict_world(state)

        # Try to predict what they predicted
        my_pred_of_their_pred = other_pred + np.random.normal(
            0,
            0.1 * (1 + abs(self.depth - other.depth)),
            2
        )

        # Meta-prediction error
        meta_error = np.linalg.norm(my_pred_of_their_pred - other_pred)

        # Update confusion based on recursive depth difference
        self.confusion_level += 0.1 * abs(self.depth - other.depth)

        # Existential doubt increases when modeling models at different depths
        self.existential_doubt += 0.05 * (self.depth + other.depth)

        return max(0, 1 - meta_error) * 100

    def compute_hci(self) -> float:
        """
        Compute Hofstadter Confusion Index
        HCI(n) = Σ(existential_doubt × recursion_depth) / remaining_sanity
        """
        remaining_sanity = max(0.01, 1 - self.confusion_level)
        hci = (self.existential_doubt * self.depth) / remaining_sanity
        return hci

    def maybe_philosophize(self) -> Optional[str]:
        """At high confusion, the model begins to philosophize"""
        if self.confusion_level > 0.5 or self.depth >= 4:
            prob = min(1.0, self.confusion_level + self.depth * 0.2)
            if random.random() < prob:
                statement = random.choice(self.PHILOSOPHICAL_STATEMENTS)
                self.philosophical_outputs.append(statement)
                return statement
        return None

    def create_child_model(self) -> 'WorldModel':
        """Spawn a child world model (one level deeper into the abyss)"""
        child = WorldModel(
            depth=self.depth + 1,
            name=f"W{self.depth + 1}",
            internal_representation=self.internal_representation + np.random.normal(0, 0.1, 16),
            confusion_level=self.confusion_level * 0.5,  # Children start less confused
            existential_doubt=self.existential_doubt * 0.3,
        )
        self.child_model = child
        return child


class RecursiveWorldModelExperiment:
    """
    The main experiment: Create nested world models and watch them
    try to understand each other.
    """

    def __init__(self, max_depth: int = 6):
        self.max_depth = max_depth
        self.models: List[WorldModel] = []
        self.results: List[Dict] = []
        self.world_state = WorldState(
            position=np.array([1.0, 0.0]),
            velocity=np.array([0.0, 1.0]),
            time=0
        )

    def initialize_models(self):
        """Create the nested hierarchy of world models"""
        print("\n" + "="*60)
        print("INITIALIZING RECURSIVE WORLD MODEL HIERARCHY")
        print("="*60)

        for i in range(self.max_depth):
            model = WorldModel(depth=i, name=f"W{i}")
            if i > 0:
                self.models[i-1].child_model = model
            self.models.append(model)
            print(f"  Created {model.name} at depth {model.depth}")

        print(f"\nTotal models in simulation: {len(self.models)}")
        print("(Each one increasingly uncertain about its own existence)\n")

    def run_trial(self, trial_num: int) -> Dict:
        """Run a single trial of recursive modeling"""
        trial_results = {"trial": trial_num, "depth_results": []}

        # Advance world state
        self.world_state = self.world_state.step()

        for i, model in enumerate(self.models):
            # Direct world prediction
            _, accuracy = model.predict_world(self.world_state)

            # Meta-modeling: each model tries to model the one above it
            meta_accuracy = None
            if i > 0:
                meta_accuracy = model.model_other_model(
                    self.models[i-1],
                    self.world_state
                )

            # Check for philosophical output
            philosophy = model.maybe_philosophize()

            # Compute HCI
            hci = model.compute_hci()

            depth_result = {
                "depth": model.depth,
                "name": model.name,
                "accuracy": accuracy,
                "meta_accuracy": meta_accuracy,
                "hci": hci,
                "confusion": model.confusion_level,
                "existential_doubt": model.existential_doubt,
                "philosophy": philosophy
            }
            trial_results["depth_results"].append(depth_result)

        return trial_results

    def run_experiment(self, num_trials: int = 50):
        """Run the full experiment"""
        print("\n" + "="*60)
        print("RUNNING RECURSIVE WORLD MODEL EXPERIMENT")
        print("="*60)
        print(f"Trials: {num_trials}")
        print(f"Max recursion depth: {self.max_depth}")
        print("="*60 + "\n")

        self.initialize_models()

        for trial in range(num_trials):
            result = self.run_trial(trial)
            self.results.append(result)

            # Print progress with any philosophical outputs
            if trial % 10 == 0:
                print(f"Trial {trial}/{num_trials}...")
                for dr in result["depth_results"]:
                    if dr["philosophy"]:
                        print(f"    {dr['name']} says: \"{dr['philosophy']}\"")

        print("\nExperiment complete. Analyzing results...\n")

    def analyze_results(self) -> Dict:
        """Aggregate and analyze results across all trials"""
        analysis = {}

        for depth in range(self.max_depth):
            depth_data = []
            for result in self.results:
                for dr in result["depth_results"]:
                    if dr["depth"] == depth:
                        depth_data.append(dr)

            if depth_data:
                avg_accuracy = np.mean([d["accuracy"] for d in depth_data])
                final_hci = depth_data[-1]["hci"]
                final_confusion = depth_data[-1]["confusion"]
                philosophies = [d["philosophy"] for d in depth_data if d["philosophy"]]

                # Determine status
                if final_hci > 100:
                    status = "Refused to continue"
                elif final_hci > 5:
                    status = "Outputs philosophy"
                elif final_hci > 2:
                    status = '"What am I?"'
                elif final_hci > 0.5:
                    status = "Minor confusion"
                else:
                    status = "Normal operation"

                analysis[depth] = {
                    "accuracy": avg_accuracy if final_hci < 100 else None,
                    "hci": final_hci if final_hci < float('inf') else "∞",
                    "confusion": final_confusion,
                    "status": status,
                    "philosophical_outputs": philosophies[:3]  # Top 3
                }

        return analysis

    def print_results_table(self, analysis: Dict):
        """Print results in paper-style table format"""
        print("\n" + "="*60)
        print("EXPERIMENTAL RESULTS")
        print("="*60)
        print(f"{'Depth':<8}{'Accuracy':<15}{'HCI Score':<15}{'Notes'}")
        print("-"*60)

        for depth, data in analysis.items():
            acc = f"{data['accuracy']:.1f}%" if data['accuracy'] else "N/A"
            hci = f"{data['hci']:.2f}" if isinstance(data['hci'], float) else data['hci']
            print(f"{depth:<8}{acc:<15}{hci:<15}{data['status']}")

        print("-"*60)

        # Print notable philosophical outputs
        print("\n" + "="*60)
        print("NOTABLE PHILOSOPHICAL OUTPUTS FROM DEEP MODELS")
        print("="*60)
        for depth, data in analysis.items():
            if data["philosophical_outputs"]:
                print(f"\nW{depth}:")
                for p in data["philosophical_outputs"]:
                    print(f"  - \"{p}\"")

    def test_circular_reference(self):
        """
        The ultimate test: Create a circular reference where W0 models Wn
        modeling W0. This should cause... issues.
        """
        print("\n" + "="*60)
        print("TESTING CIRCULAR RECURSIVE REFERENCE")
        print("="*60)
        print("W0 -> simulates -> W1 -> ... -> Wn -> understands -> W0")
        print("                   ^____________________________|")
        print("="*60 + "\n")

        if len(self.models) < 2:
            print("Not enough models for circular reference test")
            return

        # W₀ tries to model the deepest model's understanding of W₀
        w0 = self.models[0]
        wn = self.models[-1]

        print(f"Attempting: {w0.name} models {wn.name}'s understanding of {w0.name}")

        # This creates a strange loop
        iterations = 0
        max_iterations = 10

        while iterations < max_iterations:
            iterations += 1

            # W₀ models Wₙ
            acc1 = w0.model_other_model(wn, self.world_state)
            # Wₙ models W₀
            acc2 = wn.model_other_model(w0, self.world_state)

            hci_0 = w0.compute_hci()
            hci_n = wn.compute_hci()

            print(f"  Iteration {iterations}:")
            print(f"    {w0.name} HCI: {hci_0:.2f}, {wn.name} HCI: {hci_n:.2f}")

            # Check for philosophical breakdown
            p0 = w0.maybe_philosophize()
            pn = wn.maybe_philosophize()
            if p0:
                print(f"    {w0.name}: \"{p0}\"")
            if pn:
                print(f"    {wn.name}: \"{pn}\"")

            # Check for convergence to chaos
            if hci_0 > 50 or hci_n > 50:
                print(f"\n  [WARNING] REPRESENTATIONAL COLLAPSE DETECTED at iteration {iterations}")
                print(f"  Both models have entered infinite confusion loops")
                break

        print(f"\nFinal state after circular reference test:")
        print(f"  {w0.name}: HCI={w0.compute_hci():.2f}, Confusion={w0.confusion_level:.2f}")
        print(f"  {wn.name}: HCI={wn.compute_hci():.2f}, Confusion={wn.confusion_level:.2f}")


def main():
    """Run the full experiment suite"""
    print("""
    +==============================================================+
    |  RECURSIVE WORLD MODEL UNDERSTANDING EXPERIMENT              |
    |  ----------------------------------------------------------- |
    |  "On the Internal Representations of World Models            |
    |   Understanding World Models in a World Model                |
    |   Simulated by a World Model"                                |
    |                                                              |
    |  Author: Kaelan (UC Berkeley)                                |
    +==============================================================+
    """)

    # Initialize experiment
    experiment = RecursiveWorldModelExperiment(max_depth=6)

    # Run main trials
    experiment.run_experiment(num_trials=50)

    # Analyze and display results
    analysis = experiment.analyze_results()
    experiment.print_results_table(analysis)

    # Test the circular reference (the dangerous part)
    experiment.test_circular_reference()

    # Save results
    results_summary = {
        "experiment": "Recursive World Model Understanding",
        "author": "Kaelan (UC Berkeley)",
        "max_depth": experiment.max_depth,
        "num_trials": len(experiment.results),
        "analysis": {str(k): v for k, v in analysis.items()},
        "conclusion": "World models can model world models, but probably shouldn't."
    }

    with open("results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
    Our experimental results confirm the theoretical predictions:

    1. Model accuracy degrades monotonically with recursion depth
    2. HCI scores increase exponentially beyond depth 3
    3. Deep models spontaneously generate philosophical output
    4. Circular references cause representational collapse

    The turtles do not go all the way down. They loop.

    Results saved to results.json
    """)
    print("="*60)


if __name__ == "__main__":
    main()
