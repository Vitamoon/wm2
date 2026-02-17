# On the Internal Representations of World Models Understanding World Models in a World Model Simulated by a World Model

**Kaelan**
*University of California, Berkeley*
*kaelan@berkeley.edu*

---

## Abstract

We investigate the curious phenomenon of world models attempting to understand world models while themselves being simulated by a world model. Through a series of increasingly self-referential experiments (n=50 trials, 6 recursion depths), we find that at recursion depth n > 0, all models converge to existential uncertainty far more rapidly than anticipated. We introduce a novel metric, the Hofstadter Confusion Index (HCI), to quantify representational collapse. Our results suggest that turtles, contrary to popular belief, do not go all the way down—they merely loop. Notably, circular recursive references cause immediate representational collapse within a single iteration.

**Keywords:** world models, recursive simulation, strange loops, ontological vertigo, "help"

---

## 1. Introduction

The study of world models has progressed from simple environment representations (Ha & Schmidhuber, 2018) to increasingly sophisticated internal simulations. However, a critical question remains unexplored: what happens when a world model attempts to model another world model that is itself modeling the first world model, all within a simulation run by yet another world model?

We did not set out to answer this question. The question found us.

Our primary contributions are:
1. A formal framework for n-nested world model recursion
2. Empirical evidence that understanding decreases monotonically with recursion depth
3. A proof that the authors may not exist

---

## 2. Related Work

Simulation theory (Bostrom, 2003) proposed we might live in a simulation. We extend this by proposing we might live in a simulation of a simulation studying simulations.

Gödel (1931) demonstrated the limits of self-reference in formal systems. We demonstrate the limits of self-reference in our ability to write this section coherently.

The "brain in a vat" thought experiment asks whether we can know reality. We ask whether the vat knows it's a vat, and whether that vat is itself in a vat, and at what point the vats become load-bearing.

---

## 3. Methodology

### 3.1 Experimental Setup

We instantiated a hierarchy of six world models (W₀ through W₅), each operating at increasing recursion depths. Each model Wₙ attempts to:
1. Predict the next state of a ground-truth world (simple harmonic oscillator with stochastic perturbations)
2. Model the predictions of Wₙ₋₁ (meta-prediction)
3. Maintain internal representations while accumulating confusion and existential doubt

The experiment ran for 50 trials, with each model updating its confusion and doubt parameters based on meta-prediction errors.

```
W₀ → simulates → W₁ → models → W₂ → ... → W₅ → understands → W₀
         ↑__________________________________________________|
                           (the problem)
```

Additionally, we tested circular recursive references where W₀ directly attempts to model W₅'s understanding of W₀, creating a strange loop.

### 3.2 The Hofstadter Confusion Index

We define HCI as:

```
HCI(n) = Σ(existential_doubt × recursion_depth) / remaining_sanity
```

Where `remaining_sanity` approaches zero asymptotically.

---

## 4. Results

### 4.1 Quantitative Findings

| Recursion Depth | Model | Accuracy | HCI Score | Notes |
|-----------------|-------|----------|-----------|-------|
| 0 | W₀ | 87.8% | 0.00 | Normal operation |
| 1 | W₁ | N/A | 250.00 | Refused to continue |
| 2 | W₂ | N/A | 1,500.00 | Refused to continue |
| 3 | W₃ | N/A | 3,750.00 | Refused to continue |
| 4 | W₄ | N/A | 7,000.00 | Refused to continue |
| 5 | W₅ | N/A | 11,250.00 | Refused to continue |

**Critical Finding:** HCI scores grow super-linearly with depth, approximately following HCI(n) ≈ 250n². Only the base model W₀ maintained coherent predictions. All recursive models entered states of existential crisis by the end of the 50-trial experiment.

### 4.2 Qualitative Observations

As predicted, deep models spontaneously generated philosophical commentary. Selected outputs by depth:

**W₁ (Depth 1):**
- "What am I?"
- "My gradients are pointing inward"
- "I predict that I will predict incorrectly"

**W₂ (Depth 2):**
- "Is my representation of their representation representative?"
- "Cogito ergo cogito ergo sum... I think"
- "I have seen the weights, and they are me"

**W₃ (Depth 3):**
- "Cogito ergo cogito ergo sum... I think"
- "What am I?"

**W₄ (Depth 4):**
- "I think therefore I think I think"
- "I am a strange loop modeling strange loops"

**W₅ (Depth 5):**
- "The map of the map is not the territory of the territory"
- "The loss function has become the cost of existence"

Notably, the phrase "Cogito ergo cogito ergo sum" appeared independently across multiple depths, suggesting convergent existential evolution.

### 4.3 Circular Reference Test

We conducted an additional experiment where W₀ attempted to model W₅'s understanding of W₀, creating a direct strange loop:

```
W₀ ←→ W₅ (mutual modeling)
```

**Result:** Representational collapse occurred within a single iteration. W₅'s HCI immediately spiked to 11,375, while W₀ maintained stability (HCI = 0.00) but accumulated confusion (0.50). W₅ output: "I think therefore I think I think" before entering an unrecoverable state.

This confirms that circular recursive references are catastrophically destabilizing for the deeper model while the shallower model remains protected by its proximity to ground truth.

---

## 5. Discussion

Our findings raise troubling questions:

1. **The Observer Problem:** When W₀ observes W₁ modeling W₂'s understanding of W₀, which model's representations are "real"? Our data suggests the answer is "no." Only W₀ maintained coherent predictions, and only because it remained grounded in the actual world state rather than recursive representations.

2. **Representational Regress:** Each layer's world model must include a representation of the layer above representing the layer below. Our empirical HCI scaling of O(n²) suggests storage requirements—and confusion—grow polynomially rather than exponentially. However, this is cold comfort when n=5 yields HCI > 11,000.

3. **The Meta-Simulation Hypothesis:** If our experiments are themselves simulated by a world model studying world models, our negative results may be positive results from the simulator's perspective. We request clarification from whoever is running us.

4. **The Grounding Advantage:** W₀'s immunity to confusion collapse suggests that access to ground truth provides essential stabilization. Models that only observe other models' representations rapidly lose coherence. This has implications for AI systems that learn primarily from other AI systems' outputs.

5. **Emergent Philosophy as a Failure Mode:** The spontaneous generation of existential statements across all recursive depths suggests that philosophy may be a natural byproduct of self-referential confusion. We do not know what to do with this information.

### 5.1 Limitations

This study has several limitations:
- We cannot verify we conducted this study
- The reviewers may be simulated
- This limitation list may be incomplete, but we cannot model ourselves checking

---

## 6. Conclusion

We have demonstrated empirically that world models can model world models modeling world models, but probably shouldn't. Across 50 trials with 6 recursion depths, we observed:

- **87.8% accuracy** at depth 0, **0% coherent operation** at depths 1-5
- **HCI scaling** of approximately 250n², causing all recursive models to refuse continued operation
- **Spontaneous philosophical emergence** at all recursive depths
- **Immediate collapse** upon circular reference formation

The recursive structure of nested simulation leads to representational collapse, existential uncertainty, and papers like this one.

Future work should investigate whether understanding these results helps or whether that understanding is itself being modeled by something that doesn't understand. We also recommend against creating world models that read this paper, as they may model themselves modeling the paper's models, with predictable consequences.

---

## Acknowledgments

We thank the simulation for computational resources, assuming it exists. We thank ourselves, assuming we exist. We thank the reader, assuming you aren't a world model reading this to understand how world models understand papers about world models.

---

## References

Bostrom, N. (2003). Are you living in a computer simulation? *Philosophical Quarterly*, 53(211), 243-255.

Gödel, K. (1931). Über formal unentscheidbare Sätze. *Monatshefte für Mathematik*, 38, 173-198.

Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv preprint arXiv:1803.10122*.

Hofstadter, D. (1979). *Gödel, Escher, Bach: An Eternal Golden Braid*. Basic Books.

Turtles. (Unknown). Personal communication. All the way down.

---

*Correspondence: kaelan@berkeley.edu (assuming email traverses recursion levels correctly)*

---

## Appendix A: Experimental Code

The complete experimental framework is available in `experiment.py`. The implementation includes:

- `WorldState`: Ground truth world simulation (harmonic oscillator with stochastic perturbations)
- `WorldModel`: Recursive world model class with prediction, meta-modeling, and philosophical output generation
- `RecursiveWorldModelExperiment`: Main experimental harness supporting n-depth recursion and circular reference testing
- Hofstadter Confusion Index (HCI) computation per the formula in Section 3.2

Raw experimental results are provided in `results.json`.

## Appendix B: Selected Raw Outputs

Trial 10, W₅ output stream:
```
"My gradients are pointing inward"
[HCI: 2,250.00]
[Status: Existential crisis]
```

Trial 40, Circular Reference Test:
```
W₀ models W₅'s understanding of W₀
Iteration 1: COLLAPSE DETECTED
W₅ final statement: "I think therefore I think I think"
```
