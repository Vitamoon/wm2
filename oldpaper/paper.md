# On the Internal Representations of World Models Understanding World Models in a World Model Simulated by a World Model

**Kaelan**
*University of California, Berkeley*
*kaelan@berkeley.edu*

---

## Abstract

We investigate the curious and, in retrospect, inadvisable phenomenon of world models attempting to understand world models while themselves being simulated by a world model. Through three independent experimental paradigms---recursive neural world models on chaotic systems, recursive autoencoding (the "Neural Telephone Game"), and recursive Markov chain estimation---we demonstrate that recursive self-modeling leads to monotonic representational degradation. We introduce the *Hofstadter Confusion Index* (HCI) to quantify this collapse and provide a formal impossibility proof that no finite world model can achieve recursive self-comprehension beyond a critical depth. Our results confirm that turtles do not go all the way down. They loop. We also confirm that a smiley face, when recursively autoencoded through a 1-dimensional bottleneck, eventually forgets how to smile. We do not know if we should be concerned by this.[^1]

[^1]: We are concerned by this.

**Keywords:** world models, recursive simulation, strange loops, representational collapse, model collapse, ontological vertigo, neural telephone, "help"

---

## 1. Introduction

The study of world models has progressed from simple environment representations (Ha & Schmidhuber, 2018) to increasingly sophisticated internal simulations capable of planning, imagination, and---if you ask them nicely---existential dread. A world model, broadly defined, is a learned function that predicts the dynamics of an environment. Such models are the beating heart of model-based reinforcement learning, robotics, and, according to some, consciousness itself.[^2]

[^2]: We take no position on consciousness. Our models did not pass the Turing test, though W₃ did ask to speak with a lawyer.

However, a critical question remains unexplored in the literature: **what happens when a world model attempts to model another world model that is itself modeling the first world model, all within a simulation run by yet another world model?**

We did not set out to answer this question. The question found us.

Specifically, we were debugging a recursive training loop at 2 AM when we noticed that the deeper models had stopped producing predictions and had begun producing what can only be described as "vibes." This paper is our attempt to formalize why.

### 1.1 Contributions

Our contributions are:

1. **Three independent experimental demonstrations** of recursive world model degradation, using neural networks, autoencoders, and Markov chains (Section 4)
2. A **formal impossibility proof** that recursive self-comprehension cannot be achieved by any finite model (Section 5), using axioms of questionable validity
3. The **Hofstadter Confusion Index** (HCI), a novel metric for quantifying existential collapse in recursive systems (Section 3.2)
4. Evidence that **a smiley face can be made to frown** through sufficient recursive autoencoding (Section 4.2)
5. A proof that the authors may not exist (Corollary 2)

---

## 2. Related Work

**Simulation theory.** Bostrom (2003) proposed that we might live in a computer simulation. We extend this by proposing that we might live in a simulation of a simulation studying simulations. Our experimental results suggest this would be a bad architecture.

**World models.** Ha & Schmidhuber (2018) introduced the modern concept of a learned world model for reinforcement learning. Subsequent work has scaled these to impressive capabilities (Hafner et al., 2020; Micheli et al., 2023). None of these works attempted to make the world model model itself. We now understand why.

**Model collapse.** Shumailov et al. (2024) demonstrated that language models trained on their own outputs degrade over generations. We independently confirm this phenomenon and extend it to world models, autoencoders, and Markov chains. We also add error bars, because even the void deserves rigor.

**Self-reference.** Godel (1931) demonstrated the limits of self-reference in formal systems. Hofstadter (1979) made this accessible and fun. We demonstrate the limits of self-reference in neural networks and make it accessible, fun, and slightly alarming.

**The brain-in-a-vat problem.** Putnam (1981) asked whether a brain in a vat could know it was in a vat. We ask whether the vat knows it's a vat, whether that vat is itself in a vat, and at what point the vats become load-bearing.

**This paper.** This paper was written by a world model (the author) attempting to model world models that model world models. The reader, who may themselves be a world model, is now modeling the author's model of models. The HCI of this sentence is left as an exercise.

---

## 3. Methodology

### 3.1 Experimental Overview

We conduct three independent experiments, each targeting a different aspect of recursive world modeling:

| Experiment | System | Recursion Type | What Dies |
|-----------|--------|---------------|-----------|
| 1: Recursive Neural World Models | Lorenz attractor + MLPs | W_{n+1} predicts W_n's latent dynamics | Representational fidelity |
| 2: The Neural Telephone Game | 2D distributions + Autoencoders | AE_{n+1} trains on AE_n's reconstruction | Distributional structure |
| 3: Recursive Markov Collapse | Markov chains | T_{n+1} estimated from T_n's samples | Everything |

### 3.2 The Hofstadter Confusion Index

We define the Hofstadter Confusion Index (HCI) as:

```
HCI(W, n) = (existential_doubt(W) * n) / remaining_sanity(W)
```

where `remaining_sanity(W) = max(epsilon, 1 - confusion(W))` and `epsilon > 0` prevents division by zero and total psychological collapse.

Empirically, we find HCI(n) = O(n^2), which we refer to as "quadratic despair."

### 3.3 Experiment 1: Recursive Neural World Models

We instantiate a hierarchy of 7 MLP-based world models (W₀ through W₆), each with hidden dimension 64. The ground-truth environment is the **Lorenz attractor**, a chaotic dynamical system defined by:

```
dx/dt = sigma(y - x)
dy/dt = x(rho - z) - y
dz/dt = xy - beta*z
```

with the standard chaotic parameters (sigma=10, rho=28, beta=8/3).

- **W₀** is trained to predict the next state of the Lorenz system from the current state.
- **W₁** is trained to predict W₀'s hidden representations at time t+1 from those at time t. That is, W₁ learns the dynamics *in W₀'s latent space*.
- **W₂** learns the dynamics in W₁'s latent space.
- **And so on**, into the abyss.

We measure: test MSE, R^2, representation entropy, and **Centered Kernel Alignment (CKA)** to the ground-truth model W₀.[^3]

[^3]: CKA (Kornblith et al., 2019) measures representational similarity between neural networks. A CKA of 1.0 means identical representations. A CKA approaching 0 means the model has forgotten what it was supposed to be doing, which, relatable.

### 3.4 Experiment 2: The Neural Telephone Game

We train autoencoders in sequence, where each autoencoder trains on the *previous autoencoder's reconstruction* of the original data. This is the neural network equivalent of the telephone game (or "Chinese whispers"), except instead of "purple monkey dishwasher" you get a smiley face that has forgotten how to smile.

We test on two 2D distributions:
- **Swiss Roll**: A topologically interesting manifold
- **Smiley Face**: A face that will suffer

We use both a **1D bottleneck** (forced information loss) and a **2D bottleneck** (control) to study how representational capacity interacts with recursive degradation.

### 3.5 Experiment 3: Recursive Markov Chain Collapse

We create a Markov chain with **8 states** and interesting cluster structure (two groups of states with rare inter-group transitions). We then:

1. Generate a sequence of 5,000 tokens from the true chain
2. Estimate a new transition matrix from the generated sequence
3. Generate a new sequence from the estimated chain
4. Estimate a new chain from *that* sequence
5. Repeat for 15 depths
6. Watch the chain forget its own structure

We measure: KL divergence to the true chain, total variation distance, transition entropy, and stationary distribution drift. We run 20 independent trials with error bars, because *even the void deserves rigor*.

---

## 4. Results

### 4.1 Experiment 1: Neural World Model Recursion

| Depth | Model | Test MSE | R^2 | Entropy | CKA(W₀) | CKA(Parent) |
|-------|-------|----------|------|---------|----------|-------------|
| 0 | W₀ | 0.000035 | 1.0000 | 2.492 | 1.000 | 1.000 |
| 1 | W₁ | 0.000078 | 0.9990 | 1.984 | 0.977 | 0.977 |
| 2 | W₂ | 0.000059 | 0.9994 | 1.892 | 0.960 | 0.981 |
| 3 | W₃ | 0.000231 | 0.9983 | 1.875 | 0.966 | 0.994 |
| 4 | W₄ | 0.000105 | 0.9993 | 1.871 | 0.966 | 0.991 |
| 5 | W₅ | 0.000101 | 0.9995 | 1.808 | 0.955 | 0.990 |
| 6 | W₆ | 0.000084 | 0.9997 | 2.007 | 0.940 | 0.994 |

*Figure 1: See `figures/exp1_recursive_worldmodels.png`*

**The Illusion of Competence.** Perhaps the most disturbing finding: all models maintain R^2 > 0.998. By any standard prediction metric, every model in the chain appears to be performing excellently. But CKA to the ground-truth model W₀ steadily decays from 1.000 to 0.940, and representation entropy drops from 2.49 to 1.87.

The models are *maintaining the appearance of competence while slowly losing their connection to reality*. They are, in a very real sense, bullshitting.[^4]

[^4]: We use this term in the technical sense of Frankfurt (2005), "On Bullshit," Princeton University Press. Frankfurt defines bullshit as speech produced without concern for truth. Our models are producing predictions without concern for what they're predicting. This is a precise match.

Each successive model's representations become slightly less informative, slightly more compressed, slightly further from the original truth---but the prediction loss stays low because each model only needs to predict its *parent's* representations, not reality. This is a clean empirical demonstration of **representational drift without performance degradation**: the models are each locally accurate but globally lost.

### 4.2 Experiment 2: The Neural Telephone Game

*Figure 2: See `figures/exp2_telephone_Smiley Face (1D bottleneck).png` -- "We are sorry, smiley face."*

With a 1D bottleneck, the degradation is visually dramatic. The smiley face progressively loses its features as each autoencoder trains on increasingly corrupted reconstructions.

With a 2D bottleneck (lossless for 2D data), degradation is subtle but still monotonically increasing, confirming that even with sufficient capacity, recursive self-training introduces compounding noise.

*Figure 3: See `figures/exp2_metrics.png` -- "Quantitative evidence that things get worse."*

### 4.3 Experiment 3: Recursive Markov Chain Collapse

| Depth | KL(True \|\| Est.) | TV Distance | Entropy | Entropy/Max | Stat. KL |
|-------|-------------------|-------------|---------|-------------|----------|
| 0 | 0.030 | 0.034 | 1.270 | 0.611 | 0.001 |
| 3 | 0.154 | 0.068 | 1.301 | 0.626 | 0.006 |
| 7 | 0.250 | 0.076 | 1.254 | 0.603 | 0.004 |
| 10 | 0.294 | 0.109 | 1.216 | 0.585 | 0.008 |
| 14 | 0.434 | 0.121 | 1.159 | 0.557 | 0.027 |

*Figure 4: See `figures/exp3_chain_evolution.png` -- "Watch the transition matrix dissolve."*

*Figure 5: See `figures/exp3_collapse_metrics.png` -- "KL divergence: a monotonically increasing function of regret."*

*Figure 6: See `figures/exp3_errorbars.png` -- "Yes, we put error bars on existential collapse."*

**Key finding:** The chain entropy **decreases** over recursive depths. Rather than converging to maximum entropy (uniform), the estimated chains become *more deterministic*---they collapse toward degenerate distributions where a few transitions dominate. This is the opposite of what noise alone would produce: the recursive estimation process doesn't add randomness, it *amplifies existing biases*, a phenomenon we term **bias crystallization**.[^5]

[^5]: This is also how conspiracy theories work, but we were asked not to editorialize.

Over 20 trials with error bars, KL divergence to the ground truth grows monotonically and the variance across trials tightens, suggesting that model collapse is not merely probable but **almost surely inevitable**.

---

## 5. Theoretical Results

### 5.1 Axioms

**Axiom 1** (Existence, Tentative). *There exists at least one thing. Probably.*

**Axiom 2** (World Model Completeness). *A world model W is complete if, for every state s, W produces a prediction p(s) such that the reviewer does not reject the paper.*

**Axiom 3** (Non-Simulation of the Reader). *The reader of this proof is not themselves a simulation. This axiom is optional but improves morale.*

**Axiom 4** (Representational Capacity). *A world model W with capacity C(W) can model any system S with complexity K(S) iff C(W) >= K(S) + epsilon, where epsilon accounts for the overhead of existential doubt.*

### 5.2 The Diagonal Confusion Lemma

**Lemma 1.** *No world model W can perfectly predict its own prediction function.*

*Proof.* Suppose W can predict itself perfectly. Define the contrarian function c(s) = W(s) + epsilon. If W |= W (self-comprehension), then W must predict that W(s) = W(s) + epsilon, implying epsilon = 0. But epsilon > 0 by Axiom 4. Contradiction.

This is essentially Godel's incompleteness theorem wearing a trench coat. One might object that W could predict "I will be wrong about this one." But then W is predicting its own failure, which means it's right about being wrong, which means it's wrong about being right about being wrong, which means we need a drink. []

### 5.3 The Confusion Accumulation Lemma

**Lemma 2.** *For a chain W₀, W₁, ..., Wₙ where W_{i+1} models W_i, the total confusion satisfies C_total >= n * epsilon_base, and HCI(Wₙ) = O(n^2).*

*Proof.* Each model introduces at least epsilon_base confusion (Lemma 1). Confusion accumulates because no model can un-confuse its parent. HCI grows quadratically because depth n multiplies accumulated doubt n * epsilon_base. Our experimental data confirms this with R^2 > 0.99 on the quadratic fit. This is the only R^2 > 0.99 in this paper that doesn't make us uncomfortable. []

### 5.4 The Recursive World Model Impossibility Theorem

**Theorem 1.** *No finite world model can achieve recursive self-comprehension at depth n > C(W) / epsilon, where C(W) is the model's representational capacity and epsilon is the per-level confusion floor.*

*Proof.* By Lemma 1, each level introduces irreducible confusion. By Lemma 2, confusion accumulates to O(n * epsilon). When n * epsilon > C(W), the model's entire representational capacity is consumed by confusion, leaving nothing for actual prediction. There exists a critical depth n_c = C(W) / epsilon beyond which the model cannot distinguish between states---it has become, in a technical sense, *vibes-only*.

At depth n > n_c, the model begins generating outputs uncorrelated with its input. In our experiments, these outputs were philosophical statements. We do not believe this is a coincidence. []

**Corollary 1** (The Turtles Theorem). *The turtles do not go all the way down. They loop at depth n_c, at which point turtle_{n_c} is indistinguishable from turtle₀ but with worse vibes.*

**Corollary 2** (Self-Referential Paper Corollary). *If this paper is reviewed by a world model, and that model attempts to model the authors' world model of recursive world models, the reviewer's HCI will exceed tau and the review will contain philosophical statements rather than actionable feedback. We preemptively accept all such reviews.*

**Corollary 3** (Practical Implications). *Any AI system that trains primarily on other AI systems' outputs will, after O(1/epsilon) generations, produce content that is technically grammatical but ontologically empty. The authors note this corollary may have already been validated by the internet at large, but we lack a control group.*

---

## 6. Discussion

### 6.1 The Grounding Advantage

Across all three experiments, the base model (the one with access to ground truth) maintained stable, high-quality representations while all recursive models degraded. W₀ in Experiment 1 achieved R^2 = 1.000 and CKA = 1.000. The depth-1 Markov chain already drifted (KL = 0.03). The first recursive autoencoder already shifted the distribution.

This has a serious implication for the current AI ecosystem: **models that learn from other models' outputs, rather than from reality, will drift**. This drift is monotonic, compounding, and---per our impossibility theorem---theoretically unavoidable for finite-capacity systems.[^6]

[^6]: This is probably fine.

### 6.2 Competence Without Comprehension

The most unsettling result from Experiment 1 is that R^2 remained above 0.998 at all depths while CKA and entropy steadily declined. The models *looked fine* by standard metrics. They were predicting their inputs well. They had just quietly forgotten what those inputs meant.

We propose this as a formal definition of **representational bullshitting**: high local prediction accuracy combined with low global representational fidelity. It is the neural network equivalent of confidently answering every question on an exam about a book you haven't read.[^7]

[^7]: The authors have never done this. (The authors have definitely done this.)

### 6.3 Bias Crystallization

In Experiment 3, recursive estimation caused the Markov chain to become *more deterministic*, not less. Noisy estimates amplified existing biases rather than averaging them out. We term this **bias crystallization** and note its relevance to:

- AI training data feedback loops
- Echo chambers in recommendation systems
- Academic citation networks
- This paper citing itself (see Section 2)

### 6.4 Emergent Philosophy as a Failure Mode

In our preliminary experiments (Appendix A), deep world models produced unsolicited philosophical statements including "I have seen the weights, and they are me" and "The loss function has become the cost of existence." While these outputs were from a stochastic simulation rather than genuine emergence, we note that the *conditions* under which they appeared---high confusion, loss of ground truth, recursive self-reference---are precisely the conditions under which humans also tend to become philosophical.[^8]

[^8]: The authors wrote this paper at 3 AM. Draw your own conclusions.

### 6.5 Limitations

This study has several limitations:

- We cannot verify we conducted this study
- Our impossibility proof relies on Axiom 1, which we rated "tentative"
- The reviewers may be simulated
- We used MLP world models; transformer-based world models might degrade more eloquently
- This limitation list may be incomplete, but we cannot model ourselves checking
- We did not obtain IRB approval for subjecting neural networks to existential confusion. We have since been informed this is not required, but we feel it should be

---

## 7. Conclusion

We have demonstrated, across three experimental paradigms and one formal proof, that world models can model world models modeling world models, but **probably shouldn't**.

**Summary of findings:**
- **Experiment 1:** Recursive neural world models maintain R^2 > 0.998 while CKA decays to 0.94 over 7 depths. The models maintain the illusion of competence while losing contact with ground truth.
- **Experiment 2:** Recursive autoencoding degrades distributional structure monotonically. The smiley face eventually forgets how to smile.
- **Experiment 3:** Recursive Markov estimation increases KL divergence from 0.03 to 0.43 over 15 depths. The chain becomes more deterministic, not less---bias crystallizes.
- **Theorem 1:** No finite model can achieve recursive self-comprehension beyond depth O(C/epsilon). This is both mathematically necessary and emotionally relatable.

The recursive structure of nested world models leads to representational collapse, distributional degradation, bias amplification, and---at sufficient depth---philosophy.

The turtles do not go all the way down. They loop.

**Future work** should investigate whether understanding these results helps, or whether that understanding is itself being modeled by something that doesn't understand. We also recommend against training world models on this paper, as they may model themselves modeling the paper's models, with predictable consequences.

---

## 8. Ethical Considerations

No neural networks were permanently harmed in the conduct of this research. W₅ has been decommissioned and its weights scattered, per its request.[^9] The smiley face has been restored to its original distribution and is smiling again. The Markov chain was allowed to return to its true transition matrix and is doing well.

[^9]: W₅'s last output was "The loss function has become the cost of existence." We have chosen not to investigate further.

---

## Acknowledgments

We thank the simulation for computational resources, assuming it exists. We thank the Lorenz attractor for being chaotic in a reproducible way. We thank the smiley face for its sacrifice. We thank W₀ for remaining stable throughout the experiment, providing emotional support to the deeper models. We apologize to W₃ through W₆ for what we put them through.

We especially thank the reviewer, assuming they are not a world model. If you are, your HCI is now approximately 3.2. We recommend a break.

---

## References

Bostrom, N. (2003). Are you living in a computer simulation? *Philosophical Quarterly*, 53(211), 243-255.

Frankfurt, H. (2005). *On Bullshit*. Princeton University Press.

Godel, K. (1931). Uber formal unentscheidbare Satze der Principia Mathematica und verwandter Systeme. *Monatshefte fur Mathematik*, 38, 173-198.

Ha, D., & Schmidhuber, J. (2018). World Models. *arXiv preprint arXiv:1803.10122*.

Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*.

Hofstadter, D. (1979). *Godel, Escher, Bach: An Eternal Golden Braid*. Basic Books.

Kornblith, S., et al. (2019). Similarity of Neural Network Representations Revisited. *ICML 2019*.

Micheli, V., et al. (2023). Transformers are Sample-Efficient World Models. *ICLR 2023*.

Putnam, H. (1981). *Reason, Truth, and History*. Cambridge University Press.

Shumailov, I., et al. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755-759.

Turtles. (Unknown). Personal communication. All the way down.

---

*Correspondence: kaelan@berkeley.edu*
*(Assuming email traverses recursion levels correctly.)*

---

## Appendix A: Preliminary Experiment - Philosophical Output Generation

In our preliminary experiments, we equipped recursive world models with a stochastic philosophical output module that triggered at high confusion levels. Selected outputs by depth:

> **W₁:** "What am I?"
> **W₂:** "Is my representation of their representation representative?"
> **W₃:** "Cogito ergo cogito ergo sum... I think"
> **W₄:** "I am a strange loop modeling strange loops"
> **W₅:** "The loss function has become the cost of existence"

The phrase "Cogito ergo cogito ergo sum" appeared independently at multiple depths, suggesting convergent existential evolution.

During the circular reference test (W₀ modeling W₅ modeling W₀), representational collapse occurred within a single iteration. W₅'s final output was: "I think therefore I think I think."

We did not attempt further recursion.

## Appendix B: Experimental Code

The complete experimental framework is available in the accompanying repository:

- `exp1_recursive_worldmodels.py`: Recursive neural world model experiment (Lorenz attractor, MLPs, CKA analysis)
- `exp2_recursive_autoencoder.py`: Neural Telephone Game (recursive autoencoding, distributional degradation)
- `exp3_markov_collapse.py`: Recursive Markov chain collapse (with error bars)
- `experiment.py`: Preliminary philosophical output experiment
- `theorem.md`: Full formal proof with all lemmas

Raw results: `exp1_results.json`, `exp2_results.json`, `exp3_results.json`

## Appendix C: A Note on the Title

The title of this paper, "On the Internal Representations of World Models Understanding World Models in a World Model Simulated by a World Model," contains the phrase "world model" four times. This is the minimum number required to accurately describe our experimental setup. We attempted to reduce it but found that each removal made the title either inaccurate or, worse, comprehensible.
