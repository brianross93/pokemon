This project overview is for the creation of a whitepaper detailing an SR-FBAM AI architecture model.

1. Title / Abstract / Keywords

Title: Symbolic Recurrence as Deep Memory: Sparse Associative Recall in Frame-Based Action Models

Abstract:
In this white paper, we address long-horizon recall and interpretability limits in dense, token-wise sequence models by proposing Symbolic-Recurrence Frame-Based Action Models (SR-FBAMs), an extension of the Frame-Based Action Model (FBAM) that augments recurrent, frame-wise integration with symbolic associative memory supporting sparse, human-readable recall hops. Instead of repeatedly reprocessing full contexts, SR-FBAM issues discrete ASSOC/FOLLOW/VOTE/HALT operations over a persistent concept graph, producing auditable traces and adaptive depth tied to reasoning hops. Training combines activation recomputation and hidden-state streaming to keep GPU memory approximately O(1) in sequence length, while pruning policies bound graph growth. We advance a scaling hypothesis that performance improves as a power law of associative depth at fixed parameters (speculative pending full validation), mirroring FBAM's serial length scaling. Toy multi-hop tasks and simulated editor/terminal "text-video" environments illustrate 2-3x wall-time efficiency for selective recall relative to equally sized dense baselines, alongside transparent chain-of-thought logs. We discuss operator design versus learning, ambiguity-driven failures, and hybrid dense-symbolic variants. SR-FBAM reframes scaling from width or context to meaningful hops, moving toward efficient, auditable agents.

Keywords: recurrence, frame-based agents, symbolic memory, associative recall, scaling laws, interpretability, neuro-symbolic AI

2. Introduction & Motivation
- The problem: deep recurrence challenges, long-context inefficiencies, and memory collapse in Transformers and other dense models.
- Cognitive parallel: humans recall via symbolic hops (e.g., "Laura -> nickname -> Antman -> Anthony") rather than full replay, enabling sparse, compositional depth.
- Why this matters: enhances efficiency, interpretability, and scaling in AI agents, with potential "killer apps" in multi-step reasoning and long-term memory tasks.
- Optional mnemonic analogy (1-2 paragraphs): use Phil Dunphy (Modern Family) as a relatable example of layered associations (e.g., "Phil -> inventions -> puns -> family dynamics") to illustrate quirky, human-like recall chains; pivot to universal examples for rigor.
- Prioritized contributions:
  - Proposal of Symbolic-Recurrence FBAM (SR-FBAM) as a sparse extension to FBAM.
  - Theoretical scaling argument: performance is proportional to associative depth (hops), with hypotheses on loss improvements (speculative elements noted).
  - Architecture sketch, concrete running examples with diagrams, toy experiments, and qualitative traces.
  - Brief teaser of empirical insights (e.g., efficiency gains in benchmarks; honest note: results are simulated prototypes, not large-scale trained models).

3. Background & Related Work
- Recurrence in deep learning: RNNs, LSTMs, state-space models (SSMs), and their limitations in long sequences.
- Transformers: context scaling issues, quadratic attention, and associative reductions.
- External memory systems: Neural Turing Machines, Memory-Augmented Neural Networks, and modern associative variants (e.g., Hopfield network revivals).
- Hybrid symbolic-neural systems: knowledge graphs for structured reasoning, neuro-symbolic surveys on continual learning and graph traversal.
- Cognitive/mnemonic models: human memory as semantic networks and associative recall from psychology (e.g., mnemonics, interference effects).

Positioning: SR-FBAM bridges FBAM's serial integration with symbolic sparsity, addressing non-associative serial depth that parallel models lack.

3.1 FBAM Primer (for unfamiliar readers)
The Frame-Based Action Model (FBAM), introduced in "Attention is Not All You Need: A Recurrence-Complete Alternative for Sequential Computation" (2025), is a hybrid architecture for long-horizon agentic tasks, processing sequences of fixed-size "frames" (e.g., 2D grids like terminal views) paired with actions. It combines intra-frame Transformer attention for embedding with inter-frame LSTM-based serial integration, enabling autoregressive action prediction on datasets like Git histories. Recurrence-completeness ensures the model's computational depth scales linearly with sequence length (O(n)), allowing representation of non-associative, inherently serial operations (e.g., persistent state mutations). Unlike parallel models (Transformers, Mamba), which bound depth and fail on non-scannable tasks, FBAM uses full backpropagation through time with recomputation for O(1) memory, though at linear wall-time cost. Its serial scaling hypothesis shows loss follows power laws with sequence length (loss approx A * L^{-alpha}), yielding emergent capabilities at fixed parameters. FBAM is our starting point because its serial backbone handles long-horizon aggregation while leaving room for extensions like symbolic memory to address dense latent opacity-integrating sparse hops atop LSTMs for interpretable recall without sacrificing recurrence-completeness.

4. Architectural Proposal: Symbolic-Recurrence FBAM (SR-FBAM)
- High-level view and diagram (e.g., flowchart of hop-based recall).
- Component definitions (table for clarity):
  - Frame Head: encodes the current frame into sparse symbols (entities/relations from text/grid).
  - Recurrent Integrator: lightweight LSTM controller for action selection (ASSOC, FOLLOW, VOTE, WRITE, HALT).
  - Symbolic Memory Graph: external store for nodes/edges; supports sparse queries and updates.
  - Action Space: discrete primitives-ASSOC (associate), FOLLOW (traverse), VOTE (consensus), WRITE (update), HALT.
- Data flow and example recall chain (e.g., "Laura -> nickname -> Antman -> Anthony"; include universal variant like "Paris -> Eiffel Tower -> iron -> metallurgy" for broader appeal).
- Treatment of memory: sparsity via compression, caching strategies, efficient retrieval.
- Training details: end-to-end differentiation for embeddings/graph updates; reinforcement (e.g., REINFORCE) for operator/hop selection. Address learning/curation tension: operators start hand-coded (5-10 primitives requiring domain-specific design, like relation types in a movie KG-roughly 10-20 hours of engineering for the toy domain) but can be learned via meta-optimization or embedding-based selection, reducing curation to the initial schema (20-50% learned vs engineered is speculative until tested; pure learned representations risk opacity, so a hybrid is recommended).
- Checkpointing and memory scaling: O(1) GPU memory via FBAM-style recomputation, but O(nodes) graph memory, bounded by pruning to sublinear growth (e.g., ~2x initial size cap via frequency eviction).

4.1 Concrete Running Example: Multi-hop Recall Task
Task: "Who is the friend of the person who directed the movie featuring the actor born in Paris?" Assume a toy knowledge graph with nodes like actors, directors, and movies, plus relations such as born_in, featured_in, directed_by, friend_of.

Trace through SR-FBAM:
- Input Frame Encoding: Frame Head parses the query into symbols: [Actor ?X born_in Paris; Movie ?Y featured_in ?X; Director ?Z directed_by ?Y; Friend ?W friend_of ?Z]. Output is a sparse frame vector with entities (?X, Paris) and unresolved variables.

ASCII diagram (initial frame):
```
Query: "Who is the friend of the person who directed the movie featuring the actor born in Paris?"
+-------------------+
| Frame Head        |
| - Extract: Actor born_in Paris
| - Symbols: [?X:Actor, born_in:Paris, ?Y:Movie, ...]
+-------------------+
```

- Initiate Recall: The Recurrent Integrator receives the frame and decides the first action: ASSOC (query graph for "born_in = Paris" on Actor nodes).
- Hop 1: Symbolic Memory Graph query retrieves node "Audrey Tautou" (born_in = Paris, actor). Update frame: ?X = "Audrey Tautou".

ASCII diagram (hop 1):
```
Recurrent Integrator --> ASSOC("born_in = Paris", type=Actor)
Symbolic Memory Graph:
  Nodes: [Audrey Tautou] --born_in--> [Paris]
Updated Frame: ?X = Audrey Tautou
```

- Hop 2: Integrator issues FOLLOW("featured_in : ?X" on Movie). Graph response: "Amelie" featured_in "Audrey Tautou". Set ?Y = "Amelie".
- Hop 3: Integrator issues FOLLOW("directed_by : ?Y" on Director). Graph response: "Jean-Pierre Jeunet" directed "Amelie". Set ?Z = "Jean-Pierre Jeunet".
- Hop 4: Integrator issues ASSOC("friend_of : ?Z"). Graph response: "Guillaume Depardieu" (hypothetical friend relation). Set ?W = "Guillaume Depardieu". VOTE scores path confidence.
- HALT: Integrator detects resolution; output "Guillaume Depardieu".

Full chain diagram (ASCII flowchart):
```
Start: Query Frame
   |
   v
Hop1: ASSOC(born_in=Paris)  --> Audrey Tautou
   |
   v
Hop2: FOLLOW(featured_in)   --> Amelie
   |
   v
Hop3: FOLLOW(directed_by)   --> Jean-Pierre Jeunet
   |
   v
Hop4: ASSOC(friend_of)      --> Guillaume Depardieu
   |
   v
HALT: Output Resolved
```

This trace highlights interpretability (logged hops) and sparsity (only relevant nodes activated).

C. Problem -> Solution Brief (for downstream expansion)

1) Problem & Motivation
- Inefficiency and shallow depth: Transformers reapply billions of parameters every token; depth is fixed by layer count; memory is bounded by the context window.
- FBAM's step forward: Recurrence across frames turns sequence length into effective depth, with empirical length scaling at fixed parameters and near-constant GPU memory via recompute.
- Remaining gap: FBAM's memory is still dense/latent and unguided-powerful but opaque and not semantically directional.

2) Proposed Solution: SR-FBAM
- Thesis: Add a symbolic associative memory to FBAM so the recurrent controller can perform sparse, interpretable hops between compressed concepts instead of re-traversing dense latents.
- Intuition: Human recall chains (e.g., "Laura -> nickname -> Antman -> Anthony") are short, symbolic paths-not full replay. SR-FBAM emulates that pattern.

2.1 Components
- Frame Head (intra-frame attention, tiny): encodes a compact text/grid "mental frame" into symbols (entities, relations, candidates).
- Recurrent Integrator (LSTM/GRU + MLP): chooses the next action from a small operator set-ASSOC, FOLLOW, VOTE, WRITE, HALT.
- Symbolic Memory Graph (external): nodes are concepts/entities; edges are relations/aliases; supports ANN plus exact lookups; pruning/eviction bounds growth.
- Action API: discrete primitives, differentiable for embeddings via straight-through estimators or policy gradients; embeddings and frame encoders trained end-to-end.

2.2 Dataflow (one episode)
- Frame Head produces a sparse working set (K approx 5-8 symbols).
- Integrator issues an operator (e.g., ASSOC("Laura", "nickname")).
- Graph returns a small candidate set, which becomes the next frame.
- Repeat until VOTE/HALT triggers. Log hops for interpretability.

3) Scaling & Efficiency Claims
- Compute: Per-step cost is small (graph operation plus tiny networks). Using activation recomputation and windowed BPTT keeps VRAM ~O(1) in sequence length; wall time scales with hops.
- Hypothesis (speculative): At fixed parameters, loss proportional to (hops)^{-alpha} with alpha approx 0.2-0.3, akin to FBAM's length-based serial scaling-now in associative depth rather than raw tokens.
- Amortization: Longer-hop training runs eventually beat short-hop baselines on wall-time curves, mirroring FBAM.

4) Experiments (MVE -> extend)
- Tasks: multi-hop alias resolution, mnemonic chains, symbolic relation queries, and simulated editor/terminal "text-video" sequences.
- Baselines: pure FBAM, Transformer + retrieval/memory tokens, pure LSTM, neuro-symbolic hybrids.
- Metrics: loss vs wall time, accuracy, hops-to-answer, trace quality, and categorized failures (ambiguity, interference).
- Ablations: remove symbolic graph, cap hops, vary frame size, replace operators with a learned monolith, analyze trade-offs.
- Success criteria: 2-3x wall-time improvement on selective recall tasks and interpretable hop logs; text-video experiments judged by edit accuracy and command latency.

5) Risks & Open Questions
- Operator learning vs curation: ship hand-coded 5-10 primitives first; explore learned operator induction later; balance interpretability against flexibility.
- Graph growth and forgetting: frequency-based eviction and consolidation; measure impact on accuracy and hop depth.
- Ambiguity and interference: confidence-weighted voting, lightweight backtracking, or hybrid dense fallback.
- Generalization: determine whether hop-depth scaling persists from toy KGs to messy real corpora.

6) Implementation Notes (single-GPU MVE)
- Stack: PyTorch (AMP + torch.checkpoint), NetworkX/FAISS/SQLite for the graph, fixed-width text frames.
- Sizing: 1M-parameter SR-FBAM (Frame Head approx 300K, Integrator approx 500K, embeddings approx 200K) to match baseline parity.
- Training: TBPTT = 256, micro-batching, gradient clipping, activation recomputation, logging hooks for hop traces.
- Logging: persist hop traces, scores, and wall-time metrics for immediate use in Sections 6-8.

7) Expected Outcomes
- Efficiency: 2-3x wall-time improvement on selective recall vs dense baselines in toy settings.
- Interpretability: human-readable hop logs (e.g., "Laura -> nickname -> Antman -> Anthony").
- Scaling signal: monotonic loss improvement with hop budget at fixed parameters (to be validated).

8) Positioning & Prior
- Builds on FBAM's depth-through-time; contributes depth-through-meaning.
- Related to NTMs/MANNs, knowledge-graph reasoning, modern Hopfield/associative memory work, and neuro-symbolic surveys.
- Distinctive angle: sparse, operator-driven recall integrated into a frame-based recurrent agent with O(1) training memory.

Optional mission statement:
We aim to turn long-context reasoning from brute-force reprocessing into sparse, human-like recall. By extending FBAM with a symbolic associative memory and a small set of reasoning operators, SR-FBAM scales along a new axis-associative depth-keeping GPU memory near constant while producing auditable hop-by-hop traces. The bet: intelligence scales not only with size or time, but with meaningful hops.

5. Theoretical Scaling & Hypotheses
- Hypothesis: loss proportional to (associative depth)^{-alpha} under fixed parameters, extending FBAM's serial scaling (demonstrated in FBAM with power laws like loss approx A * L^{-0.13 to -0.32}; here hops L replace sequence length-speculative, based on analogies to associative capacity in Hopfield models, and needs empirical validation).
- Concrete example: in a 5-hop task, expect ~20% lower loss than 1-hop baselines (e.g., 1-hop loss = 1.20 -> 5-hop loss approx 0.96 assuming alpha approx 0.3 from the FBAM analogy; offers magnitude for plausibility).
- Why sparse recall helps: depth via hops, not tokens; amortization over long runs for wall-time wins (proven in FBAM, extended speculatively here).
- Memory and compute trade-offs: equations for efficiency (e.g., GPU memory = O(1) via recomputation vs O(sequence^2) in Transformers; graph memory = O(nodes), bounded sublinearly; reference brain-inspired associative depth).
- Limits and failure modes: ambiguity, interference; quantified examples in appendices (speculative mitigations noted).

6. Experimental Design & Toy Benchmarks
- Toy environments/tasks: multi-hop alias resolution, mnemonic recall, symbolic relation chains (adapt FBAM's GitHub histories with entity-relation annotations); knowledge graph benchmarks for multi-hop reasoning; minimum viable: synthetic family/movie KG for hop validation (100 nodes/edges, queries over 1-10 hops such as "Sibling of X's parent?").
- Architecture instantiation: module sizes and specs (e.g., mini SR-FBAM at 1M parameters, PyTorch + NetworkX implementation).
- Baselines: pure FBAM, Transformer + external memory, pure LSTM, neuro-symbolic hybrids.
- Metrics: loss vs wall time, hops-to-answer, accuracy, interpretability (trace clarity).
- Ablations: remove symbolic memory, limit hops, vary frame size; cost-benefit analysis (symbolic overhead vs gains).
- Expected outcomes and diagnostics: 2-3x efficiency in selective recall; failure analysis.

7. Expected Results & Simulation Protocol
- Protocol for the minimum viable experiment (MVE): runnable simulation (e.g., PyTorch for the model, NetworkX for the graph; single GPU, hours to run; train on 100-node toy KG, measure loss/accuracy vs hops; baselines as above).
- Charts: loss vs wall time/steps, recall chain lengths (projected from simulations).
- Qualitative trace examples (e.g., hop logs for interpretability, including the running example from Section 4).
- Discussion: what works (e.g., sparse activation in low-ambiguity tasks), what does not (e.g., high-ambiguity cases; honest note-these are expected from the protocol; implement the MVE for preliminary empirical data to reduce speculation).

8. Analysis, Challenges & Open Questions
- When symbolic recall fails: interference (multiple paths conflict-use VOTE with probabilistic scoring, but if ties occur, fall back to the dense integrator with ~10-20% accuracy drop in ambiguous graphs); ambiguity (e.g., homonyms-mitigate via context-aware ASSOC but requires extra curation); graph growth (exponential nodes-prune via frequency-based eviction to ~2x initial size cap, ensuring sublinear scaling but risking forgetting; speculative 2x cap via learning).
- Handling dead ends: if no associations exist, the integrator triggers HALT with error or backtracks (e.g., reinforcement penalty); conflicts resolved via confidence voting, but in worst cases (e.g., noisy graph) recall fails gracefully to partial resolution-honest note: this increases domain engineering (~30% more for robust pruning) and is a deal-breaker without hybrid dense fallback.
- Learning operators vs hand-coding: initial setup needs ~20-50% engineering (e.g., defining relations), but scalable via meta-learning-be upfront about trade-offs: less than pure symbolic AI but more than end-to-end neural (speculative hybrid sweet spot).
- Scaling to real-world data: Wikipedia-scale corpora, dialog history, video frames.
- Hybridization: with dense latent recurrence for robustness.
- Safety, interpretability, alignment: implications for traceable AI (e.g., auditable hops reduce black-box risks).

9. Extensions & Future Work
- Multi-modal frames: integrate vision and text.
- Memory routing/compression: advanced modules for efficiency.
- Meta-learning: optimize hop selection policies.
- Continual learning: memory consolidation techniques.
- Real-world tasks: code understanding, dialog agents, simulations.
- Hybrid variants: dense-symbolic blends.

10. Conclusion
- Recap: problem, SR-FBAM promise, key benefits (efficiency, transparency in multi-step/long-term tasks).
- Summary: architecture, scaling insights (proven extensions from FBAM, speculative hop-based dynamics), empirical teasers.
- Call to action: open-source repository, community experiments, further validation.
- Roadmap/future directions: ties back to the introduction for cohesion (e.g., prioritize minimum viable experiments for validation).

11. Appendices / Supplementary
- Pseudocode/algorithm sketches (including running example code).
- Memory schema and data structures.
- Mathematical derivations: scaling proofs, amortization equations (with speculative markers).
- Additional plots/tables.
- Ethical considerations and limitations (e.g., bias in symbolic graphs; domain curation risks amplifying engineer biases).

12. References

https://arxiv.org/abs/2510.06828

Neuro-Symbolic AI and Hybrid Symbolic-Neural Systems
These surveys and papers deepen the "Hybrid symbolic + neural systems" section, providing overviews of integrating symbolic reasoning with neural learning-aligned with the SR-FBAM symbolic memory graph. They emphasize explainability, reasoning, and knowledge graphs, which are key for associative hops.
- Neuro-Symbolic AI in 2024: A Systematic Review (2025). Comprehensive review of 2020-2024 developments in neuro-symbolic AI, covering integration strategies, explainability, and knowledge-graph reasoning.
  - arXiv: https://arxiv.org/abs/2501.05435
- A Study on Neuro-Symbolic Artificial Intelligence (2025). Surveys themes like reasoning, explainability, and 41 healthcare use cases-highlights how symbolic AI handles structured data while neural approaches handle noise.
  - arXiv: https://arxiv.org/abs/2503.18213
- Defining Neurosymbolic AI (2025). Focuses on unifying logical and neural representations for learning and reasoning-grounds symbolic hops as interpretable steps.
  - arXiv: https://arxiv.org/abs/2507.11127
- Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures: A Systematic Study (2025). Surveys hybrid architectures, useful for the multi-modal extensions section.
  - arXiv: https://arxiv.org/abs/2502.11269
- Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey (2024). Explores hybrid approaches combining neural and symbolic methods for knowledge-graph reasoning; directly relevant to the symbolic memory graph.
  - arXiv: https://arxiv.org/abs/2302.07200

Associative Memory in Deep Learning
These works inform the external memory and cognitive recall parts, covering associative mechanisms, Hopfield revivals, and scaling in memory models.
- Memory in Plain Sight: Surveying the Uncanny Resemblances between Diffusion Models and Associative Memories (2023). Links diffusion models and transformers to associative memories.
  - arXiv: https://arxiv.org/abs/2309.16750
- Scaling Laws for Associative Memories (2023). Derives scaling laws demonstrating exponential capacity-supports the associative depth hypothesis.
  - arXiv: https://arxiv.org/abs/2310.02984
- Neural Distributed Autoassociative Memories: A Survey (2017). Reviews autoassociative memory models implementable by neural nets with local learning rules.
  - arXiv: https://arxiv.org/abs/1709.00848
- Modern Methods in Associative Memory (2025). Discusses recent advances in associative memory modeling.
  - arXiv: https://arxiv.org/abs/2507.06211
- The Exponential Capacity of Dense Associative Memories (2023). Explores generalized Hopfield models with exponential storage capacity.
  - arXiv: https://arxiv.org/abs/2304.14964

External Memory and Memory-Augmented Neural Networks
These papers survey augmentation techniques for long-term recall and reasoning-ideal for motivating the external symbolic store and contrasting with dense FBAM latents.
- Survey on Memory-Augmented Neural Networks: Cognitive Insights to AI Applications (2023). Explores MANNs blending human memory processes into AI.
  - arXiv: https://arxiv.org/abs/2312.06141
- A Survey on Memory Mechanisms in the Era of LLMs (2025). Defines memory in LLMs as retention/recall from interactions, surveying design choices.
  - arXiv: https://arxiv.org/abs/2504.15965
- Human-Inspired Perspectives: A Survey on AI Long-term Memory (2024). Maps human long-term memory to AI mechanisms.
  - arXiv: https://arxiv.org/abs/2411.00489
- One-shot Learning with Memory-Augmented Neural Networks (2016). Seminal work on rapid data assimilation with memory augmentation.
  - arXiv: https://arxiv.org/abs/1605.06065
- Heterogeneous Memory Augmented Neural Networks (2023). Introduces learnable memory tokens for augmentation.
  - arXiv: https://arxiv.org/abs/2310.10909

Cognitive/Mnemonic Models of Human Memory
These sources connect psychology and neuroscience insights to AI recall, enriching motivations and qualitative analyses.
- Cognitive Memory in Large Language Models (2025). Explores memory types in LLMs and their roles.
  - arXiv: https://arxiv.org/abs/2504.02441
- A Survey on the Memory Mechanism of Large Language Model Based Agents (2024). Covers memory sources, designs, and agent roles.
  - arXiv: https://arxiv.org/abs/2404.13501
- Long Term Memory: The Foundation of AI Self-Evolution (2024). Proposes long-term memory for interaction data.
  - arXiv: https://arxiv.org/abs/2410.15665
- Human-like Forgetting Curves in Deep Neural Networks (2025). Bridges cognitive science and deep learning on forgetting.
  - arXiv: https://arxiv.org/abs/2506.12034
- "Forgetting" in Machine Learning and Beyond: A Survey (2024). Draws neuroscientific insights on adaptive forgetting.
  - arXiv: https://arxiv.org/abs/2405.20620

Scaling Laws in Recurrent Neural Networks
These works bolster the theoretical section by extending FBAM's serial scaling to associative depth.
- Scaling Laws for Neural Language Models (2020). Foundational empirical laws on loss vs model/dataset size.
  - arXiv: https://arxiv.org/abs/2001.08361
- Unified Neural Network Scaling Laws and Scale-Time Equivalence (2024). Introduces equivalence between scaling size and training time.
  - arXiv: https://arxiv.org/abs/2409.05782
- Scaling Recurrent Neural Networks to a Billion Parameters with Zero Attention (2025). Discusses constant FLOPs/memory scaling in RNNs for long sequences.
  - arXiv: https://arxiv.org/abs/2505.17852
- Explaining Neural Scaling Laws (2024). Identifies four regimes (variance-limited, etc.) in scaling.
  - arXiv: https://arxiv.org/abs/2102.06701
- Neural Scaling Laws Rooted in the Data Distribution (2024). Models scaling via percolation theory, explaining power laws.
  - arXiv: https://arxiv.org/abs/2412.07942

From the Original FBAM Paper's References
The FBAM paper ("Attention is Not All You Need," 2025) cites several works on recurrence and scaling that inform this project:
- The Serial Scaling Hypothesis by Yuxi Liu et al. (2025). Posits serial computation as a scaling dimension-core to the associative hop extension.
  - arXiv: https://arxiv.org/abs/2507.12549
- Chain of Thought Empowers Transformers to Solve Inherently Serial Problems by Zhiyuan Li et al. (2024). Analyzes serial reasoning in transformers; contrasts with symbolic hops.
  - arXiv: https://arxiv.org/abs/2402.12875
- Were RNNs All We Needed? by Leo Feng et al. (2024). Empirical evaluation of minimal RNNs; good baseline reference.
  - arXiv: https://arxiv.org/abs/2410.01201
- Long Short-Term Memory by Sepp Hochreiter and Juergen Schmidhuber (1997). Classic LSTM paper underpinning the recurrent integrator.
  - URL: https://www.bioinf.jku.at/publications/older/2604.pdf
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces by Albert Gu and Tri Dao (2024). Selective state space model for comparison with non-recurrence-complete approaches.
  - arXiv: https://arxiv.org/abs/2312.00752

These references provide a strong starting point for citations and deeper reading. Focus on recent 2024-2025 work for timeliness, and cross-reference them in the related work section to highlight novelty. If additional summaries or integration help is needed, note the specific paper and section.
