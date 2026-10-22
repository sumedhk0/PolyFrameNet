# v2: one general polymer architecture, trained on hydrocarbons for the hackathon

Status: revised 2026-10-21 after design discussion. This revision replaces the
carbon-centred draft (commit 665099c). Deadline: trained, benchmarked and written up by
the afternoon of 2026-10-22, with a full explanation by 6 pm.

## Intent

**The principle.** The hackathon model and the general polymer model are the same
architecture and the same code. The only difference is the data each is trained on.

- **Hackathon:** trained on a small subsample of OPoly26 hydrocarbon frames, on the laptop
  GPU.
- **Later:** the same code is trained on all of OPoly26 on a cluster, and nothing changes
  but the data config.

**Consequences.** Everything a general polymer needs is built and gated now, even where
hydrocarbon data will barely exercise it:
- any element;
- heteroatom block rules;
- charge and spin inputs;
- long-range electrostatics;
- a dispersion tail.

**v1 is kept.** v1 (the PE-only model) stays in the repo, untouched, as the baseline.

**Invariants carried over from v1:**
- forces come from autodiff;
- nothing is cached across steps;
- frame atoms are chosen by topology only;
- rotations are passed as raw 3-vectors and 3x3 matrices, never compressed;
- arbitrary choices are symmetrised, never selected;
- every radial function goes smoothly to zero at its cutoff.

**Success criteria:**
1. **Gates pass on every chemistry, untrained.** The untrained model passes every
   correctness gate on hydrocarbons *and* on heteroatom polymers it was never trained on:
   PEO, nylon-6, PMMA, PVC, PTFE and PET fragments, plus a charged fragment and an ion
   pair. This is the evidence that the architecture is general.
2. **Hydrocarbon accuracy.** v2 and MACE are trained on the same hydrocarbon subsample
   with the same split, and their test accuracy is reported.
3. **PE comparison.** v2 is evaluated on v1's PE test frames: did generality cost accuracy
   on PE?
4. **MD benchmarks.** Speed, memory and NVE are measured on a PE melt and on a
   polystyrene system.

**Out of scope:**
- reactive chemistry;
- full OPoly26 training;
- periodic DFT data;
- excited states.

## Data facts (scripts/survey_ch.py, the 70 735 C/H frames on disk)

- **Clean frames.** 70 325 have a clean bond graph: every H has one C neighbour, and every
  C has 2-4 neighbours.
- **Multiplicity.** Every frame is 1.
- **Charge is not stored.** Hydrocarbon frames are taken as neutral. The general fetch
  must store charge, and ColabFit should carry it; the gap is recorded for the cluster
  run.
- **Carbon environments.** Aromatic CH and aromatic C dominate:

  | Environment | Count |
  |---|---|
  | (2,1) | 1.97 M |
  | (3,0) | 0.97 M |
  | (2,2) | 1.36 M |
  | (1,3) | 0.87 M |
  | (3,1) | 0.30 M |
  | (4,0) | 79 k |
  | (1,2) | 43 k |
  | sp carbons (2,0) and (1,1) | 10 k |
  | (0,x) fragments | 6 k |

- **Size.** Median 179 atoms per frame, maximum 357.

## Stage 0a: bonds (element-general)

- **Bond test.** Two atoms are bonded if their distance is below s · (r_cov,i + r_cov,j),
  using covalent radii (Cordero 2008) and s = 1.2.
  - This replaces v1's C-C / C-H thresholds.
  - It must reproduce v1's bonds on every v1 PE frame (gate).
- **Search.** KD-tree, periodic as in v1.
- **Hard rejections:**
  - an H bonded to more than one atom;
  - an atom with more neighbours than its element allows, max_valence[Z] (C 4, N 4,
    O 3, H 1, halogens 1, Si 4, S 6, P 5).
- **Frames under `reactivity/` are dropped.**

## Stage 0b: the block rule (graph only, no bond orders)

Bond orders are never perceived. Inferring them from geometry is fragile on carved,
charged or strained fragments, and a wrong guess changes the blocks.

**Rules, applied in order:**
1. **Ring systems.** Every ring system is one block. Rings are bonds that lie on a cycle
   (bridges are the non-ring bonds). Fused rings merge.
2. **Non-ring atoms.** Every remaining non-terminal atom starts as its own block.
3. **Terminal atoms.** Atoms with degree 1 join their neighbour's block. This covers
   H, F, Cl, Br, carbonyl O, nitrile N and so on. Two terminal atoms bonded only to each
   other (H2, HCl) form one block. An atom with degree 0 (a monatomic ion) is its own
   block.
4. **Linear atoms.**
   - A linear atom is a non-ring C with degree 2 (sp: nitrile C, alkyne, allene centre).
     A degree-2 N is treated as bent: imines are bent, and azides are rare and out of
     scope.
   - The linear atom and all of its neighbours' blocks merge into one block (union).
     That is symmetric, so no choice is made.
   - An alkyne chain C≡C therefore merges with the blocks at both ends.
5. **Size cap.** If a block has more than 32 atoms, it is split at ring-ring single
   bonds: bridges inside a fused-through-bridges system. If it still has more than 32,
   the frame is rejected.

**Examples:**

| Polymer | Blocks |
|---|---|
| PE | CH2 |
| PS | CH2, CH, C6H5 |
| PMMA | CH2, C(quaternary), C(=O), O, CH3 |
| PEO | CH2, O |
| Nylon | C(=O), NH, CH2 |
| PET | C6H4, C(=O), O, CH2 |
| PVC | CH2, CHCl |
| PTFE | CF2 |
| Polythiophene | each ring |

**Known cost.** Acyclic double bonds and amide C-N bonds are cut. Their stiff torsions
are learned by Stage 3 rather than inside a block.

**Future option (off in v2).** When reliable bond orders exist (production MD topology
files), conjugated acyclic units could be merged.

**Canonical colours.**
- **Refinement.** Every atom gets a Weisfeiler-Lehman colour, refined over (Z, degree,
  ring flag, neighbour colours) until the partition is stable (at most 20 rounds).
  Stability guarantees that atoms in the same class have the same multiset of neighbour
  colours. The variant rule relies on that.
- **Hashing.** Colours are hashed with hashlib, so they are stable across processes. Do
  not use Python's `hash`.
- **Determinism.** Colours are permutation-invariant by construction. Every "choose
  canonically" in this spec means: choose the colour class that is smallest (fewest
  atoms), ties broken by colour value. That class is a set of atoms, so no individual
  atom is ever selected.

## Stage 1: frames by colour-class variants

- **Anchor atoms of block I.**
  - The canonical colour class among the block's atoms with at least 2 neighbours.
  - Linear atoms and degree-1 atoms are excluded as anchors.
- **Variant.**
  - (c, a, b, m), where c is an anchor and a is in the canonical colour class of N(c).
  - b is in the canonical colour class of N(c) \ {a}, and m ∈ {+1, -1}.
  - Here N(c) is all bonded neighbours, inside or outside the block.
- **Frame:**
  - e1 = unit(r_a - r_c);
  - e2 = unit(r_b - r_c with its e1 component removed);
  - e3 = m (e1 × e2);
  - origin = the centroid of the block's atoms.
- **Exactness.** Any symmetry of the system maps colour classes to colour classes, so the
  variant set is closed under the symmetry. Mean over variants is therefore exactly
  O(3)- and permutation-invariant.
  - WL colours may merge some non-equivalent atoms. That only adds variants, which is
    still exact.
  - No automorphism enumeration is needed.
- **Conditioning.**
  - a and b are bonded neighbours of a non-linear atom, so the angle a-c-b is
    90-125 degrees in practice.
  - The gate reports the minimum of sin(a-c-b) on all data.
- **Variant counts (expected):**

  | Block | Variants |
  |---|---|
  | CH2 in PE | 4 (as in v1) |
  | CH3 | 6 |
  | Phenyl (ipso anchor) | 4 |
  | Benzene | about 24 |
  | Quaternary C | up to 24 |

- **Frame-free blocks.**
  - Which: a block with no anchor (monatomic ion, linear molecule such as CO2 or N2, H2).
  - What it gets: frame-free invariants only, namely distances and type. It contributes
    no projected features: P1 and P2 are zero for its atoms, and Stage 2 uses distances
    only.
- **Storage.**
  - Variants are stored flat: `var_block`, `var_c`, `var_a`, `var_b`, `var_m`.
  - Means are scatter-means over `var_block`.
  - No padding.

## Stage 2: shared block encoder

- **One encoder for every block type**, so new monomers need no new weights.
- **Padding.** Each block is padded to the size cap.
- **Encoder steps, for each variant v of block I:**
  1. **Atom input.**
     - The frame coordinates F_v^T(r_k - o) of atom k.
     - A species embedding of atom k (element embedding table for Z ≤ 86).
     - Its degree and ring flag.
     - An external-bond count.
  2. **Pair features.** For each pair (k, l) inside the block, MLP(gaussians(|r_k - r_l|),
     emb_k, emb_l).
  3. **Atom state.** x_k = MLP(atom input_k + Σ_l pair_kl).
  4. **Block feature.** H_I,v = MLP(Σ_k x_k).
- **Averaging.** Per-atom features h_k and the block feature H_I are means over v.
- **Invariance to atom labelling inside the block.** It comes from the sums. The
  invariance to frame choice comes from the variant mean. No hand-written groups are
  needed.
- **Cost.** Pair features cost O(n_block²) per block. With a cap of 32 that is small, and
  for v1-like CH2 blocks n = 3.

## Stage 3: bonded channel, 3 layers on the block graph

- **Block graph.**
  - Blocks are joined by a 1-hop edge when a covalent bond crosses them.
  - A 2-hop edge joins blocks that share a neighbour block.
- **Message J → I, for each variant v of I:**
  - Inputs: H_I and H_J.
  - Σ over atoms a in J of ψ(F_v^T(r_a - o_I), emb_a).
  - Class-pooled cross-block pair Gaussians, pooled by species pair. Species pairs are
    folded into a learned pair embedding, not a fixed 3-class list.
  - A one-hot of the hop type.
- **Edge message.** m_IJ = mean over v.
- **Update.** H_I += MLP(H_I, Σ_J m_IJ).
- **Why no R_IJ.** J's frame is not needed, because J's atom positions in I's frame
  encode the relative orientation. Raw 3-vectors enter.

## Valence prior

- **Pairs.** Every topological 1-2, 1-3 and 1-4 pair, found by BFS over the atom graph.
- **Input.** 48 Gaussians of r_ij from 0.8 to 4.4 Å, the one-hot of the distance, and a
  symmetric pair encoding (u_i + u_j, u_i * u_j).
  - u_i is the embedding of atom i's (Z, degree, ring flag).
  - That lets C-C single, aromatic and double bonds, and C-O versus C=O, have different
    minima without bond orders.
- **Energy split.** Half of each pair energy goes to each atom.

## Stage 4: non-bonded, one pass

- **As in v1:**
  - global Cartesian moments M0, M1 and M2, with a cutoff of 4.5 Å;
  - every pair is kept;
  - a Bessel radial basis with a smooth envelope.
- **Radial weights.** They use a learned species-pair embedding, not a fixed species-pair
  table.
- **Per variant.**
  - Project F_v^T M1 and F_v^T M2 F_v.
  - Form the 2-, 3- and 4-body products B_v.
  - Take the mean of B_v over v.
  - Products are taken before the mean, so orientation information survives.

## Stage 5: long range (new, built and gated now)

**5a. Latent-charge electrostatics** (in the style of latent Ewald summation,
Cheng 2024).
- **Charges.**
  - q_i = MLP(final atom features).
  - q_i is then shifted so that Σ q_i equals the frame's total charge Q, an input.
  - The shift is uniform per structure.
- **Energy.**
  - Clusters: E_es = k Σ_{i<j} q_i q_j erf(r_ij / σ) / r_ij over all pairs. Erf screening
    removes the 1/r singularity, so the energy is smooth.
  - Periodic: Ewald summation with the same Gaussian smearing. That is a real-space
    erfc sum within r_c plus a reciprocal sum over k vectors, with the self term
    removed. The Ewald α is chosen from box size.
- **No supervision.** Charges are not fitted to any reference charge. They are latent
  and learned from energies and forces only.

**5b. Dispersion tail.**
- **Energy.** E_disp = -Σ_{i<j} C6_ij / (r_ij^6 + R0_ij^6) · S(r_ij).
- **C6.** C6_ij = sqrt(C6_i C6_j). C6_i = softplus(MLP(atom features)), so it depends on
  the environment.
- **R0.** R0_ij comes from learned per-element radii.
- **Switch.** S is a C2-smooth switch from 1 at 8 Å to 0 at 10 Å.
- **What it covers.** The VV10 attraction that the 4.5 Å Stage 4 cannot see. The short
  range is learned by Stage 4.
- **Neighbour list.** A second list at 10 Å (vesin), built on every call.

**Global conditioning.**
- Total charge Q and spin multiplicity S are per-structure inputs.
- An embedding of (Q, S) is added to every atom's Stage 2 input. This follows UMA.
- Hydrocarbon training uses Q = 0, S = 1 only. The conditioning is present, exercised by
  the gates and untrained.

## Readout

- **Per-atom energy.** eps_i = MLP([h_i, H'_I, B_i, species embedding]).
- **Total energy.**
  - E = Σ_i (eps_i + valence_i) · scale + E_es + E_disp + Σ_i E0[Z_i].
  - E0 is stored as int64 micro-eV and the total is summed in float64, as in v1.
  - E0 is fitted by least squares per element on the training set. Elements absent from
    the training set get E0 = 0, and the checkpoint records which ones are fitted.
- **Forces.** F = -dE/dr by autodiff, through every term.

## Data and training (hackathon config)

The data is selected by a config, so moving to the cluster changes only the config.

**Hackathon config:**
- **Elements.** {H, C} only.
- **Frames.** Clean frames under the general block rules.
- **Split.** By system key, after stripping the `_Hterm` and `_frame` suffixes.
- **Size.** About 3 000 frames (about 540 k atoms) subsampled by system.
- **PE overlap.** Every v1 PE frame is included. v1's PE val and test systems are forced
  into v2's val and test splits.
- **Proportions.** 80/10/10 by system.

**Training:**
- Loss: 20 · MSE(per-atom E) + MSE(F).
- AdamW amsgrad, cosine schedule over wall clock, EMA 0.99.
- float32 training, float64 evaluation.
- Budget: 120 min.

**MACE baseline:**
- The same settings as for v1.
- Trained on the identical extxyz split, at the same time. The GPU sharing is disclosed.

**Cluster config (recorded, not run):**
- **Data.** All of OPoly26, all elements, all charges and spins.
- **Prerequisite.** The fetch script stores charge.
- **Sampling.** Balanced across chemistry folders.
- **Training.**
  - Multi-GPU DDP, or the same loop with gradient accumulation.
  - No architecture change.

## Code layout

| Module | Contents |
|---|---|
| `src/frame_potential/v2/graph.py` | Bonds by covalent radii, the block rule, WL colours, the variant table, the block graph and valence pairs. Pure numpy and networkx. |
| `src/frame_potential/v2/batch.py` | Collate (padded block atoms, variants, Q and S), the 4.5 Å and 10 Å neighbour lists. |
| `src/frame_potential/v2/model.py` | `PolyFrameNetV2`: Stages 1-6. It reuses `mlp`, `gaussians` and `bessel_envelope` from v1. |
| `src/frame_potential/v2/longrange.py` | Erf Coulomb for clusters, Ewald for periodic, and dispersion. |
| `src/frame_potential/v2/data.py` | Config-driven loader and subsampler. |
| `train.py` | `--arch v2`, plus `--data-config`. |
| `calculator.py` | Picks the architecture from the checkpoint. |

Test molecules come from `builders`, or from small hand-written xyz files in
`tests/data/`:
- PS and PE chains;
- PEO, nylon-6, PMMA, PVC, PTFE and PET fragments;
- a periodic PE melt;
- benzene;
- methane;
- Na+ Cl- (frame-free).

v1 code is untouched.

## Gates (tests/test_v2_gates.py; every test molecule unless noted)

**Symmetry and smoothness:**
- rotation + translation, reflection, permutation of all atoms, wrap: |ΔE| < 1e-9 eV;
- finite-difference forces: < 1e-8 eV/Å. E0 is set to 0, as in v1;
- cutoff smoothness at 4.5 Å and in the 8-10 Å dispersion switch.

**Consistency:**
- batch equals single;
- calculator equals model;
- float32 vs float64.

**Stage 0 and Stage 1:**
- **Bonds.** On every v1 PE frame, the covalent-radius bonds equal v1's bonds.
- **Variant-set oracle (true oracle).** Apply a random permutation + rotation + mirror.
  Rebuild everything from scratch. For each block, the multiset of per-variant Stage 2
  inputs must equal the original (matched after sorting, to 1e-9).
- **Negative control.** Drop one variant per block. The rotation or permutation gate must
  then fail.

**Stage 5:**
- **Cluster Coulomb.** Erf Coulomb on a 2-charge cluster equals the analytic value.
- **Ewald, two checks.**
  - It converges as the box grows: the cluster limit matches.
  - It is independent of α at fixed accuracy.
- **Charge conservation.** Σ q_i = Q to 1e-10, for Q ∈ {-1, 0, +1}.
- **Frame-free blocks.** Na+ Cl- gives finite E and F, and passes every symmetry gate.

## Build order

1. graph.py, with the bond, block, colour and variant gates.
2. Model Stages 1-4 and the readout, with the symmetry gates on all molecules.
3. Stage 5 and the long-range gates.
4. Data loader and a smoke train.
5. Full training of v2 and MACE, run together.
6. Benchmarks and writeup.

Commit after each step.

## Risks and fallbacks (deadline-driven, ordered)

| Risk | Fallback |
|---|---|
| Ewald slips | Ship Coulomb for clusters only, and disable E_es for periodic systems, with an explicit error flag. Training data is clusters, so the hackathon numbers are unaffected. Disclosed. |
| Too slow at 1k atoms because of variant counts | Average B over v before the products for most channels. |
| Variable-block encoder too slow | Bucket blocks by size and run one encoder call per bucket. |
| Data imbalance (aromatics dominate), PE accuracy falls | Measured by criterion 3. Report it; do not hide it. |
| Overall time | Cut PS MD first, then the dispersion tail. Without it, nothing models attraction beyond 4.5 Å; that matters for bulk density, but little for the cluster energies in the training data. Disclosed if cut. |
