# PolyFrameNet, explained from first principles

A teaching document for presenting this repository. Every claim about the code cites
`file:line`. Paths are relative to `src/frame_potential/` unless they start with `tests/` or
`docs/`. "v1" is the polyethylene-only model (`model.py`, `topology.py`, `batch.py`), and "v2"
is the general-polymer model (`v2/*.py`).

Numbers quoted for accuracy, speed and NVE all come from `docs/writeup.md` and are **v1**
numbers. At the time of writing this repository records no v2 accuracy or speed results,
so do not quote any.

---

## 0. What an interatomic potential is

### 0.1 Energy in, forces out

A molecular dynamics (MD) simulation moves atoms by Newton's law, so it needs the force on
every atom at every timestep (about 1 fs). The exact answer comes from quantum mechanics
(DFT), which costs minutes to hours per step. An **interatomic potential** is a cheap
function

```
E = E(r_1, ..., r_N; Z_1, ..., Z_N)      positions r_i, atomic numbers Z_i
```

that mimics the DFT energy. A **machine-learned interatomic potential (MLIP)** is such a
function written as a neural network and fitted to DFT energies and forces.

**Forces are the negative gradient of the energy:** `F_i = -dE/dr_i`. In this code that
gradient is taken by PyTorch autodiff, never predicted by a separate network head
(`model.py:472-478`, `energy_and_forces`).

**Why autodiff and not a force head?** A force field that is the gradient of a scalar is
*conservative*. The work done around any closed loop is zero, so total energy
(kinetic + potential) is conserved in NVE MD. A directly predicted force is generally not
a gradient of anything. Then an atom can go round a loop and come back with more energy than
it started with, and the simulation heats up at any timestep. This is invariant 1 in
`CLAUDE.md`.

### 0.2 Symmetry: what the energy must not depend on

Physics says the energy of an isolated system is unchanged by:

| Operation | Meaning | What breaks if violated |
|---|---|---|
| Translation | shift every atom by the same vector | spurious net force on the whole system |
| Rotation | rotate the whole system | spurious torque; results depend on lab orientation |
| Reflection (mirror) | mirror the whole system | mirror-image conformers (gauche+ and gauche-) get different energies |
| Permutation | relabel identical atoms (swap two H) | energy depends on the arbitrary order in the input file |
| Periodic wrap | move an atom by a lattice vector | energy jumps when an atom crosses the box boundary |

Together, rotations and reflections form the group **O(3)**. Rotations alone form **SO(3)**
(determinant +1). A model can be built to be **invariant** (the output does not change) or
**equivariant** (the output transforms along with the input, as a force vector does). The
energy must be invariant. The forces then come out equivariant automatically, because they
are the gradient of an invariant.

A model can learn symmetry approximately from augmented data. Exact symmetry by
construction is better, for two reasons:

- it removes a source of error everywhere;
- it cannot fail on a configuration that training never covered.

### 0.3 Smoothness

MD integrates forces along a trajectory. Any jump in E, or in its gradient, as atoms move
injects or removes energy. Cutoffs are therefore made smooth, so that a neighbour entering
the cutoff sphere contributes exactly zero energy and zero force at the moment it enters.

### 0.4 What MACE does (the baseline)

MACE, like NequIP and Allegro, is an **equivariant message-passing network**:

- **Features carry spherical-tensor indices.** Each atom holds scalar (l = 0), vector
  (l = 1) and higher-l features. Under rotation they transform by known matrices (Wigner D).
- **Messages combine features by tensor products.** A message from j to i couples j's
  features with the direction r̂_ij, expanded in spherical harmonics Y_lm. The coupling uses
  Clebsch-Gordan coefficients, which are the only way to multiply two spherical tensors and
  get a spherical tensor back.
- **Body order comes from symmetric contractions.** Taking products of the per-atom sums is
  the ACE trick (Section 2.6).
- **Two layers** give a receptive field of 2 × r_cut.

The price is that tensor products are expensive and memory-hungry. In v1's benchmark MACE
needed about 1.6 MB per atom against 0.13 MB for PolyFrameNet (`docs/writeup.md:374-375`).
MACE also rediscovers the chemical bonds from distances at every step.

---

## 1. The core idea

### 1.1 Three facts that generic MLIPs throw away

1. **Polymers are built from repeating units.** The chain is cut into **blocks**, and every
   block of the same kind shares the same weights.
2. **The covalent bond graph is sparse and fixed** (non-reactive MD). Bonds are read once from
   the first geometry and frozen. They decide blocks, frame atoms and which pairs get a valence
   term (`topology.py:76-106`, `v2/graph.py:69-96`).
3. **The backbone supplies a local coordinate frame.** Every block has atoms that are always
   bonded in the same pattern, and those atoms define three orthonormal axes.

### 1.2 Why a frame removes the need for tensor products

Let `F` be a 3×3 matrix whose columns are the block's axes e1, e2, e3, built from the block's
own atom positions. Rotate the whole system by Q:

```
r -> Q r         =>   e_k -> Q e_k   =>   F -> Q F
v = r_a - o      =>   v -> Q v
F^T v  ->  (QF)^T (Q v) = F^T Q^T Q v = F^T v          (Q^T Q = I)
```

So `F^T (r_a - o)`, the atom's coordinates expressed in the block's own axes, is **rotation
invariant**. So is `F^T M1` for any vector-valued quantity M1, and `F^T M2 F` for any
matrix-valued M2. After projection everything is plain numbers, and every network downstream
is an ordinary MLP. No spherical harmonics, no Clebsch-Gordan coefficients, no irreps.

This moves the difficulty rather than removing it. The model is now only as good as its
frames, and frames can fail in two ways:

- **Ambiguity.** Which hydrogen defines e2? Which way along the chain is e1?
- **Degeneracy.** If the two atoms that define e1 and e2 become collinear, e2 is undefined.

The rest of the design deals with those two failure modes.

### 1.3 "Symmetrize, never select"

Whenever the frame recipe involves an arbitrary choice (which H, which chain direction,
which handedness), the code does not pick one. It **evaluates every choice and averages**.

- **Why not select?** Suppose you pick "the H nearest to some reference". When two
  hydrogens swap order as atoms vibrate, the choice flips, the frame jumps, and the energy
  jumps. That discontinuity breaks energy conservation.
- **Why averaging works.** The average over a *set of choices that the symmetry maps onto
  itself* is exactly invariant, and it is as smooth as each term. This is the Reynolds
  operator, or group averaging.

### 1.4 The six invariants (`CLAUDE.md`), with a one-line why

1. **Forces are `-dE/dr` by autodiff, never a force head.** Otherwise the force field is not
   conservative and energy drifts in NVE.
2. **No caching across timesteps.** Energy must be a pure function of the current positions.
   A cached value carries an old geometry into a new step and breaks the gradient.
3. **Frame atoms are selected by topology, never by geometric ranking.** A ranking flips
   discontinuously when two neighbours cross, so the frame jumps.
4. **Never compress a rotation.** No representation of SO(3) with fewer than 5 numbers is
   continuous (Zhou et al. 2019). Euler angles and quaternions all have a seam where they
   jump. Rotations enter as 9 raw matrix entries (v1) or as raw 3-vectors (v2).
5. **Symmetrize over arbitrary choices, never resolve them.** Averaging is smooth and
   symmetric; selecting is neither.
6. **Radial functions go smoothly to zero at the cutoff.** A hard truncation makes E jump
   when an atom crosses r_cut.

> The recurring bug in this design is an arbitrary or torn choice leaking into a quantity
> that must be smooth. For every new representation, ask: what happens on the trajectory
> where it has to jump?

---

## 2. v1: polyethylene end to end

The pipeline is in `model.py:1-23`. The data flow is:

```
positions, cell
  │
  ├─ Stage 0  topology (once): bonds, chains, C2 blocks, block graph, valence pairs
  │           radius graph (every call): (i, j, shift) within 4.5 Å
  ├─ Stage 1  per block: origin o, 4 frames F (fwd, rev, fwd-mirror, rev-mirror)
  ├─ Stage 2  per block: dense MLP, averaged over its symmetry group -> h_i (64), H_I (128)
  ├─ Stage 3  2 message-passing layers on the block chain -> H'_I
  ├─ Valence  learnable f(r) on 1-2, 1-3, 1-4 pairs
  ├─ Stage 4  one radius-graph pass: Cartesian moments M0, M1, M2 projected into the block frame
  └─ Stage 6  eps_i = MLP(...), averaged over 4 frames; E = Σ(eps_i + val_i)·s + Σ E0[Z_i]
```

### 2.1 Stage 0: topology (`topology.py`)

- **Bonds.** C-C < 1.85 Å and C-H < 1.35 Å, found with a KD-tree that is periodic-aware
  (`topology.py:34-35, 76-106`).
- **Validation.** Every H has exactly one C neighbour. Every C has exactly 4 neighbours, at
  most 2 of them carbon. Anything else (rings, branches, radicals, unsaturation) raises
  `TopologyError`, and the frame is dropped (`topology.py:141-149`).
- **Chains.** Each chain is walked from an end carbon (`topology.py:151-174`). Carbons left
  unvisited mean a ring, or a chain bonded to its own periodic image. Both are refused.
- **Blocks.** Each chain is cut into C2 pairs (`topology.py:182-212`):

  | Type | Atoms | Slot order |
  |---|---|---|
  | `PE_end` | CH3-CH2- (7) | Cm, Cc, Hm1, Hm2, Hm3, Hc1, Hc2 |
  | `PE_mid` | -CH2-CH2- (6) | C1, C2, H1a, H1b, H2a, H2b |

  An n-carbon chain (n even, at least 4) gives 2 end blocks and (n-4)/2 mid blocks.
- **Odd chains are rejected** (`topology.py:184-185`). Reversing an odd chain would move the
  3-carbon block to the other end, so the partition would depend on storage direction.
- **Block graph.** Hop-1 edges join consecutive blocks; hop-2 edges join blocks two apart
  (`topology.py:209-212`).
- **Valence pairs.** All atom pairs within bond distance 1-3 (BFS), with class
  (d-1)·3 + species pair, giving 9 classes (`topology.py:268-285`).
- **Radius graph.** Recomputed on every call and never cached (`batch.py:146-233`). It
  returns integer triples (i, j, S), where S is a lattice shift. The edge vector
  `r_j - r_i + S·cell` is then built inside autograd (`batch.py:248-254`), so forces are exact
  derivatives.

### 2.2 Stage 1: frames (`model.py:103-128`)

For one block with frame carbon c and its two hydrogens h:

```
e1 = unit(r_C2 - r_C1)
s  = Σ_h [ u_h - (u_h·e1) e1 ],    u_h = unit(r_h - r_c)    (perpendicular parts, summed)
e2 = unit(s)
e3 = e1 × e2
o  = centroid of the block
```

- **Symmetric in the two H.** Swapping them leaves s unchanged, so no H is ever selected.
- **Orthonormal by construction.** e2 is built perpendicular to e1 before normalising.
  (The spec's original blend of two complete frames was not orthonormal: spec review
  flag 4, `docs/writeup.md:108-110`.)
- **End blocks use the CH2 hydrogens** (`model.py:124-126`). The three perpendicular parts of
  a CH3 sum to about zero (three-fold symmetry), so any recipe symmetric in a methyl's
  hydrogens degenerates.
- **Four variants** (`model.py:127-128`):
  - `fwd` uses C1 and its hydrogens;
  - `rev` uses C2 and its hydrogens, with e1 flipped (chain read the other way);
  - each also has a **mirror** copy F·diag(1, 1, -1).

  End blocks have an intrinsic direction (methyl to CH2), so their fwd and rev are the same
  frame.

**Why the mirror copies.** e3 = e1 × e2 is a *pseudovector*. Under an improper Q (det = -1),
e1 → Qe1 and e2 → Qe2, but (Qe1) × (Qe2) = det(Q)·Q(e1 × e2) = -Q e3. So a mirror maps F to
**Q F diag(1, 1, -1)**, not to QF. Without the mirrored variants the energy was SO(3)- but not
O(3)-invariant: gauche+ and gauche- butane got different energies. No rotation test can see
this, because every rotation has det +1. Code review found it, and the reflection gate now
pins it (`tests/test_gates.py:213-222`).

### 2.3 Stage 2: block encoder with group averaging (`model.py:134-184`)

- **Input per block instance** (`model.py:145-149, 168-178`):
  - the frame coordinates F^T(r - o) of every slot (n × 3);
  - a species embedding of 8 per slot;
  - Gaussian expansions (8 Gaussians, 0.8-3.2 Å) of all intra-block distances.

  That is 6·11 + 15·8 = **186 inputs** for a mid block and 7·11 + 21·8 = **245** for an
  end block.
- **Network.** One MLP with 3 hidden layers of 128 maps the input to n·64 per-atom features
  plus a 128-dim block feature.

**Group averaging** (`model.py:74-97, 170-184`). Each group element is a pair
(frame variant, slot permutation):

| Block | Group | Size |
|---|---|---|
| Mid | {H1a↔H1b} × {H2a↔H2b} × {fwd, rev} × {mirror} | 2·2·2·2 = **16** |
| End | S3 on the methyl H × {Hc1↔Hc2} × {mirror} | 6·2·2 = **24** |

- **Reversal** swaps C1↔C2 and the two H pairs, and switches to the rev frame
  (`model.py:79`).
- **One batched call.** All G relabelled inputs go through the MLP together. Per-atom outputs
  are un-permuted back to their atoms (`model.py:182-183`), then everything is averaged.
- **Why this is exact.** Any relabelling of the block's atoms maps the set of G inputs onto
  itself, so the mean cannot change.
- A **block-type embedding** is added to H (`model.py:405`).

### 2.4 Stage 3: bonded channel with R_IJ (`model.py:190-212, 412-441`)

There are 2 layers of message passing on the block chain, with messages flowing J → I.
The edge input (`model.py:206-207`) is:

```
[ H_I (128), H_J (128),
  R_IJ = F_I^T F_J        9 raw entries   (model.py:423)
  d_IJ = F_I^T (o_J - o_I)  3 entries, plus its norm  (model.py:424)
  pooled pair Gaussians   3 species classes × 24 Gaussians (1-7 Å), summed over all
                          cross-block atom pairs, / 4    (model.py:429-438)
  hop one-hot (2) ]
```

- **Variant pairing.** Messages are evaluated with all 4 variants, using the *same* variant
  for both blocks, and averaged (`model.py:205-210`). The pairing is closed under symmetry:
  reversing the chain flips every mid block at once, so (fwd, fwd) and (rev, rev) swap.
- **Update.** Messages are summed per receiver, with a residual update
  `H ← H + MLP([H, agg])` (`model.py:211-212`).
- **R_IJ is 9 raw numbers** (invariant 4). It is the relative orientation of the two blocks,
  and it is invariant because both frames rotate together: (QF_I)^T(QF_J) = F_I^T F_J.

### 2.5 Valence prior (`model.py:218-244`)

A learnable **Urey-Bradley force field**: one MLP f_class(r) applied to every topological
1-2 (bond), 1-3 (angle) and 1-4 (torsion-sensitive) distance.

- **Input:** 48 Gaussians on 0.8-4.4 Å plus a 9-class one-hot.
- **Energy split:** half of each pair energy goes to each atom (`model.py:244`).
- **No cutoff needed.** The pair set is fixed by topology, so pairs never enter or leave.

**Why it was added.** Bond and angle forces are the largest in the system. Without the prior,
the 186-245-input block MLP had to dig these stiff one-dimensional functions out of raw
coordinates. With it, validation force MAE fell from about 340 to 121 meV/Å at the same
budget (`docs/writeup.md:304-309`).

### 2.6 Stage 4: Cartesian moments and body order (`model.py:250-320`)

**Radial basis** (`model.py:250-262`): sin(kπr/r_c)/r times a polynomial envelope (p = 6)
that goes to zero at r_c = 4.5 Å with continuous first and second derivatives.

**Per edge (i, j):** φ_ij,c = R_c(r_ij; species pair) · gate_c(x_j), with 16 channels
(`model.py:294-300`).

**Per atom: moments, accumulated once in the global frame** (`model.py:302-309`):

```
M0_c = Σ_j φ_ij,c                 (16,)
M1_c = Σ_j φ_ij,c  r̂_ij           (16, 3)
M2_c = Σ_j φ_ij,c  r̂_ij r̂_ij^T    (16, 3, 3)       each divided by avg_nbr
```

**Projection into the atom's own block frame** (`model.py:311-316`):

- F^T M1 gives 3 numbers per channel.
- F^T M2 F gives a symmetric matrix; its 6 unique entries are kept.
- With M0, that is A_i = 16 × (1 + 3 + 6) = **160 invariants**.
- **Why this is cheap.** The moments are computed once per atom, and only the cheap
  projection runs per frame variant. Averaging over 4 variants therefore costs per atom, not
  per edge.

**Body order** (`model.py:317-320`):

```
B_i = [ W1 A,   (W2a A) ⊙ (W2b A),   (W3a A) ⊙ (W3b A) ⊙ (W3c A) ]      3 × 64
```

- **Why products give body order.** A is a sum over neighbours, so it is 2-body (centre plus
  one neighbour). A product of two such sums, (Σ_j a_j)(Σ_k b_k) = Σ_{j,k} a_j b_k, contains
  every neighbour *pair*, so it is 3-body, at the cost of one sum instead of a double loop. A
  triple product is 4-body. This is the ACE/MACE trick, done here on frame-projected scalars
  instead of with Clebsch-Gordan couplings.
- **Caveat.** These counts are radius-graph body orders. The gate x_j already contains block
  and bonded-channel context, so the true dependence is richer.

**The exclusion experiment.** The spec said Stage 4 should exclude same-chain pairs within
L_b block hops (`batch.py:236-245`), so that the two channels have disjoint jobs. An ablation
found the opposite (`docs/writeup.md:319-350`):

| Stage 4 sees | Test force MAE after 8 min |
|---|---|
| Spec rule (exclude pairs within L_b hops) | 115.6 meV/Å |
| Everything except the atom's own block | 44.3 meV/Å |
| **Every pair** | **34.4 meV/Å** |

The final model uses `--exclude-hops -1`, which keeps every pair. The likely reason is that
Stage 3 only sees block-level summaries, while the 1-5 and 1-6 H···H contacts that set gauche
vs trans energies need atom-resolved detail.

### 2.7 Readout and E0 (`model.py:454-469`)

- **Per-atom energy.** For each of the 4 variants,
  eps_i,v = MLP([h_i, H'_I, B_i,v, species_emb]). The results are averaged over v.
- **Total energy.** `E = Σ_i (eps_i + valence_i) · s + Σ_i E0[Z_i]`.
  - s is the training force RMS (`train.py:102-104`).
  - E0 is the per-element reference energy, fitted by least squares on the training set.

**Why E0 is stored as int64 micro-eV** (`model.py:365-373, 463-466`):

- E0[C] ≈ -1037 eV, so a 1000-atom total is about -3×10^5 eV. float32 has a 24-bit mantissa,
  so at that magnitude its spacing is about 1/32 eV.
- `model.to(float32)` would round both the buffer and the sum, quantising the total energy to
  tens of meV. That swamps an NVE drift measurement.
- **The fix.** E0 lives in an integer buffer (`e0_micro`), which `.to(float32)` never touches.
  It is converted to float64 inside forward, and the sum over atoms is always taken in
  float64. Training targets also stay float64 (`batch.py:71-72`).

### 2.8 Training (`train.py`)

```
L = 20 · mean_s ((E_pred - E_ref)_s / N_s)^2  +  1 · mean (F_pred - F_ref)^2      (train.py:131-134)
```

- **Forces in the loss.** Forces use `create_graph=True`, so the force loss differentiates
  through the gradient (a double backward).
- **Optimiser.** AdamW with amsgrad, weight decay 1e-6, and cosine decay on the *wall-clock*
  fraction of the budget (`train.py:124-126`). EMA of the weights with decay 0.99
  (`train.py:36-41`), gradient clipping at norm 10.
- **Precision.** float32 training, float64 evaluation.
- **Final v1 run.** lr 5e-3, at most 1000 atoms per batch, 90 min (`docs/writeup.md:551-553`).
- **Model selection** by val E_RMSE + F_RMSE/20 (`train.py:148`).
- **Split by MD system, never by frame.** Frames carved from the same MD cell are strongly
  correlated, so a frame-level split would leak.

---

## 3. v2: general polymers end to end

The spec is `docs/superpowers/specs/2026-10-21-hydrocarbon-generalization-design.md`; its
principle is one architecture, where only the data changes. The v2 pipeline, from
`v2/model.py:1-17`:

```
Stage 0  bonds by covalent radii; blocks by a graph rule; WL colours; variant table (once)
Stage 1  one frame per variant row (c, a, b, m); frame-free rows get F = 0
Stage 2  ONE shared encoder for every block type; ragged; averaged over the block's variants
Stage 3  3 layers on the block graph; message J->I sees J's atoms in I's frame
Valence  1-2/1-3/1-4 pairs, conditioned on both atoms' type embeddings
Stage 4  4.5 Å moments, projected per (atom, variant), products, then averaged
Stage 5  latent-charge Coulomb (erf / Ewald) + learned C6 dispersion (10 Å)
Stage 6  E = Σ (eps + valence)·scale + E_es + E_disp + Σ E0[Z]   (float64)
```

### 3.1 Stage 0a: bonds by covalent radii (`v2/graph.py:69-96`)

- **Bond test.** Atoms are bonded if r < 1.2 · (r_cov,i + r_cov,j), with Cordero radii from
  ASE (`graph.py:31, 94`). This replaces v1's per-pair thresholds and works for any element.
- **Hard rejections** (`graph.py:137-141`):
  - an H with degree other than 1;
  - an atom with more neighbours than `MAX_VALENCE` allows (`graph.py:34`: C 4, N 4, O 3,
    halogens 1, S 6, P 5).
- **Gate.** The v2 bonds equal v1's on a PE melt and on noise-free PE clusters
  (`tests/test_v2_graph.py:27-33`), and on every real v1 PE frame
  (`tests/test_v2_graph.py:36-44`).
- **What the gate found.** On a *noisy* cluster, one C-H stretched to 1.323 Å. That is inside
  v1's 1.35 Å cut but outside v2's 1.2 × (0.76 + 0.31) = 1.284 Å. The two bond rules disagree
  on distorted geometry, and the test documents it (`tests/test_v2_graph.py:30-32`).

### 3.2 Stage 0b: the graph-only block rule (`v2/graph.py:128-184`)

Bond orders are never perceived. Guessing double bonds from geometry is fragile on carved,
charged or strained fragments, and a wrong guess would change the blocks.

The rule is implemented with union-find (`graph.py:150-172`):

1. **Ring systems are one block.** Ring bonds are the non-bridge bonds, found with
   `nx.bridges` (`graph.py:143-147`). Unioning every ring bond merges fused rings
   (`graph.py:161-162`). A bond joining two separate rings, as in biphenyl, is a bridge, so
   the two rings stay separate blocks.
2. **Every other non-terminal atom starts as its own block.** Union-find singletons do this
   implicitly, and a degree-0 ion is its own block.
3. **Terminal atoms (degree 1) join their neighbour** (`graph.py:163-165`): H, F, Cl,
   carbonyl O, nitrile N.
4. **Linear atoms merge with all their neighbours' blocks** (`graph.py:148, 166-169`). A
   linear atom is a non-ring C with degree 2 (nitrile C, alkyne, allene centre). The merge is
   a union, so it is symmetric and nothing is chosen.
5. **Size cap: more than 32 atoms raises `TopologyError`** (`graph.py:177-179`). The spec
   said "split at ring-ring single bonds", but those are bridges, which rule 1 already cuts.
   So the code rejects instead, and the plan records this as a deliberate deviation
   (`plans/...:2599`).

| Molecule | Blocks (as the code labels them) |
|---|---|
| PE chain | `CH3[C]`, `CH2[C2]` ... (one carbon per block, not two as in v1) |
| isobutane | `CH[C3]` + 3 × `CH3[C]` |
| toluene | `C6H5[C]` + `CH3[C]` |
| benzene | `C6H6` (one block) |
| methyl formate | `CHO[O]`, `O[C2]`, `CH3[O]` (`tests/test_v2_graph.py:78-81`) |
| CH3CN | one block (the linear nitrile C merges CH3 and N) |
| CO2 | one block with no anchor, so frame-free |

Labels come from `registry.block_label`. The bracket lists the elements of external
neighbours.

**Known cost.** Acyclic double bonds and amide C-N bonds are cut between blocks, so their
stiff torsions must be learned by Stage 3 and the valence term.

Also computed per atom (`graph.py:136-148, 183-184`):
- `degree`;
- `ring` (the atom has a ring bond);
- `n_ext` (the number of bonds leaving the atom's block).

### 3.3 Weisfeiler-Lehman colours (`v2/graph.py:99-125`)

**The problem.** Frame atoms must be chosen by topology, and a choice like "pick an H" must
never single out one particular atom.

**WL refinement** (`graph.py:105-117`):
- **Start.** Each atom's colour is hash(Z, degree, ring flag).
- **Each round.** New colour = hash(old colour, sorted multiset of neighbour colours).
- **Stop** when the number of distinct colours stops growing, or after 20 rounds.

**Isomorphism invariance.** The colour depends only on the graph, never on atom labels. Any
relabelling, and any graph automorphism, maps each colour class onto itself.

**Hashing is process-stable.** `blake2b` is used, not Python's `hash()`, which is salted per
process (`graph.py:99-102`). Otherwise colours, and with them the variant table, would
differ between runs.

**The canonical class** (`graph.py:120-125`). Among a set of atoms, take the colour class
with the fewest members, with ties broken by colour value. The result is a *set* of atoms,
chosen by an invariant property. If the set has more than one member, every member gets used,
one variant each.

### 3.4 Frame variants (c, a, b, m), and why the set is closed (`v2/graph.py:187-201`)

For block I:

- **c (anchor).** Every atom of the canonical class among the block's atoms with degree ≥ 2
  that are not linear (`graph.py:189, 195`).
- **a (e1 atom).** Every atom of the canonical class of N(c), all bonded neighbours, inside
  or outside the block (`graph.py:196`).
- **b (e2 atom).** Every atom of the canonical class of N(c) \ {a} (`graph.py:197`).
- **m (mirror).** Both +1 and -1 (`graph.py:198`).

The frame for one row is built in `v2/model.py:106-118`:

```
e1 = unit(r_a - r_c)
e2 = unit((r_b - r_c) - ((r_b - r_c)·e1) e1)      Gram-Schmidt
e3 = m · (e1 × e2)
o  = centroid of the block's atoms                (model.py:139-140)
```

**Why the set is exactly closed under every symmetry:**

- **Relabelling.** Colours move with the atoms, so the relabelled system produces the same
  set of (c, a, b, m) atom triples under new names.
- **Rotation Q.** Each frame becomes QF. Same set, rotated.
- **Mirror.** Each frame becomes Q F diag(1, 1, -1), which is the m → -m partner. Same set.

Since the set maps onto itself, any mean over it is exactly invariant. **No automorphism
group is ever enumerated**, unlike v1's hand-written groups.

- **If WL merges two atoms that are not truly equivalent**, the code only gets extra
  variants. The mean stays exact.
- **The colour-value tie-break is not a "select" in the dangerous sense.** It picks a class
  by a graph property that never changes during MD, so it cannot flip.

**Variant counts, measured on the actual code:**

| Block | c | a | b | m | Variants |
|---|---|---|---|---|---|
| CH3 (in PE or isobutane) | 1 | 1 (the external C) | 3 H | 2 | **6** |
| interior CH2 in PE | 1 | 1 C | 1 (the other C) | 2 | **2** (4 when WL ties the two C) |
| CH in isobutane | 1 | 1 (the H) | 3 methyl C | 2 | **6** |
| phenyl in toluene | 1 (a size-1 ring class) | 1 | 2 | 2 | **4** |
| benzene | 6 C | 1 H | 2 ring C | 2 | **24** (`tests/test_v2_graph.py:60-63`) |
| CO2, Na+, Cl- | none | | | | **1 frame-free row** |

The CH2 row differs from the spec's "4, as in v1". WL colours usually tell apart a CH2's two
carbon neighbours by their distance to the chain ends. The CH2 frame is then C-C-C based
(e1 to one C, e2 from the other C), not built from the hydrogens as in v1. The test allows 2
or 4 (`tests/test_v2_graph.py:52-57`).

**Storage.** Variants are stored flat as `var_block`, `var_cab` (R, 3) and `var_m` (R,), with
no padding (`graph.py:47-49`). `attach_rows` builds two row tables (`graph.py:246-261`):

- `(ev_edge, ev_var)`: one row per (Stage 3 edge, variant of its receiver);
- `(av_atom, av_var)`: one row per (atom, variant of its block).

**Conditioning.** a and b are both bonded to the non-linear atom c, so the angle a-c-b is
normally 90-125°. `min_frame_sin` reports the worst sin(a-c-b) (`graph.py:268-282`). Every
test molecule must have sin > 0.5 (`tests/test_v2_graph.py:90-102`).

### 3.5 Frame-free blocks

A block with no eligible anchor (a monatomic ion, CO2, H2) gets one row with cab = -1 and
m = 0 (`graph.py:190-194`). Its frame is set to the zero matrix (`model.py:118`), so:

- its frame coordinates are 0 (Stage 2 sees distances and types only);
- its P1 and P2 projections are 0 (Stage 4 sees M0 only).

Other blocks still see its atoms in *their* frames.

**Numerical guard.** Frame-free rows compute frames from zero vectors before the
`torch.where` masks them, and `_normalize` adds eps = 1e-12 under the square root
(`model.py:48-50`). A gate checks that a float32 double-backward through these rows stays
finite. Inf × 0 would give NaN and poison every weight (`tests/test_v2_gates.py:204-217`).

### 3.6 Stage 2: a ragged, shared encoder (`v2/model.py:126-168`)

**Atom type embedding u** (`model.py:126-131`), 32 dimensions:

```
u_i = Linear([z_emb(Z_i) (16), onehot(degree≤6) (7), ring (1), onehot(n_ext≤4) (5)])
      + Linear([Q, S-1])          charge/spin conditioning, same for every atom of a structure
```

**Ragged bookkeeping** (`model.py:143-147`):
- `block_atoms` is an integer (B, n_max) table padded with -1.
- `torch.nonzero(mask)` lists the real (block, slot) pairs.
- `relA[i] = r_i - o_block(i)`.

No *feature* tensor is padded to the largest block in the batch.

**The encoder, for each block:**

```
pair term (variant-free):   for every ordered pair (k, l) in the block
    pin = [gaussians(|r_k - r_l|; 16 on 0.8-7 Å), u_k, u_l]            (model.py:153-158)
    P_k = Σ_l pair_net(pin)                     -> (N, 64)              (model.py:159)
per (atom, variant) row:
    coords = F_v^T (r_k - o)                    (A, 3)                  (model.py:161-162)
    x      = atom_net([coords, u_k, P_k])       (A, 64)                 (model.py:163)
per variant:
    Hv     = block_net(Σ_{k in block} x)        (R, 128)                (model.py:164)
averages:
    H_I    = mean_v Hv                          (B, 128)                (model.py:165)
    h_k    = mean over k's variant rows of x    (N, 64)                 (model.py:166)
```

- **Atom labelling inside a block** is handled by the sums (Σ_l, Σ_k), which are order-free.
- **Frame choice** is handled by the variant mean.
- **One encoder for all block types**, so a new monomer needs no new weights.
- **Implementation detail.** The spec wrote "atom input + Σ pair", but the code concatenates
  `[coords, u, P]` rather than adding.

**Profiling note** (`model.py:35-38`). Every gather uses `index_select`. The plain
`x[index]` form has a sort-based backward that took about 25% of CPU step time.

### 3.7 Stage 3: messages that see J's atoms in I's frame (`v2/model.py:170-205`)

For every directed block edge J → I and every variant v of I (the `ev` rows):

```
dO     = o_J - o_I (minimum image)                                     (model.py:177-179)
cJ     = F_v^T (r_a - o_I)  for every atom a of J                       (model.py:184)
g      = Σ_{a in J} geo_net([cJ, u_a])            (rows, 32)            (model.py:185-186)
pooled = Σ_{a in I, b in J} pz(Z_a)⊙pz(Z_b) ⊗ gaussians(r_ab; 16 on 1-8 Å) / 4  (Ed, 64)
                                                                        (model.py:187-196)
efeat  = [pooled, onehot(hop)]                    (Ed, 66)              (model.py:197)
msg    = MLP([H_I, H_J, g, efeat])               per (edge, variant) row
m_IJ   = mean over I's variants                   (Ed, 128)             (model.py:203)
H_I   += upd([H_I, Σ_J m_IJ])                     3 layers              (model.py:199-205)
```

- **Block graph** (`graph.py:203-223`):
  - a hop-1 edge joins two blocks that share a covalent bond;
  - a hop-2 edge joins two blocks that share a neighbour block and are not already hop-1;
  - both directions are stored.
- **Why there is no R_IJ.** The positions of J's atoms in I's frame already encode J's
  orientation relative to I. J's frame never enters, so only I's variants need averaging,
  and v1's same-variant pairing trick is unnecessary. The vectors enter raw, which satisfies
  invariant 4.
- **Species-pair pooling.** v1's fixed CC/CH/HH classes become a learned symmetric
  embedding product pz(Z_a)⊙pz(Z_b), which works for any element.
- **No cutoff envelope on these Gaussians.** The pair set is fixed by topology (`model.py:53-55`).
- **A bug this code was written to avoid** (`model.py:171-173`). The update MLP runs for
  every block, even when the batch has no edges. Otherwise a block with no neighbours
  (benzene alone) would get a different H depending on whether other structures in the batch
  had edges. Section 6 covers it.

### 3.8 Valence prior with type-conditioned pairs (`v2/model.py:207-217`)

- **Pairs.** All 1-2, 1-3 and 1-4 pairs from a BFS with cutoff 3 (`graph.py:225-230`).
- **Inputs:**
  - 48 Gaussians on 0.8-4.4 Å;
  - a one-hot of the topological distance d (3);
  - a symmetric pair code [ũ_i + ũ_j, ũ_i ⊙ ũ_j], where ũ = Linear(u) has 8 dimensions.
- **Network.** MLP 67 → 64 → 64 → 1, with half of each pair energy to each atom.
- **Why condition on types instead of bond orders.** u contains Z, degree, ring flag and
  n_ext, so a C=O (O of degree 1) and a C-O (O of degree 2) get different learned minima
  without any bond-order perception.

### 3.9 Stage 4: per-(atom, variant) projection (`v2/model.py:219-240`)

- **Neighbour context.** x_atom = [h_atom (64), H_I (128)] = 192 dimensions
  (`model.py:220`).
- **Radial weights.** φ_ij = radial(bessel(r) ⊗ pair-embedding) ⊙ gate(x_j), with 16
  channels.
  - The pair embedding is nb_pair(Z_i) ⊙ nb_pair(Z_j), 8 dimensions.
  - `radial` is bias-free (`model.py:81-84`), so every edge term is exactly zero at the cutoff,
    where the Bessel envelope is zero.
- **Moments.** M0, M1 and M2 are computed once per atom, as in v1, and scaled by 1/avg_nbr
  (`model.py:229-234`).
- **Projection, per (atom, variant) row** (`model.py:235-237`): P1 = F_v^T M1 and
  P2 = triu(F_v^T M2 F_v), giving A (rows, 160).
- **Products, then the mean** (`model.py:238-240`):
  `Bv = [lin1 A, lin2a A ⊙ lin2b A, lin3a A ⊙ lin3b A ⊙ lin3c A]` (rows, 192), then
  `Bn = mean over the atom's variant rows` (N, 192).

**Why the products come before the mean.** The mirror partner of a variant has P1 and P2
components along e3 with the opposite sign. A mean of A would cancel them, and every
orientation detail along the mirror axis would be lost. A product such as (e3 component)²
is mirror-even, so it survives the mean.

**Exclusion.** v2 keeps every pair; the exclusion mask was dropped, following v1's ablation.

### 3.10 Stage 5a: latent-charge Coulomb (`v2/longrange.py`)

**Charges** (`model.py:254`, `longrange.py:31-35`):
- q_i = 0.1 · q_net(final features).
- q_i is then shifted uniformly per structure so that Σq_i = Q, the structure's total charge
  (`neutralize`).
- **Charges are latent:** no reference charges are used, and they are learned only through
  energies and forces.

**Kernel:** φ(r) = erf(r/σ)/r, with σ = 1 Å (`model.py:96`). This is the Coulomb energy of
Gaussian-smeared charges. It is finite as r → 0 (it tends to 2/(σ√π)), so there is no
singularity.

**Clusters** (`longrange.py:44-52`): E = k_e Σ_{i<j} q_i q_j erf(r_ij/σ)/r_ij over all pairs
in the structure, with k_e = 14.399645 eV·Å (`longrange.py:22`). There is no cutoff, and the
cost is O(N²).

**Periodic: Ewald** (`longrange.py:55-95`). The idea is to split the kernel into a short-range
piece, summed in real space, and a smooth long-range piece, summed in reciprocal space:

```
erf(r/σ)/r = [erf(r/σ) - erf(αr)]/r      +      erf(αr)/r
              short-ranged (real space)          smooth (k-space)
```

- **α.** By default α = sqrt(-ln 1e-6)/r_c ≈ 0.372 Å⁻¹ for r_c = 10 Å (`longrange.py:57`).
- **Real space** (`longrange.py:61-62`). The real-space term is multiplied by a C² switch
  from 9 to 10 Å, so the energy stays smooth in the positions (invariant 6). The switch
  changes the energy only where erfc(αr)/r is about 2×10⁻⁷ per unit charge pair, which is
  negligible.
- **Reciprocal space** (`longrange.py:64-90`): (2π/V) Σ_{k≠0} exp(-k²/4α²)/k² · |S(k)|², with
  structure factor S(k) = Σ q_i e^{ik·r_i}. Only half the k-space is enumerated, and the result
  is doubled.
- **Self term** (`longrange.py:91`): -α/√π · Σq². It cancels each charge's interaction with its
  own smooth Gaussian.
- **Background term** (`longrange.py:92-93`): -πQ²/(2Vα²), for a charged cell
  (neutralising background).
- **Boundary conditions:** tin foil.

**Gates** (`tests/test_v2_longrange.py`):

| Test | Lines | Checks |
|---|---|---|
| Analytic | 31-37 | erf Coulomb of 2 charges equals the analytic value to 1e-12 |
| α independence | 53-56 | α = 0.35 and α = 0.45 agree to 1e-6 eV |
| Cluster limit | 59-69 | Ewald minus the dipole term 2π\|M\|²/3V converges to the cluster value, with residual × L⁵ constant (the right convergence order) |
| Charge conservation | 72-77 | Σq = Q to 1e-10 for Q ∈ {-1, 0, +1} |

### 3.11 Stage 5b: dispersion tail (`longrange.py:98-105`)

```
E_disp = -Σ_{i<j} sqrt(C6_i C6_j) / (r^6 + R0_ij^6) · S(r)
C6_i   = 10 · softplus(c6_net(features))               environment-dependent  (model.py:257)
R0_ij  = r0_i + r0_j,   r0 = softplus(per-element parameter), initialised to 1.7 Å   (model.py:95, 258)
S      = C² switch from 8 Å to 10 Å
```

- **Neighbour list.** A second list at 10 Å, rebuilt every call (`model.py:249-251`).
- **Why it exists.** It covers the attraction beyond Stage 4's 4.5 Å. The R0⁶ term damps the
  1/r⁶ divergence at short range.
- **Gates:**
  - E < 0 inside the switch and exactly 0 beyond 10 Å (`tests/test_v2_longrange.py:80-87`);
  - the full-model energy is smooth through the 8-10 Å switch
    (`tests/test_v2_gates.py:138-151`).

### 3.12 Charge and spin conditioning

- **How it enters.** (Q, S-1) for each structure goes through `qs_lin` and is added to every
  atom's u (`model.py:63, 125, 131`). It therefore reaches every stage that uses u. Q also
  fixes Σq in Stage 5.
- **Defaults.** `collate_v2` defaults to Q = 0 and S = 1 (`v2/batch.py:122-123`).
- **What the data supplies.** The data loader reads multiplicity from the dataset, but
  charge is not stored in the hydrocarbon data, so Q is 0 (`v2/data.py:75-76`).
- **Status.** The conditioning is live but **untrained**: hydrocarbon training never varies
  it.
- **Gate.** Q = +1, S = 2 changes E by more than 1e-6, and E stays rotation- and
  permutation-invariant (`tests/test_v2_gates.py:106-112`).

### 3.13 Readout (`v2/model.py:242-261`)

```
xf  = [x_atom (192), Bn (192), z_small(Z) (16)]                     (N, 400)
eps = (readout(xf) + val_atom) · scale                               (N,)
E   = index_add over structures of (eps.double() + E0[Z])            float64
    + E_es + E_disp
```

- **One readout, not four.** v2 averages B before the readout, so it runs once per atom.
  v1 instead ran the readout once per variant and averaged the outputs.
- **E0.** An int64 micro-eV buffer of length 100 (`model.py:99`). It is fitted per element by
  least squares (`v2/data.py:136-144`), and elements absent from training get 0.
- **Parameters:** 673 136 (v2) against 656 274 (v1), as measured by instantiating both
  classes.

### 3.14 Block registry: seen, similar or novel (`v2/registry.py`)

- **Diagnostic only.** The model never reads the registry; reading it would be "sharing
  values", which invariant 2 forbids.
- **Three levels of block identity**, each computed inside the block only
  (`registry.py:36-57`):
  - **L0** (composition): a multiset of (Z, sorted Z of external neighbours);
  - **L1:** one WL round over internal bonds;
  - **L2:** WL refined to stability.
- **Build** (`registry.py:86-111`), from the training set after training (`v2/train.py:174`):
  - a count per hash at each level;
  - for each L2 class, a readable label and the mean of the block embedding H_I from
    Stage 2, which is invariant because it is a variant average;
  - the spread: the median within-class RMS distance;
  - the threshold: the 95th percentile of within-class distances, in units of the spread.
- **Verdict** (`registry.py:128-133`):
  - `seen` when L2 matches;
  - `similar` when L1 matches, or when the distance to the nearest class mean is under the
    threshold;
  - otherwise `novel`.
- **Where it surfaces.** The calculator warns once per new topology about novel blocks
  (`v2/calculator.py:37-42`).
- **Example** (`tests/test_v2_registry.py:35-52`):
  - a registry built from PE gives `seen` for every block of a PE chain;
  - isobutane gives 3 `seen` (CH3) and one non-seen CH;
  - benzene matches at no level.

### 3.15 Data and training (`v2/data.py`, `v2/train.py`)

- **Hackathon config.** `{"elements": {H, C}, "n_extra": 1500}` (`data.py:29-32`):
  - every frame of v1's PE systems;
  - plus whole systems drawn at random until at least 1500 more frames;
  - the `all` config takes everything.
- **Dropped frames:**
  - `reactivity/` frames are dropped (`data.py:54`);
  - frames whose worst frame variant has sin(a-c-b) < 0.3 are dropped (`data.py:27, 86-88`).
- **Split** (`data.py:124-133`):
  - v1's PE systems keep their v1 split, so v1's PE test systems stay held out;
  - every other system is split 80/10/10 by system.
- **Training** (`train.py:76-180`):
  - the same loss as v1;
  - default lr 2e-3, at most 2000 atoms per batch;
  - EMA 0.99, gradient clipping at 10;
  - non-finite steps are skipped and counted (`train.py:142-146`);
  - `avg_nbr` is measured on the first 50 training frames (`train.py:111-116`);
  - outputs: `best.pt`, `registry.json`, and results on val, test and v1's PE test frames.

---

## 4. Worked examples: isobutane and toluene

Isobutane, (CH3)3CH, is in the test set (`tests/v2_molecules.py:11`). All counts below were
produced by running `topology_from_geometry_v2` on the ASE g2 geometry. Indices are ASE's:
atom 0 is the central C, atom 1 its H, and atoms 2, 6 and 10 are the methyl carbons.

### 4.1 Stage 0

- **Size:** N = 14 atoms, 13 bonds.
- **Degrees:** central C 4, methyl C 4, every H 1. No ring atoms.
- **Blocks.** Each carbon is non-terminal, so each starts its own block. The H join their
  carbon.

```
 block 0  CH[C3]   atoms {0, 1}            n_ext(C0) = 3
 block 1  CH3[C]   atoms {2, 3, 4, 5}      n_ext(C2) = 1
 block 2  CH3[C]   atoms {6, 7, 8, 9}
 block 3  CH3[C]   atoms {10, 11, 12, 13}
 block_atoms: (4, 4), with row 0 padded by -1
```

- **WL colours:** 4 classes (central C, central H, methyl C, methyl H).
- **Block graph:**
  - hop-1: CH–CH3 × 3, so 6 directed edges;
  - hop-2: CH3–CH3 pairs that share the CH, so 3 pairs and 6 directed edges;
  - Ed = 12.
- **Valence pairs:** 64 in all, with 13 at d = 1, 24 at d = 2 and 27 at d = 3.

### 4.2 Stage 1: variants

| Block | c | N(c) classes | a | b | Variants |
|---|---|---|---|---|---|
| CH | C0 | {H1} size 1, {C2, C6, C10} size 3 | H1 | C2, C6 or C10 | 1·1·3·2 = **6** |
| each CH3 | its C | {C0} size 1, {3 H} size 3 | C0 | each of its 3 H | 1·1·3·2 = **6** |

- **R = 24** variant rows in total.
- **Why 6 for the CH.** Its true symmetry is C3v, which has order 6 and includes mirrors, and
  the variant set is exactly an orbit of that group.
- **Frame conditioning.** The smallest sin(a-c-b) is 0.93.

### 4.3 Tensor shapes through the forward pass

| Quantity | Shape | Where |
|---|---|---|
| positions | (14, 3) | input |
| u (type embedding) | (14, 32) | `model.py:126-131` |
| origins o, frames F | (4, 3), (24, 3, 3) | `model.py:140-141` |
| intra-block ordered pairs | 38 = 1·2 + 3·12 | `model.py:153` |
| pair features → P | (38, 64) → (14, 64) | `model.py:158-159` |
| (atom, variant) rows `av` | 84 = 2·6 + 3·(4·6) | `graph.py:256-259` |
| coords F^T(r-o), x | (84, 3), (84, 64) | `model.py:162-163` |
| Hv → H (block) | (24, 128) → (4, 128) | `model.py:164-165` |
| h_atom | (14, 64) | `model.py:166` |
| Stage 3 (edge, variant) rows `ev` | 72 = 12 edges · 6 | `graph.py:253-255` |
| (edge-variant, J-atom) rows → g | 252 → (72, 32) | `model.py:182-186` |
| cross-block atom pairs → pooled | 144 → (12, 64) | `model.py:187-196` |
| message per row → m_IJ → H | (72, 128) → (12, 128) → (4, 128) | `model.py:202-205` |
| valence pair energies | (64,) → (14,) | `model.py:214-217` |
| radius edges at 4.5 Å | 182 (directed) | `model.py:221` |
| M0, M1, M2 | (14, 16), (14, 16, 3), (14, 16, 3, 3) | `model.py:230-234` |
| A per (atom, variant) | (84, 160) | `model.py:237` |
| Bv → Bn | (84, 192) → (14, 192) | `model.py:238-240` |
| xf → eps | (14, 400) → (14,) | `model.py:243-244` |
| E | (1,), float64 | `model.py:246-261` |

Stage 5 for a cluster:
- the Coulomb term runs over the 91 = C(14, 2) pairs;
- the dispersion term uses the 10 Å list, which for a 14-atom molecule is every pair, twice.

### 4.4 Toluene, for contrast

- **Size:** N = 15.
- **Blocks:** 2, namely `C6H5[C]` (11 atoms) and `CH3[C]` (4 atoms), so block_atoms is
  (2, 11).
- **Phenyl variants: 4.**
  - The anchor class is the smallest ring-C class. Both the ipso and the para carbon form
    size-1 classes, and the colour hash broke the tie toward **para** in this run, so c is
    the para C.
  - a is its H (a size-1 class); b is either meta C (a class of 2).
  - That gives 1·1·2·2 = 4.
  - The spec said "ipso anchor". Either class gives an exact average.
- **Methyl variants:** 6.
- **Totals:**
  - R = 10 variant rows, Ed = 2 (one hop-1 pair), `ev` = 10, `av` = 11·4 + 4·6 = 68;
  - intra-block ordered pairs 122, cross-block pairs 88, valence pairs 66.
- **Cost.** A 5-atom phenyl costs 4 variants, while benzene alone costs 24, because the
  substituent breaks the ring's symmetry. The cost of a block follows its symmetry.

---

## 5. Correctness gates

All gates run in float64 (`tests/conftest.py`) on an **untrained, randomly initialised**
model. A symmetry bug then shows at order 1 and cannot hide behind training. Finite-difference
tests use E0 = 0, because a large constant offset costs float64 digits in E(r+h) - E(r-h)
while contributing zero gradient.

### 5.1 v1 (`tests/test_gates.py`, 21 tests)

| Gate | Lines | Catches |
|---|---|---|
| Block decomposition | 57-65 | wrong cut; an atom in no block or in two |
| Radius graph, vesin vs 27-image brute force; multi-structure batch | 68-89, 177-187 | missing or extra neighbours; wrong shifts; edges between different structures |
| Rotation + translation, cluster and periodic, \|ΔE\| < 1e-9 | 92-101 | any non-invariant feature, such as F instead of F^T |
| Permutation (all atoms), cluster and periodic | 104-113 | label-dependent slots or chain direction |
| Wrap by lattice vectors | 116-119 | missing minimum-image corrections |
| Autodiff vs central finite differences, h = 1e-5, < 1e-8 eV/Å | 122-129 | non-differentiable ops; detached tensors; cached values |
| Cutoff smoothness: pull two chains apart | 132-148 | a jump at r_cut (second difference spikes); E not exactly constant beyond the cutoff |
| **Negative control** | 151-162 | checkers that cannot fail: a rotation-dependent and a label-dependent fake energy must be flagged |
| "Oracle" (mean over 8 relabellings) | 165-174 | weak, admitted in the docstring: just a second permutation check |
| All-pairs model, same suite | 197-210 | the trained configuration (`exclude_hops = -1`) |
| **Reflection** | 213-222 | SO(3)-only invariance (the pseudovector bug) |
| float32 vs float64, output dtype float64 | 225-239 | E0 rounding; a total not summed in float64 |
| Butane (2 end blocks, no mid) | 242-248 | edge case: no mid blocks, no hop-2 edges |
| Batch equals single | 251-261 | cross-talk between structures in a batch |
| Calculator equals model | 264-276 | the ASE wrapper diverging from the model |

### 5.2 v2 (`tests/test_v2_*.py`)

The symmetry checkers are v1's (`gates.py`); they take any energy function. They run on
hydrocarbons, on **heteroatom fragments never trained on** (PEO, nylon-like amide, ester,
C-Cl, CF4, pyridine, thiophene, nitrile, CO2, water: `tests/v2_molecules.py:14-15`), on a
PEO chain with an ion pair, and on a periodic PE melt.

| Gate | Lines in `test_v2_gates.py` | Catches |
|---|---|---|
| Rotation, permutation, mirror on 4 cluster systems | 65-71 | a variant set not closed under the symmetry |
| Periodic rotation, permutation, mirror; wrap | 74-84 | block or variant atoms straddling the boundary |
| Finite differences, cluster and periodic | 87-95 | non-conservative paths, including Ewald and dispersion |
| Frame-free and linear blocks (CO2, CH3CN, ions, 2-butyne) | 98-103 | NaN or crashes from zero frames |
| Charged input | 106-112 | conditioning that is dead, or that breaks symmetry |
| **Variant-set oracle (a true oracle)** | 115-127 | see 5.3 |
| **Negative control: drop one variant per block** | 130-135 | the rotation or mirror gate must then fail, which proves the gates can see a broken average |
| Smoothness at 4.5 Å and through the 8-10 Å switch | 138-151 | a hard cutoff anywhere |
| Batch equals single (benzene + hetero; periodic) | 154-166 | the isolated-block update bug (Section 6) |
| float32 vs float64 | 169-177 | E0 rounding |
| **Oracle can fail** | 180-186 | an oracle that cannot fail: moving one atom by 0.01 Å must be detected |
| Calculator equals model | 189-201 | wrapper drift |
| float32 double-backward is finite | 204-217 | NaN from zero frame vectors under force training |

Graph gates (`test_v2_graph.py`):
- bonds match v1 (27-44);
- PE blocks are CH2/CH3 with 2, 4 or 6 variants (47-57);
- benzene is 1 block with 24 variants (60-63);
- the terminal and linear rules (66-75);
- every molecule's frames have sin > 0.5 (90-102);
- the row tables are consistent (105-113);
- rejections (123-132);
- batch offsets (135-148);
- `min_frame_sin` flags a collinear H (151-161).

### 5.3 The oracle and why it is "true"

v1's "oracle" only averaged E over relabellings. If the model is invariant, all those
energies are identical, so the test adds nothing. v2's oracle (`v2/gates.py:55-100`) checks
the *construction* directly, independently of the network:

1. For every block, list the variants. For each variant, compute an (n, 4) signature of
   (Z, F_v^T(r - o)) for the block's atoms, using numpy rather than the model code.
2. Apply a random relabelling, a rotation and a mirror, and rebuild everything from scratch.
3. Match the two sets of signatures per block by **optimal assignment** (Hungarian
   algorithm), over both atoms and variants. They must agree to 1e-9.

If the multiset of encoder inputs is identical, any function averaged over it is identical.
So this proves closure of the variant set for any weights.

Two controls keep it honest:
- the **oracle-can-fail** test checks that it detects a real 0.01 Å change;
- the **dropped-variant** negative control checks that the energy gates detect a broken
  variant set.

---

## 6. Bugs found during the build, and what each taught

The training loss looked clean through almost all of these. That is the central lesson:
**force MAE does not detect symmetry or smoothness bugs; gates do.**

| Bug | How it showed | Caught by | Lesson |
|---|---|---|---|
| Stage 4 projected the quadrupole as F M2 F instead of F^T M2 F | E changed under rotation, for periodic boxes only | periodic rotation gate | Test periodic and cluster separately; one may mask the other. |
| SO(3)- but not O(3)-invariant: e3 is a pseudovector | gauche+ ≠ gauche-; a mirrored cluster changed E by 6.3e-4 eV | code review, then the new reflection gate | A rotation test cannot see a mirror bug (det +1). The fix is mirror variants. |
| float32 rounding of E0 (about 0.06 eV per 1000 atoms) | would swamp NVE drift | code review | Store offsets as int64 micro-eV and sum in float64. |
| `torch.cdist` has no double backward | force training crashed | first training run | Force training needs every op twice-differentiable; write distances by hand (`model.py:167`). |
| FD gate failed at h = 1e-4 | O(h²) truncation on the stiff valence Gaussians | diagnosed by shrinking h | Not every failing gate is a model bug; measure the truncation error (`tests/test_gates.py:34-37`). |
| Evaluation silently ran in float32 | `collate` followed the process default dtype | code review | Make dtype explicit at every boundary. |
| `np.mod(-1e-20, L) == L` | KD-tree crash at the box edge | code review | `x[x >= L] = 0` guard (`topology.py:92`). |
| Mirrored cell diag(-L, L, L) | KD-tree assumed a positive diagonal | reflection gate | A test can exercise code paths you did not think about (`topology.py:90`). |
| Calculator reused topology when only the atom *count* matched | silent wrong energies for a different molecule of the same size | code review | Compare the actual atomic numbers (`calculator.py:30`). |
| A rerun overwrote `best.pt` with epoch-0 weights | final evaluation looked untrained | noticed in results | Refuse a non-empty `--out` (`train.py:87-90`). |
| **Batch ≠ single in v2 Stage 3** | benzene alone vs benzene batched with an edged molecule gave different E | `test_batch_equals_single` (`tests/test_v2_gates.py:154-166`) | The update MLP ran only when the batch had edges, so an edgeless block's H depended on its batch-mates. Batching is a symmetry too. Fix: always run `upd` (`v2/model.py:171-173, 199-205`). |
| **Singular frame in training data** | one CH3 with an H at 180° to its C-C bond had sin(a-c-b) = 0; e2 was undefined, forces of about 1e4 eV/Å appeared once per epoch, and training diverged | traced from the loss spikes | Curation: drop frames with min sin < 0.3; every other frame has ≥ 0.64 (`v2/data.py:22-27, 86-88`). Training now skips and counts non-finite steps (`v2/train.py:142-146`), and the default lr dropped to 2e-3. A real geometric weakness; see Section 7. |
| **Registry changed the caller's model dtype** | `nn.Module.to()` is in place, so building a report cast the calculator's float32 model to float64 | `test_registry_report_leaves_model_dtype_alone` (`tests/test_v2_registry.py:75-88`) | Diagnostics must not mutate their inputs. Fix: `copy.deepcopy` first (`v2/registry.py:75`). |
| **The oracle itself mis-paired near-ties** | sorting signatures by rounded keys mis-paired nearly equal rows | developing the oracle | Match by optimal assignment, not sort (`v2/gates.py:58-59, 83-100`). Test your tests (`test_oracle_can_fail`). |
| **float32 E0 artifact in MACE** | MACE's float32 NVE trace had std 0.46 meV/atom; 0.33 of it was float32-vs-float64 rounding of E0 sums at about -3.4e5 eV | applying the v1 lesson to the baseline | The float32 NVE comparison overstated our advantage. Reported honestly; the like-for-like float64 run did not finish (`docs/writeup.md:407-420`). |
| **Bond rules disagree on noisy geometry** | v1 and v2 bonds differed on a noisy cluster (C-H 1.323 Å) | `tests/test_v2_graph.py:30-32` | Thresholds are a modelling choice, so document where they diverge. |
| **The v1 "oracle" could not fail** | it averaged E over relabellings of an already-invariant model | self-review (`tests/test_gates.py:165-169`) | Led to the v2 variant-set oracle and its negative control. |

---

## 7. Honest limitations and open problems

### 7.1 Frame degeneracy

**The weakness.** The v2 frame for a row is singular when a, c and b become collinear
(sin(a-c-b) → 0). Near that point e2 = unit(b - c projected) rotates very fast, so frame
coordinates, and therefore forces, have gradients of order 1/sin. v1 was protected by
construction: the CH2 hydrogens are about 120° apart around the C-C axis, and the measured
minimum |s| was 0.58. v2's general rule is not protected.

**Today's handling is curation.** Training frames with min sin < 0.3 are dropped, which is
one frame. At inference nothing protects an MD trajectory that wanders into such a geometry.

**Proposed fix: sin²-weighted variant averaging.** Replace the plain mean over variants with

```
f_block = Σ_v w_v f_v / Σ_v w_v,     w_v = sin²(angle a_v - c_v - b_v)
```

- **Still exactly invariant.** w_v depends only on distances and angles, which are invariant,
  and the variant set is still closed. The weighted mean is therefore still exactly O(3)- and
  permutation-invariant.
- **Smooth.** w_v is smooth and polynomial in the coordinates.
- **The degenerate variant fades out.** Its weight goes to zero quadratically while its
  gradient blows up only as 1/sin, so the product w·f stays bounded with a bounded gradient.

**Open issue.** When *every* variant of a block degenerates at once, the denominator tends to
0. Examples: a CH3 whose three H all fold onto the C-C axis, or a block with a single variant
pair. A smooth floor (adding ε) or a frame-free fallback is needed there. Neither is
implemented.

### 7.2 Other limitations

**Data:**
- **Clusters only.** OPoly26 frames are carved, H-capped clusters of at most about 360 atoms.
  So the model has never seen periodic DFT, a long-range electrostatic environment, or bulk
  density.
- **Latent charges and C6 are unsupervised.** They are trained only through cluster energies
  and forces.
- **Charge is not stored in the hydrocarbon data,** so Q = 0 always, and charge/spin
  conditioning is untrained.
- **v1 data is small:** 631 PE frames from 326 systems.

**MD:**
- **No stress, so no NPT.** The calculator implements only energy and forces
  (`v2/calculator.py:18`). Density, R_g and T_g, the observables the project claims to care
  about, are not measured.
- **NVE ran 3 ps, not the specified 100 ps** (v1). Rare frame discontinuities could take
  longer to show.
- **The v1 NVT melt ran at 0.50 g/cm³,** not the real 0.78 g/cm³.

**Speed (measured on v1):**
- **Launch-bound.** 27× more atoms cost only 3.3× more time. Python and kernel launches
  dominate at 1k atoms, so CUDA graphs or `torch.compile` would help.
- **The neighbour list is only 2-5% of a step.**
- **CPU topology build.** Graph construction (networkx, Python loops in `attach_rows`) runs on
  the CPU, once per system.
- **Per-structure Python loops** in Ewald (`longrange.py:66`) and in vesin
  (`batch.py:208`).
- **O(N²) cluster Coulomb.**

**Comparison fairness:**
- MACE ran without cuEquivariance and with a 5.0 Å cutoff (ours: 4.5 Å).
- The two models shared the GPU.
- Single seed each, so no error bars.

**Topology:**
- **Non-reactive.** Bonds are read once. The calculator rebuilds topology only when the
  atomic-number array changes (`v2/calculator.py:35`). So it would silently keep a stale bond
  graph for an isomer with the same atom ordering, or after a bond breaks.
- **A chain bonded to its own periodic image** (an "infinite" chain) makes every backbone
  bond part of a cycle. v2 would read it as a ring system: one huge block, rejected above 32
  atoms, or wrongly treated as a ring if small. This is still an open decision in
  `CLAUDE.md`.
- **The 32-atom cap rejects large fused aromatics.** Coronene, C24H12, has 36 atoms.
- **Acyclic double bonds and amide bonds are cut,** so their torsion barriers must be learned
  across blocks.
- **The WL cap of 20 rounds.** In chains longer than about 40 atoms, atoms far from the ends
  may share colours. That keeps the average exact but adds variants (the CH2 count can be 4
  instead of 2).

**Design open questions** (`CLAUDE.md`):
- **Serial vs separable energy.** Serial is implemented, so multiple-timestep integration is
  not possible.
- **The one-layer non-bonded bet,** at a 4.5 Å cutoff.
- **Determinism.** GPU scatter-add makes training non-bitwise-reproducible.

**Documentation drift.** The v1 docstrings at `model.py:5` and `model.py:195` still say "two
variants", but the code uses 4 (`model.py:96`).

---

## 8. Glossary

| Term | Meaning here |
|---|---|
| **MLIP** | Machine-learned interatomic potential: a neural network E(r, Z) fitted to DFT. |
| **DFT** | Density functional theory, the quantum reference. OPoly26 uses ωB97M-V/def2-TZVPD. |
| **Conservative force** | A force that is the gradient of a scalar energy, so energy is conserved in NVE. |
| **NVE / NVT / NPT** | MD ensembles: constant energy; constant temperature (thermostat); constant pressure (needs stress). |
| **Drift** | Slope of total energy against time in NVE. It should be about zero. |
| **O(3) / SO(3)** | Rotations plus reflections / rotations only. |
| **Invariant / equivariant** | Unchanged by the symmetry / transforms along with it. |
| **Pseudovector** | A cross product. It flips sign relative to a true vector under a mirror. |
| **Frame F** | 3×3 orthonormal matrix with columns e1, e2, e3, built from a block's atoms. |
| **Frame variant** | One choice of (anchor, e1 atom, e2 atom, mirror). Averaging over the closed set gives exact invariance. |
| **Frame-free block** | A block without a valid anchor (ion, CO2). F = 0 and it uses distances only. |
| **Reynolds operator / group averaging** | Averaging a function over a symmetry group to make it invariant. |
| **Block** | A rigid fragment: a v1 C2 unit; in v2 a ring system, a non-terminal atom with its terminal atoms, or a merged linear unit. |
| **Bond graph / radius graph** | Covalent bonds (fixed) / all pairs within a cutoff (rebuilt every call). |
| **Block graph** | Blocks as nodes. Hop-1 edges share a bond; hop-2 edges share a neighbour block. |
| **Bridge** | A graph edge whose removal disconnects the graph, so it lies on no cycle. Non-bridge bonds are ring bonds. |
| **WL colouring** | Weisfeiler-Lehman refinement: iteratively hash an atom's label with its neighbours' labels. |
| **Canonical colour class** | The smallest colour class in a set, ties broken by colour value. A set of atoms, never one atom. |
| **Anchor (c)** | The frame's origin atom for e1 and e2: a non-linear block atom with degree ≥ 2. |
| **Valence prior** | A learnable Urey-Bradley-like term on 1-2, 1-3 and 1-4 topological distances. |
| **1-n pair** | Two atoms n-1 bonds apart (1-2 bonded, 1-3 angle, 1-4 torsion). |
| **Cartesian moments M0/M1/M2** | Neighbour sums of a weight, the weight times the direction, and the weight times the outer product of directions. |
| **Body order** | How many atoms jointly enter one energy term. Products of neighbour sums raise it (the ACE trick). |
| **Bessel basis + envelope** | Radial functions sin(kπr/r_c)/r times a polynomial that goes C²-smoothly to 0 at r_c. |
| **Latent charges** | Per-atom charges predicted by the network, learned only through energies and forces. |
| **Ewald summation** | The periodic Coulomb sum, split into a real-space short-range part and a k-space long-range part. |
| **α, σ** | The Ewald splitting parameter; the Gaussian charge width (1 Å). |
| **Tin-foil boundary** | Ewald convention with no surface-dipole term. |
| **C6 / R0** | Dispersion coefficient / damping radius in -C6/(r⁶ + R0⁶). |
| **Switch function** | 1 - x³(10 - 15x + 6x²): a C² step from 1 to 0. |
| **E0** | Per-element reference energy (least-squares fit), stored as int64 micro-eV. |
| **EMA** | Exponential moving average of the weights (decay 0.99), used for evaluation and checkpoints. |
| **Gate** | A correctness test that must pass before any accuracy claim. |
| **Negative control** | A deliberately broken input that a gate must flag, proving the gate can fail. |
| **Oracle** | An independent construction of the right answer, compared against the implementation. |
| **Registry** | A training-coverage dictionary of block types (seen, similar, novel). Diagnostic only. |
| **OPoly26** | The polymer DFT dataset (Levine et al., arXiv 2512.23117); public mirror at ColabFit. |

---

## 9. Twenty likely judge questions, with crisp answers

1. **Why not just use MACE?**
   It is accurate but pays for symmetry with tensor products and rediscovers bonds every
   step. On the same PE data our v1 tied it on forces (13.71 vs 13.73 meV/Å MAE) and lost
   slightly on energy (0.33 vs 0.28 meV/atom). It was 2.3× faster at 1k atoms and needed
   about 12× less memory per atom.

2. **Are your forces conservative?**
   Yes. Forces are -dE/dr by autodiff (`model.py:472-478`), checked against central finite
   differences to 1e-8 eV/Å, and v1 NVE drift was 0.0003 meV/atom/ps over 3 ps.

3. **How is rotation invariance exact, not approximate?**
   Every geometric input is projected into frames built from the block's own atoms, and
   F^T v is invariant because (QF)^T(Qv) = F^T v. Gates check |ΔE| < 1e-9 eV under random
   rotations on an untrained model.

4. **Then why do you need to average over frames at all?**
   Building a frame involves arbitrary choices: which H, which direction, which handedness.
   Picking one would break permutation or mirror symmetry, or jump discontinuously.
   Averaging over a closed set of choices is exact and smooth.

5. **What does "closed set" mean, and how do you get it without enumerating automorphisms?**
   Any symmetry must map the set of variants onto itself. Variants are built from
   Weisfeiler-Lehman colour classes, which are graph invariants, so relabelling, rotation and
   mirror (m → -m) all map the set onto itself. The true oracle test verifies this by
   matching encoder inputs one to one.

6. **Isn't this just data augmentation?**
   No. Augmentation makes symmetry approximately learned. Here the forward pass itself is
   the average, so invariance holds to machine precision for any weights, including on
   chemistry never seen.

7. **What happens with mirror images and chirality?**
   Energy is a scalar, and physics gives mirror images equal energy, so the model is
   O(3)-invariant via the ±e3 variants. v1 originally missed this: gauche+ and gauche- had
   different energies, caught in review, and a reflection gate now pins it.

8. **What if a frame degenerates?**
   Then e2 is undefined. It happened once in training data: an H at 180° to a C-C bond,
   giving 1e4 eV/Å forces. We curate (min sin ≥ 0.3), and the proposed fix is sin²-weighted
   variant averaging, which is still exact and makes degenerate variants fade out smoothly.

9. **How many variants, and what does it cost?**
   It follows the block's symmetry: CH3 6, interior CH2 2-4, phenyl 4, benzene 24. Stage 4
   moments are built once per atom, and only a cheap 3×3 projection runs per variant, so
   averaging costs per atom, not per edge.

10. **Why fix the bond graph? What about reactions?**
    v1 and v2 are non-reactive by design: bonds are read once. A reactive version would need
    the block rule to change smoothly, which this architecture does not do.

11. **How does it generalise to a polymer it never saw?**
    One shared encoder for all block types, element embeddings, and a graph-only block rule.
    The untrained model already passes every gate on PEO, amide, ester, PVC, PTFE, pyridine
    and thiophene fragments. Accuracy on new chemistry is a data question, and the registry
    flags novel blocks at inference.

12. **What is body order here?**
    The per-atom neighbour sums A are 2-body. Elementwise products A·A and A·A·A are 3- and
    4-body, because a product of sums contains all neighbour pairs and triples at one sum's
    cost (the ACE trick), with no Clebsch-Gordan coefficients needed.

13. **Why did excluding bonded pairs from Stage 4 hurt?**
    The bonded channel sees only block summaries. The atom-resolved 1-5 and 1-6 H···H
    contacts that set gauche vs trans energies live in the radius graph. Keeping all pairs
    gave 34.4 vs 115.6 meV/Å test force MAE in 8-minute runs.

14. **Why store E0 in int64 micro-eV?**
    E0[C] ≈ -1037 eV, so a 1000-atom total of about -3×10^5 eV is quantised at about 1/32 eV
    in float32, which swamps drift. Integer offsets plus a float64 sum fix it. The same
    artifact was found in MACE's float32 NVE.

15. **Is the speed comparison fair?**
    Partly. Same GPU, same ASE interface, float32. But MACE ran without cuEquivariance, with
    a 5.0 Å vs 4.5 Å cutoff and a CPU neighbour list, and its 8k-atom number is dominated by
    paging. The robust claim is memory: about 1.6 MB vs 0.13 MB per atom.

16. **How do you prevent train/test leakage?**
    By splitting by MD system, never by frame. v1's PE test systems stay in v2's test split.

17. **How do the latent charges know what to be?**
    They don't, directly. They are an unsupervised latent, shifted to sum to Q and learned
    only through energies and forces. The gates verify charge conservation and Ewald
    correctness, not physical charge values.

18. **Is 3 ps of NVE enough?**
    No. The spec asks for 100 ps. 3 ps catches gross non-conservation; rare frame flips
    would need longer runs. This is the first thing to extend.

19. **Can you predict density or T_g?**
    Not yet. There is no stress output, so no NPT. The training data are clusters, and the
    NVT melt ran below the real density.

20. **What is the single biggest technical risk, and the next step?**
    Frame degeneracy under the general v2 block rule. Next: sin²-weighted variant averaging,
    a stress/virial output for NPT, periodic training data, and a 100 ps NVE gate.

Bonus questions:

- **Why WL colours rather than real automorphism groups?**
  WL is polynomial and never needs the group. If it over-merges atoms, the cost is extra
  variants, never a wrong answer.

- **Doesn't tie-breaking by colour value "select" something?**
  It selects a *class* by a graph property that never changes during MD, so it can never
  flip. Invariant 3 forbids geometric selection, not topological selection.
