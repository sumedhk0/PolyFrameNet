# v2: generalise PolyFrameNet from polyethylene to all hydrocarbon polymers

Status: design approved in chat (option 1, carbon-centred blocks), 2026-10-21. Deadline:
trained, benchmarked and written up by the afternoon of 2026-10-22, with a full
explanation by 6 pm.

## Intent

v1 handles only linear, saturated, even-length PE chains, and so used 631 of the 70k
C/H frames already downloaded. v2 is a single potential for any non-reactive hydrocarbon
polymer. That covers PE, PP, PS, PIB, polybutadiene, aromatics and branched chains, in
any fragment OPoly26 carved. It keeps v1's invariants: forces by autodiff, no caching,
topological frame selection, raw rotation matrices, symmetrise never select, smooth cutoff.

Success means all of the following:
1. All v1 correctness gates pass on v2, on molecules that exercise every carbon type,
   including aromatic rings, quaternary carbons and methane.
2. v2 and MACE are trained on the same small subsample of hydrocarbon frames with the
   same split, and their test accuracy is reported.
3. v2 is also evaluated on v1's PE test frames. This answers whether generality cost
   accuracy on PE.
4. Speed, memory and NVE are measured as for v1, on a PE melt and on one aromatic
   (polystyrene) system.

Out of scope: heteroatoms, charged systems, long-range electrostatics, reactive frames,
sp carbons (alkynes, allenes), stress/NPT.

## Data facts (scripts/survey_ch.py, all 70 735 C/H frames)

- 70 325 frames have a clean bond graph. In such a graph every H has one C neighbour and
  every C has 2-4 neighbours.
- Every frame has multiplicity 1. The charge is not stored.
- Carbon environments, written (C neighbours, H neighbours):

  | Environment | Count | Meaning |
  |---|---|---|
  | (2,1) | 1.97 M | aromatic CH or =CH- |
  | (2,2) | 1.36 M | CH2 |
  | (3,0) | 0.97 M | aromatic or substituted sp2 C |
  | (1,3) | 0.87 M | CH3 |
  | (3,1) | 0.30 M | CH |
  | (4,0) | 79 k | quaternary C |
  | (1,2) | 43 k | =CH2 |
  | (2,0), (1,1) | 10 k | sp carbon |
  | (0,4), (0,3), (0,2) | 6 k | methane and small fragments |

- Frames have a median of 179 atoms and a maximum of 357.
- Source paths group into MD systems with about 3 frames each.

## Block definition

- **A block is one carbon plus the hydrogens bonded to it.** Block type is the carbon's
  environment (nC, nH).
- **Allowed types:**
  - sp3 (4 neighbours): (4,0), (3,1), (2,2), (1,3), (0,4);
  - sp2 (3 neighbours): (3,0), (2,1), (1,2).
- **Rejected at topology time:**
  - any carbon with 2 neighbours (sp; the frame would be collinear);
  - (0,3) and (0,2) (charged or radical fragments);
  - an H that is not bonded to exactly one C;
  - an H-H bond.
- **The block graph is the C-C bond graph.** It is a general graph, so branches and rings
  are fine. Block edges are 1-hop (bonded carbons) and 2-hop (carbons sharing a
  neighbour).
- **Valence pairs.** They are found by BFS over the full atom graph, as in v1, at
  topological distances 1, 2 and 3.
- **OPoly26 path filter.** Frames under `reactivity/` are dropped (AFIR reaction paths).

## Stage 1: frames per block, enumerated over topological choices

For carbon c with bonded neighbours N(c):
- **e1 candidates.** The neighbours that are carbon, or all neighbours when c has none
  (methane).
- **Variant.** One variant is an ordered pair (a, b), with a an e1 candidate and b in
  N(c), b != a, combined with a mirror flag m in {+1, -1}.
- **Frame:**
  - e1 = unit(r_a - r_c);
  - e2 = unit(r_b - r_c minus its e1 component);
  - e3 = m (e1 x e2);
  - origin = r_c.
- **Why it never degenerates.** Any two bonds of an sp2 or sp3 carbon are 105-125 degrees
  apart, so sin(angle) is at least about 0.8. The gate measures the minimum on real data.
- **Variant counts.**
  - Per block: |e1 candidates| x (deg - 1) x 2.
  - By type: CH3 = 6, CH2 = 12, aromatic CH = 8, aromatic C = 12, quaternary C = 24,
    methane = 24.
  - Hydrogens share their carbon's variants.
- **Invariance.**
  - The variant set is mapped onto itself by every rotation, reflection and relabelling
    of equivalent atoms.
  - So any quantity averaged over a block's full variant set is O(3)- and
    permutation-invariant.
  - Variants are chosen by topology only, so the average is smooth in the positions.
  - This is v1's fwd/rev/mirror average, generalised.
- **Storage.**
  - Variants are stored flat: `var_block`, `var_a`, `var_b` and `var_m`, one row per
    variant.
  - Means are taken with a scatter-mean over `var_block`.
  - No padding.

## Stage 2: block encoder (deep sets, no hand-written groups)

For each variant v of block I:
- **Input:**
  - the type embedding of I;
  - Σ over hydrogens h of φ(F_v^T (r_h - r_c));
  - Σ over carbon neighbours j of φ_C(F_v^T (r_j - r_c), type_j).
- **Why no group tables are needed.** The sums make the input invariant to hydrogen and
  neighbour labelling. v1's 16- and 24-element tables go away.
- **Block feature.** H_I = mean over v of MLP(input_v).
- **Per-atom features.** For hydrogens, h_i = mean over v of MLP_H(F_v^T(r_h - r_c),
  H-pre). Carbons use H_I.

## Stage 3: bonded channel, 3 layers on the block graph

For an edge J -> I (1-hop or 2-hop):
- **Message per variant v of I:**
  - MLP(H_I, H_J, Σ over atoms a in J of ψ(F_v^T(r_a - r_c,I), species_a), pooled pair
    Gaussians, hop one-hot).
  - The sum over J's atoms encodes J's orientation in I's frame. It replaces R_IJ, which
    would need J's frame too: averaging over the product of both variant sets costs up
    to 24 x 24.
  - Raw 3-vectors enter; nothing is compressed.
- **Edge message.** m_IJ = mean over v.
- **Update.** H_I += MLP(H_I, Σ_J m_IJ). The sum is over the edges into I.

## Valence prior

As in v1: an MLP over 48 Gaussians of r_ij for every 1-2, 1-3 and 1-4 pair. The input
also gets a symmetric encoding of the two endpoint block types, (t_i + t_j, t_i * t_j).
That lets C-C single (1.54 Å), aromatic (1.39 Å) and double (1.34 Å) bonds have
different minima. In v1 the class was only the species pair.

## Stage 4: non-bonded, one pass

- **Moments.** Global Cartesian moments M0, M1, M2 per atom, as in v1, cutoff 4.5 Å, all
  pairs kept (v1's ablation).
- **Projection per variant.** F_v^T M1 and F_v^T M2 F_v.
- **Body order per variant.** The 2-, 3- and 4-body products B_v.
- **Average.** B = mean over v. The products are nonlinear, so averaging after them keeps
  orientation information. Averaging the raw projections first would not.

## Readout

- **Per-atom energy.** eps_i = MLP([h_i, H'_I, B_i, species, type]).
- **Total energy.** E = Σ (eps_i + valence_i) · scale + Σ E0[Z]. E0 is stored as int64
  micro-eV and the total is summed in float64, as in v1.
- **Forces.** F = -dE/dr.

## Data, split and training

- **Pool.** Clean frames under the block rules.
- **Grouping.** By system key (path with the `_Hterm` / `_frame` suffix removed).
- **Subsample.** About 3 000 frames (about 540k atoms), drawn by system.
- **PE frames.** All of v1's PE frames are included. v1's PE val and test systems stay in
  v2's val and test splits.
- **Split.** 80/10/10 by system.
- **Training.** Same objective and optimiser as v1:
  - 20 · per-atom E MSE + F MSE;
  - AdamW, cosine schedule over wall clock, EMA 0.99;
  - float32 training, float64 evaluation.
  - Budget: 120 min.
- **MACE.** Same settings as the v1 baseline, on the identical extxyz split. It is trained
  concurrently; the GPU sharing is disclosed, as in v1.

## Code layout

- **`topology.py`.** New `build_topology_general`, which builds:
  - block = carbon, with its type;
  - `block_atoms` padded to 4 slots (C plus up to 3 H), with a mask;
  - C-C block edges (1-hop and 2-hop);
  - the variant table.

  The v1 PE path stays as it is, so v1 checkpoints still load.
- **`model.py`.** v2 goes in a new module, `model_v2.py`, as `PolyFrameNetV2`. It reuses
  `mlp`, `gaussians`, `bessel_envelope` and the moment code. v1 is untouched.
- **`batch.py`.** Collate gains the variant table and per-block type.
- **`data.py`.** Gains a hydrocarbon loader and a subsampler.
- **`train.py`.** A `--model v2` flag.
- **`calculator.py`.** Picks the model by checkpoint.
- **Builders.** Gain a polystyrene chain (or a melt from OPoly26 frames tiled) for MD.

## Gates (tests/test_gates_v2.py)

Each gate runs on these test molecules:
- an aromatic oligomer (styrene trimer);
- a branched alkane (with a quaternary C);
- a diene;
- methane;
- a periodic PE melt.

The gates:
- rotation + translation, reflection, permutation of all atoms, wrap: |ΔE| < 1e-9;
- finite-difference forces: < 1e-8 eV/Å;
- cutoff smoothness;
- batch equals single;
- calculator equals model;
- float32 vs float64.

Two further tests:
- **A true oracle (closes the v1 gap).**
  - Stage 1 claims that its variant set is closed under every symmetry.
  - The test applies a random relabelling plus a rotation plus a mirror, rebuilds the
    variant table, and checks the per-block multiset of variant features is identical
    before and after.
  - That checks the set itself, not just E.
- **Negative control.** Drop one variant per block and assert the rotation or permutation
  gate fails.

## Risks

- **Cost.** Mean variants per atom is about 10, against 4 in v1. Stage 3 and Stage 4 cost
  scales with it. If speed falls below MACE at 1k atoms, average B before the products
  for the cheap channels and keep products only for a few channels.
- **Data imbalance.** Aromatic carbons dominate. PE accuracy may drop. That is measured
  by success criterion 3.
- **Time.** If training or the gates slip, cut the polystyrene MD before the PE
  comparison.
