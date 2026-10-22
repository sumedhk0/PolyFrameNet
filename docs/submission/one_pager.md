# PolyFrameNet: a polymer potential that knows what a polymer is

**Code:** https://github.com/sumedhk0/PolyFrameNet (MIT) · **Data:** OPoly26 (ωB97M-V/def2-TZVPD)

## Problem

Polymer simulation needs quantum accuracy at the scale of tens of thousands of atoms.
General machine-learned potentials such as MACE get the accuracy by carrying equivariant
tensor features on every edge. That is expensive, and on a 12 GB GPU MACE runs out of
memory near 8,000 atoms. Polymers have exploitable structure: a small vocabulary of
repeating units, a covalent graph that never changes, and a backbone that defines local
directions. Generic models discard all three.

## Approach

PolyFrameNet reads the bond graph once and cuts it into **rigid blocks**:
- each ring system is one block;
- every other heavy atom is a block;
- terminal atoms join their neighbour's block.

The rule uses graph structure only, so it is unique on any fragment. Each block gets
local orthonormal **frames** built from its bonded atoms. Vectors expressed in a frame
are rotation-invariant, so every network is a plain MLP: no tensor products.

The usual failure of frame methods is choosing the frame. A frame picked by "nearest
atom" jumps discontinuously and breaks energy conservation. We never pick: every stage
averages over all frames drawn from **Weisfeiler–Lehman colour classes** of the bond
graph. That set is closed under every rotation, reflection and relabelling, so the
energy is *exactly* O(3)- and permutation-invariant. Forces are exact gradients.

Pipeline:
1. shared block encoder;
2. 3 message-passing layers on the sparse block graph;
3. a learned valence prior;
4. one 4.5 Å many-body layer;
5. latent-charge electrostatics (Ewald) and a learned dispersion tail;
6. charge/spin conditioning.

The architecture handles any element and any non-reactive polymer; only the training
data changes.

## Results

All models use identical system-level splits and the same reference energies, on an RTX
4080 Laptop GPU.

![results](../figures/submission.png)

| Force MAE (meV/Å) | Hydrocarbon test (782) | PE test (77) | Train time |
|---|---|---|---|
| **PolyFrameNet v2** | 50.4 | 24.1 | 77.5 min |
| MACE | 27.0 | 16.5 | ~150 min |
| Allegro | ALLEGRO_TEST | ALLEGRO_PE | 90 min |
| DeepMD | 211 | 175 | 63 min |

- **v1 (polyethylene only).** It ties MACE: 13.71 vs 13.73 meV/Å force MAE.
  - It is 2.3× faster at 1k atoms.
  - At 7.8k atoms MACE spills past the 12 GB card and is 87× slower; v1 uses 1.1 GB.
  - At 26k atoms v1 runs in 3.5 GB, where MACE's peak allocation is 41.6 GB.
- **v2 (general architecture, trained on 3,429 hydrocarbon frames)** currently trails
  MACE in accuracy. Its training was cut at 77.5 min, after a run lost to one corrupt
  training geometry (below). Speed: V2_SPEED.
- **21 correctness gates pass on an untrained model:**
  - rotation, reflection, permutation and wrap to 1e-9 eV;
  - finite-difference forces;
  - a true variant-set oracle.

  They run on *ether, amide, ester, C–Cl, CF₄, pyridine, thiophene, nitrile and ion*
  fragments the model never trained on, so the symmetry guarantees hold for any
  chemistry.
- **Block registry.** For any new polymer it flags each building block as seen,
  similar or novel relative to the training data, so users know when not to trust the
  model.

## What we learned (honest notes)

- **A frame can go singular.** One training frame had an H at exactly 180° to a C–C
  bond. That made one frame singular and produced ~10⁴ eV/Å forces once per epoch. We
  found and removed it. The principled fix, sin²-weighted frame averaging, is next.
- **A benchmark artifact.** MACE's noisy NVE energy in float32 is a rounding artifact
  of summing -1037 eV/atom offsets in float32. Our model stores the offsets as integers
  and sums in float64.

**Next:**
- train on all of OPoly26 on a cluster;
- stress output for NPT density and glass transition;
- CUDA graphs to remove launch overhead.
