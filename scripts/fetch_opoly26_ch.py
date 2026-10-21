"""Pull every C/H-only configuration out of OPoly26 (ColabFit mirror on Hugging Face).

The ColabFit mirror is public and ungated, unlike facebook/OMol25. DuckDB reads the
remote parquet over HTTP range requests, so filtering on `element_filter` touches only
that column plus the row groups that contain matches; the 39 GB train split is never
downloaded whole.

Output: one .npz per source shard in --out, each holding ragged arrays flattened with
offsets (numbers, positions, forces) plus per-frame energy, multiplicity and source name.
Downstream (frame_potential.data) applies the polyethylene topology filter.

Units are as ColabFit stores them: eV and eV/Angstrom, positions in Angstrom.
"""

import argparse
import json
import pathlib
import sys

import duckdb
import numpy as np

BASE = "https://huggingface.co/datasets/colabfit/{ds}/resolve/main/co/co_{i}.parquet"
SHARDS = {"OPoly26-val": 1, "OPoly26-train": 16}


def fetch(ds: str, i: int, out: pathlib.Path) -> None:
    dest = out / f"{ds}_{i}.npz"
    if dest.exists():
        print(f"skip {dest.name} (exists)")
        return
    url = BASE.format(ds=ds, i=i)
    con = duckdb.connect()
    rows = con.sql(
        f"""
        select configuration_id, names, multiplicity, energy, atomic_numbers,
               positions, atomic_forces
        from '{url}'
        where element_filter = ',C,H,'
        """
    ).fetchall()
    numbers, positions, forces, offsets = [], [], [], [0]
    energy, mult, names, ids = [], [], [], []
    for cid, nm, m, e, z, r, f in rows:
        numbers.append(np.asarray(z, dtype=np.int64))
        positions.append(np.asarray(r, dtype=np.float64))
        forces.append(np.asarray(f, dtype=np.float64))
        offsets.append(offsets[-1] + len(z))
        energy.append(e)
        mult.append(m if m is not None else -1)
        names.append(nm[0] if nm else "")
        ids.append(cid)
    np.savez_compressed(
        dest,
        numbers=np.concatenate(numbers) if numbers else np.zeros(0, np.int64),
        positions=np.concatenate(positions) if positions else np.zeros((0, 3)),
        forces=np.concatenate(forces) if forces else np.zeros((0, 3)),
        offsets=np.asarray(offsets, dtype=np.int64),
        energy=np.asarray(energy, dtype=np.float64),
        multiplicity=np.asarray(mult, dtype=np.int64),
        names=np.asarray(names),
        ids=np.asarray(ids),
    )
    print(f"{dest.name}: {len(rows)} C/H frames", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--only", choices=list(SHARDS), default=None)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for ds, n in SHARDS.items():
        if a.only and ds != a.only:
            continue
        for i in range(n):
            try:
                fetch(ds, i, a.out)
            except Exception as exc:  # keep going; a rerun skips finished shards
                print(f"FAILED {ds} {i}: {exc}", file=sys.stderr, flush=True)
    (a.out / "SOURCE.json").write_text(
        json.dumps({"source": "huggingface.co/datasets/colabfit (OPoly26)", "filter": ",C,H,"})
    )


if __name__ == "__main__":
    main()
