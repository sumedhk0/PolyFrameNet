"""Merge per-model eval JSONs (<kind>_<split>.json from eval_ase.py) into one table.
Usage: python scripts/eval_v2_all.py <eval dir>   -> prints markdown, writes all.json"""

import json
import pathlib
import sys

if __name__ == "__main__":
    d = pathlib.Path(sys.argv[1])
    rows: dict[str, dict] = {}
    for f in sorted(d.glob("*_*.json")):
        if f.name == "all.json":
            continue
        kind, split = f.stem.rsplit("_", 1)
        rows.setdefault(kind, {})[split] = json.loads(f.read_text())
    (d / "all.json").write_text(json.dumps(rows, indent=1))
    nan = float("nan")
    print("| model | test E MAE (meV/atom) | test F MAE (meV/A) | PE test E MAE | PE test F MAE |")
    print("|---|---|---|---|---|")
    for k, v in rows.items():
        t, p = v.get("test", {}), v.get("pe", {})
        print(f"| {k} | {t.get('e_mae_meV_atom', nan):.3f} | {t.get('f_mae_meV_A', nan):.2f} | "
              f"{p.get('e_mae_meV_atom', nan):.3f} | {p.get('f_mae_meV_A', nan):.2f} |")
