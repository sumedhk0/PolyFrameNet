#!/usr/bin/env bash
# All speed / NVE benchmarks on Linux (WSL2), one model and one measurement per process,
# nothing else on the GPU. Usage: bash scripts/run_benchmarks_wsl.sh <data dir> <out dir>
set -uo pipefail
D=${1:-$HOME/polyframenet_data}
OUT=${2:-$D/runs/bench_wsl}
mkdir -p "$OUT"
PFN=$HOME/venvs/pfn/bin/python
AL=$HOME/venvs/bl_allegro/bin/python
DP=$HOME/venvs/bl_deepmd/bin/python

run() {  # venv kind model structure task out
  echo "== $2 $4 $5"
  timeout 1800 "$1" scripts/bench_ase.py "$2" "$3" "$4" "$5" "$6" 2>&1 | tail -1
}

for r in 1 2 3; do
  run "$PFN" v2 "$D/runs/v2_main_c/best.pt" "$D/melts/melt_r$r.xyz" speed "$OUT/speed_v2_r$r.json"
  run "$PFN" v1 "$D/runs/pfn_o3/best.pt" "$D/melts/melt_r$r.xyz" speed "$OUT/speed_v1_r$r.json"
  run "$AL" allegro "$D/runs/allegro_v2/best.ckpt" "$D/melts/melt_r$r.xyz" speed "$OUT/speed_allegro_r$r.json"
  run "$DP" deepmd "$D/runs/deepmd_v2/model.pth" "$D/melts/melt_r$r.xyz" speed "$OUT/speed_deepmd_r$r.json"
done
for r in 1 2; do  # MACE at 26k atoms needs ~42 GB and pages for ~30 min (measured in v1)
  run "$PFN" mace "$D/runs/mace_v2/mace_v2.model" "$D/melts/melt_r$r.xyz" speed "$OUT/speed_mace_r$r.json"
done
run "$PFN" v2 "$D/runs/v2_main_c/best.pt" "$D/melts/melt_nve.xyz" nve "$OUT/nve_v2.json"
echo BENCH_DONE
