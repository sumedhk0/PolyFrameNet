"""Fill baselines/deepmd/input.json for a run and train with the PyTorch backend, then
freeze. Run inside the bl_deepmd venv.
Usage: python baselines/deepmd/launch.py <deepmd_v2 dir> <run dir> <numb_steps>
DeepMD writes checkpoints to its working directory, so it is started with cwd=<run dir>."""

import json
import pathlib
import subprocess
import sys

if __name__ == "__main__":
    data, run, steps = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
    run.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((pathlib.Path(__file__).parent / "input.json").read_text())
    cfg.pop("_comment", None)
    cfg["training"]["training_data"]["systems"] = (data / "train.txt").read_text().split("\n")
    cfg["training"]["validation_data"]["systems"] = (data / "val.txt").read_text().split("\n")
    cfg["training"]["numb_steps"] = steps
    cfg["learning_rate"]["decay_steps"] = max(steps // 200, 100)
    (run / "input.json").write_text(json.dumps(cfg, indent=1))
    py = sys.executable
    subprocess.run([py, "-m", "deepmd", "--pt", "train", "input.json", "--skip-neighbor-stat"],
                   cwd=run, check=True)
    subprocess.run([py, "-m", "deepmd", "--pt", "freeze", "-o", "model.pth"], cwd=run, check=True)
