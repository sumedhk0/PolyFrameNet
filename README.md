# Repository for PolyFrameNet, a polymer-specific MLIP for efficient, accurate MD simulation.

## Development setup

Keep the virtual environment outside OneDrive: syncing a venv's thousands of files is
slow and can lock files mid-install.

```powershell
uv venv --python 3.13 $HOME\venvs\PolyFrameNet
& $HOME\venvs\PolyFrameNet\Scripts\Activate.ps1
uv pip install -e ".[dev]"
```

This installs the CPU build of torch, which is all the correctness tests need. For GPU
training, swap in the CUDA build (uv picks the version matching the installed driver):

```powershell
uv pip install --reinstall torch --torch-backend=auto
```

## Checks

```powershell
pytest        # every test runs in float64, see tests/conftest.py
ruff check .
```
