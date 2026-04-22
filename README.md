# AeroGUI — Atmospheric Particle Analysis Platform

A Python desktop application for aerosol measurement data analysis,
built with PyQt5 + Matplotlib.  All scientific algorithms are
self-contained in `modules/algo/` — no Julia runtime, no external
bridge, no extra dependencies beyond the six listed in `pyproject.toml`.

---

## Quick start with uv

[uv](https://docs.astral.sh/uv/) is the recommended way to manage
dependencies and run the project.

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and set up

```bash
git clone <your-repo-url>
cd aero_gui

# Create the virtual environment and install all dependencies
uv sync
```

blabla I am on my new branch

### 3. Run the application

```bash
uv run aero-gui
```

That's it. uv automatically uses the Python version pinned in
`.python-version` and the exact package versions locked in `uv.lock`.

---

## Daily workflow

| Task | Command |
|---|---|
| Run the app | `uv run aero-gui` |
| Sync dependencies after a `git pull` | `uv sync` |
| Add a new dependency | `uv add <package>` |
| Remove a dependency | `uv remove <package>` |
| Update all packages | `uv lock --upgrade` |
| Run a one-off script | `uv run python my_script.py` |
| Open a shell inside the venv | `source .venv/bin/activate` |

---

## Alternative: plain pip

If you prefer not to use uv:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## Tab overview

### 1 · Data
- Load CSV or Excel files (File → Open, or the button)
- Assign columns: pick the time column and the count columns (one per diameter bin)
- **2D heatmap**: x = scan index / time, y = diameter, colour = counts
- **1D time series**: sum counts over a diameter range and plot vs. time
- Export any column subset to CSV or Excel

### 2 · Measurement Model
- Configure DMA geometry, flow rates, voltage range, CPC parameters
- Choose model type (triangular transfer function + sigmoid/step CPC)
- Compute and visualise the kernel matrix A
- Save kernel + parameters to CSV or Excel

### 3 · Inversion
- Methods: Tikhonov regularisation (with L-curve), truncated SVD, NNLS, EM
- Invert a single scan or all scans at once
- Save retrieved dN/dlogDp to CSV

### 4 · Parameter Estimation
- Fixed Interval Kalman Smoother (FIKS) via pure-Python `modules/algo/kalman.py`
- Optional size-correlated process noise Q via `modules/algo/stochproc.py`
- Identity or persistence (r·I) state-transition models
- Smoothed trajectory, uncertainty bands, filtered-vs-smoothed plots
- Save results as `.npz` or `.csv`

### 5 · Simulation
- Configure an initial log-normal size distribution
- Toggle GDE mechanisms: coagulation, condensation, nucleation, wall losses
- Run the GDE solver in a background thread (GUI stays responsive)
- Visualise as 2D heatmap and total N(t) time series
- Save the full simulation matrix to CSV

---

## Project structure

```
aero_gui/
├── pyproject.toml          ← uv / pip project definition
├── .python-version         ← pinned Python version (uv)
├── .gitignore
├── uv.lock                 ← exact locked dependency versions (commit this)
├── requirements.txt        ← plain-pip fallback
├── main.py                 ← application entry point
├── modules/
│   ├── algo/               ← pure-Python translations of BAYROSOL + NMOpt
│   │   ├── kalman.py       ← EKF.jl:       KF, EKF, FIKS
│   │   ├── gde.py          ← AeroMec2.jl:  GDE solver
│   │   ├── measurement.py  ← AeroMeas.jl:  SMPS forward model & kernel
│   │   ├── stochproc.py    ← StochProc.jl: covariance tools
│   │   ├── optimisation.py ← NMOpt:        BFGS, L-BFGS-B
│   │   └── utils.py        ← utilsFun.jl:  softplus, logistic, quadrature
│   ├── data_model.py       ← AerosolDataset: load, transform, export
│   ├── measurement_model.py← DMA/CPC forward model (Python-native)
│   ├── inversion.py        ← Tikhonov, SVD, NNLS, EM inversions
│   └── simulation.py       ← GDE simulator (Python-native)
└── ui/
    ├── main_window.py      ← top-level window + tab host
    ├── tab_data.py         ← Tab 1: data loading & visualisation
    ├── tab_model.py        ← Tab 2: measurement model
    ├── tab_inversion.py    ← Tab 3: inversion
    ├── tab_estimation.py   ← Tab 4: parameter estimation (FIKS)
    └── tab_simulation.py   ← Tab 5: GDE simulation
```

---

## References

- Ozon et al. (2021), *Retrieval of process rate parameters in the GDE using Bayesian state estimation*, GMD — DOI: 10.5194/gmd-14-3715-2021
- Ozon et al. (2021), *Aerosol formation and growth rates from chamber experiments using Kalman smoothing*, ACP — DOI: 10.5194/acp-21-12595-2021
- Wiedensohler (1988), *An approximation of the bipolar charge distribution*, J. Aerosol Sci.
- Seinfeld & Pandis (2006), *Atmospheric Chemistry and Physics*
- Nocedal (1980), *Updating quasi-Newton matrices with limited storage*, Math. Comp.
- Moré (1994), *Line search algorithms with guaranteed sufficient decrease*
