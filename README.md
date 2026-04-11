# AeroGUI — Atmospheric Particle Analysis Platform

A Python desktop application for aerosol measurement data analysis,  
built with PyQt5 + Matplotlib.

---

## Installation

```bash
# Clone / download the project, then:
pip install -r requirements.txt
```

---

## Launch

```bash
python main.py
```

---

## Tab overview

### 1 · Data
- **Load** CSV or Excel files (File → Open, or button)
- **Assign columns**: pick the time column and the count columns (one per diameter bin)
- **2D heatmap**: x = scan index / time, y = diameter, colour = counts
- **1D time series**: sum counts over a diameter range and plot vs. time
- **Export subset**: save any selection of columns to CSV or Excel

### 2 · Measurement Model
- Configure DMA geometry (radii, length) and flow rates
- Set scan voltage range and number of channels
- Set CPC parameters (D₅₀, σ)
- Choose model type (triangular transfer function + sigmoid/step CPC efficiency)
- **Compute kernel** → visualise transfer matrix
- **Save** kernel + parameters to CSV or Excel

### 3 · Inversion
- Choose an inversion method:
  - **Tikhonov regularisation** (zeroth / first / second order, tunable λ)
  - **Truncated SVD** (truncate at k singular values)
  - **NNLS** (non-negative least squares, no regularisation)
  - **EM** (Expectation-Maximisation, Poisson noise model)
- Invert a single scan or all scans at once
- **L-curve** tool to select the optimal Tikhonov λ
- Save retrieved dN/dlogDp to CSV

### 4 · Parameter Estimation
- Define which instrument parameters to fit (initial value + bounds)
- Supported methods:
  - **Least squares** (L-BFGS-B)
  - **Maximum likelihood** (Poisson)
  - **Differential evolution** (global, requires bounds)
- Compare observed vs. fitted counts in the plot panel

### 5 · Simulation
- Set up a log-normal initial size distribution
- Toggle physical processes: coagulation, condensation, nucleation, deposition
- Run the GDE solver (runs in a background thread — GUI stays responsive)
- Visualise evolution as a 2D heatmap and total N(t) time series
- Save the full simulation matrix to CSV

---

## Extending the code

| What you want to add | Where to edit |
|---|---|
| New inversion method | `modules/inversion.py` → subclass `InversionMethod`, add to `INVERSION_METHODS` |
| New estimation method | `modules/param_estimation.py` → subclass `ParameterEstimator`, add to `ESTIMATION_METHODS` |
| New model variant | `modules/measurement_model.py` → add key to `MeasurementModel.MODELS` |
| Julia integration | Call Julia via `juliacall` or `subprocess` in the relevant module, keeping the same Python interface |
| New data manipulation | `modules/data_model.py` → add a method, wire it up in `ui/tab_data.py` |

---

## Julia interoperability

The Python module interfaces are designed to be thin wrappers.  
To use your existing Julia code, install **juliacall** (`pip install juliacall`) and replace
the body of e.g. `EMInversion.solve()` with:

```python
from juliacall import Main as jl
jl.include("path/to/your_inversion.jl")
result = jl.your_em_function(A, counts)
```

---

## File structure

```
aero_gui/
├── main.py                  # entry point
├── requirements.txt
├── modules/
│   ├── data_model.py        # AerosolDataset class
│   ├── measurement_model.py # DMA/CPC forward model
│   ├── inversion.py         # inversion methods
│   ├── param_estimation.py  # parameter estimators
│   └── simulation.py        # GDE simulator
└── ui/
    ├── main_window.py       # top-level window + tab host
    ├── tab_data.py          # Tab 1
    ├── tab_model.py         # Tab 2
    ├── tab_inversion.py     # Tab 3
    ├── tab_estimation.py    # Tab 4
    └── tab_simulation.py    # Tab 5
```
