# COT-He Functional

**Designing explicit functionals for the charge density in terms of a potential**

A computational physics package implementing Connector Theory (COT) for cubic Helium (fcc He solid), developed as part of the research by Muhammed Hüseyin Güneş at LSI, École Polytechnique, Institut Polytechnique de Paris.

## Core Idea

Instead of solving the Kohn-Sham (KS) Schrödinger equation to get the electron charge density `n(r)`, this package approximates it directly as an explicit functional of the KS potential `v_KS(r)`, using model data from the Homogeneous Electron Gas (HEG) and Connector Theory (COT).

## Package Structure

```
cot_he_functional/
├── cot_functional/          # Main Python package
│   ├── config.py            # Material parameters & configuration
│   ├── grid.py              # Reciprocal/real space grid setup
│   ├── fourier.py           # FFT routines (ABINIT-compatible)
│   ├── lda.py               # LDA XC functionals (Perdew-Zunger)
│   ├── lindhard.py          # Lindhard function (real & reciprocal space)
│   ├── heg.py               # HEG model: Thomas-Fermi density, v_tilde
│   ├── connector.py         # Connector potential (all approximation types)
│   ├── ks_solver.py         # KS Hamiltonian construction & diagonalization
│   ├── observables.py       # Sphere integral, diagonal slices, MADE
│   └── io.py                # File I/O helpers
├── data/                    # Reference data (.dat files)
│   ├── equilibrium/         # a_l = 8.016 Bohr
│   ├── compressed/          # a_l < 8.016
│   ├── isolated/            # Isolated He
│   └── qe/                  # Quantum ESPRESSO outputs
├── scripts/                 # Command-line tools
├── notebooks/               # Analysis & figure notebooks
└── tests/                   # Unit tests
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from cot_functional.config import load_config
from cot_functional.grid import setup_grid
from cot_functional.connector import compute_connector
from cot_functional.heg import n_heg

# Load configuration for equilibrium He at Ecut=15 Ha
cfg = load_config(ecut=15, a_l=8.016, data_dir="./data/equilibrium")

# Setup grid (r-space, G-space, k-points)
grid = setup_grid(cfg)

# Build KS potential and compute connector
# ... see notebooks/ for full examples
```

## Connector Approximations

| Name       | Code key          | Description |
|------------|-------------------|-------------|
| LPA        | `'lpa'`           | Thomas-Fermi: n_h(V_ks) |
| LRA        | `'lra'`           | Linear response, no connector |
| COT1       | `'local'`         | Local connector: v0 = v(r) |
| COT1-av    | `'bilocal_rrc'`   | Self-consistent bilocal |
| COT1-alpha | `'bilocal_rrp'`   | Non-self-consistent bilocal |

## Requirements

- Python >= 3.8
- numpy, scipy, matplotlib
- chi_mhg (`pip install git+https://github.com/moegunes/chi_mhg.git`)
