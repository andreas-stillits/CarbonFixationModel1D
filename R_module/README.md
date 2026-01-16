# Minimal R interface for tau, gamma estimation

This folder provides a lightweight R analogue of the `python_module` for
estimating the parameters (tau, gamma) from a user's own data.

## Contents

- `module.R`: R functions implementing the same logic as
  `python_module/helpers.py` and `python_module/interface.py`:
  - `newton_solver()` – internal numeric root finder for the cellular conductance.
  - `estimate_Ci_values()` – estimate intercellular CO2 concentration `Ci`.
  - `estimate_gm_star_values()` – estimate mesophyll conductance at the
    compensation point `gm*`.
  - `vector_estimate_parameters()` – numeric entry point: vectors of
    `(gm*, gs, gias)` → vectors `tau`, `gamma`.
  - `dataframe_estimate_parameters()` – data.frame entry point with similar
    behaviour to the Python `pandas_estimate_parameters()`.

There is no full R package scaffolding here; just source the file or place it
on your R library path and use the functions directly.

## Usage

### 1. Numeric vectors ("numpy" style)

Given numeric vectors `gm_star`, `gs`, and `gias` (all same length, units
mol / m^2 / s):

```r
source("R_module/module.R")

res <- vector_estimate_parameters(
  mesophyll_conductances = gm_star,
  stomatal_conductances = gs,
  intercellular_airspace_conductances = gias
)

tau   <- res$tau
gamma <- res$gamma
```

Invalid entries (non-positive or `NA`) are returned as `NaN`.

### 2. Data frames ("pandas" style)

You can instead work with a `data.frame`, mirroring the Python
`pandas_estimate_parameters()`.

The function expects (after any column remapping):

- Always: `gs` (stomatal conductance), `gias` (IAS conductance)
- And **one** of the following options:
  - Option A: `gm_star`
  - Option B: `An`, `Ci`, `C_star`
  - Option C: `An`, `Ca`, `C_star`

Example with default column names:

```r
source("R_module/module.R")

res_df <- dataframe_estimate_parameters(df)
# res_df now has added numeric columns `tau` and `gamma`
```

If your column names differ, pass a `column_map` named list, where the names
are the expected keys ("gs", "gias", "gm_star", "An", "Ci", "Ca", "C_star") and
the values are the actual column names in your data frame:

```r
res_df <- dataframe_estimate_parameters(
  df,
  column_map = list(gs = "gs_CO2", gias = "ias_conductance")
)
```

The function returns a copy of the input data.frame with two extra columns:
`tau` and `gamma`.
