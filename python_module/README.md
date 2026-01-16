# This module provides basic functionality for estimating (tau, gamma) from a readers own data.

## Contents:
- helpers.py: helper functions for estimating: 
    Substomatal concentration $C_i$, 
    Mesophyll conductance at the compensation point $g_m^*$,
    And the subcellular component of mesophyll conductance $g_c$

- interface.py: functions that accept either numpy arrays or a pandas data frame and returns $(\tau, \gamma)$ estimates.

- example_script.ipynb: a notebook showcasing how one might use these functions in a python script or notebook

- example_data.csv: data file for the purposes of demonstration


## Usage:

Estimation of $(\tau, \gamma)$ requires three quantities to be known - all in units of mol/m2/s:
- Stomatal conductance $g_s$
- IAS conductance $g_{IAS}$
- Mesophyll conductance at the compensation point $g_m^* = A_N/(C_i - C^*)$

The function 'numpy_estimate_parameters' will take these three as flat np.ndarray inputs and return arrays: taus, gammas

The function 'pandas_estimate_parameters' will do the same given a dataframe, provided that the following sets of columns exists:
- Option A: provide columns 'gs', 'gias', 'gm_star'
- Option B: provide columns 'gs', 'gias', 'An', 'Ci', 'C_star' ($g_m^*$ is infered from its definition)
- Option C: provide columns 'gs', 'gias', 'An', 'Ca', 'C_star' ($C_i$ is infered from $C_a - A_N/g_s$, and then $g_m^*$ from it)

The function will expect columns to be named as above by default but can take a dictionary 'column_map' as a keyword argument,
where the user may provide a mapping from the expected keys to those present in their dataset, e.g.:

column_map = {"gs": "gs_CO2", "gias": "ias_conductance"}

And so forth. The function will take a data frame and return an identical data frame with the two columns "tau", "gamma" added.

See 'example_script.ipynb'