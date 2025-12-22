# CarbonFixationModel1D
Functionality for data analysis and generation of plots published in the article:

Title: Mapping Carbon Fixation to Two Effective Parameters: a framwork towards data-informed model simplifications 

Journal: ...

DOI: ...

Authors: A. Stillits, T.E. Knudsen, A. Trusina

Abstract: 

Identifying the steps limiting net $CO_2$ assimilation rate in plant leaves, is essential for improving crop yield and resilience. However, disentangling the combined effects of multiple traits remains a major challenge. Biophysical models can resolve how multiple traits combine to control the rate-limiting steps, but there is no systematic way to identify the appropriate level of spatial and mechanistic resolution - ranging from simple resistance models to detailed anatomical simulations. Here, we propose that the necessary level of model resolution is species-specific. We apply a minimal reaction-diffusion model and reduce $CO_2$ fixation in leaves to two key parameters. These parameters comprise a compact phase space in which three rate-limiting regimes emerge naturally: stomatal uptake, intercellular diffusion, and subcellular processes. Mapping diverse plant species into this phase space reveals dominant co-limitations by stomatal and subcellular processes. The same parameter space provides quantitative criteria for model selection. Our results suggest that a simple model parameterized with bulk leaf properties will suffice to relate assimilation rate and leaf physiology for the vast majority of plants. However, a precise estimation of bulk properties from 3D anatomical data and models will be crucial. This framework offers a scalable path for interpreting complex trait data and guiding model selection.

FILE STRUCTURE

- data (all data used for tables and figures)
    - raw_data (Datasets provided in previous studies)
        - gm_dataset_Knauer_et_al_2022.csv 
            - Meta-dataset assembled by Knauer et al. 2022 (DOI: https://doi.org/10.1111/nph.18363)
            - 1883 datapoints across 617 species and 13 plant functional types
            - Gas exchange measurements as well as physiological data
        - dehydrated_dataset_Momayyezi_et_al_2022.csv
            - Dataset provided by Momayyezi et al. 2022 (DOI: https://doi.org/10.1111/pce.14287)
            - Study of 11 accessions of Juglans Regia (walnut) in different geographical locations and climates under drought conditions
            - Gas exchange measurements as well as physiological data
            - The table as presented here is constructed from Momayyezi et al. 2022, Table 1 (main text) and Table S2 (supplementary material)
        - watered_dataset_Momayyezi_et_al_2022.csv
            - Dataset provided by Momayyezi et al. 2022 (DOI: https://doi.org/10.1111/pce.14287)
            - Study of 11 accessions of Juglans Regia (walnut) in different geographical locations and climates under well-watered conditions
            - Gas exchange measurements as well as physiological data
            - The table as presented here is constructed from Momayyezi et al. 2022, Table 1 (main text) and Table S2 (supplementary material)
    - saved_data (Datasets that have been manipulated from data in raw_data or obtain from parameter search)
        - Knauer2022_filtered.csv
            - Data from Knauer et al. 2022 where we have extracted only the data that allows estimation of effective parameters $(\tau, \gamma)$
        - Knauer2022_hypostomatous.csv
            - Data from Knauer et al. 2022 where we have estimated $(\tau,\gamma)$ assuming hypostomatous boundary conditions (See supplementary S7, S8)
        - Knauer2022_most_ias_limited.csv
            - Data from Knauer et al. 2022 where we have filtered for $\tau,\gamma > 0.5$ aiming at generating figure 5 (S8) that explores amphistomatous boundary conditions and its consequences for the most extreme datapoints (in the sense of IAS limitation to net assimilation rate).
        - Momayyezi2022_dehydrated.csv
            - Data from Momayyezi et al. 2022 where we have estimated $(\tau,\gamma)$ for drought conditions
        - Momayyezi2022_watered.csv
            - Data from Momayyezi et al. 2022 where we have estimated $(\tau,\gamma)$ for well-watered conditions
        - sensitivities.txt
            - Data reflecting the sensitivity ($\eta(\tau,\gamma)$) heatmap in e.g. figure 3C
    - scripts
        - datareader_Knauer2022.ipynb
            - Jupyter notebook for analysis of data provided by Knauer et al. 2022.
            - Contains extensive information about the decisions leading to the extracted and subsequently presented data
        - datareader_Momayyezi2022.ipynb
            - Jupyter notebook for analysis of data provided by Momayyezi et al. 2022.
        - parameter_search_sensitivity.ipynb
            - Jupyter notebook containing the algorithm and parameters used for generating the sensitivity parameter search (sensitivities.txt)
        - spatial_embedding_error.ipynb
            - Jupyter notebook exploring the differences in $g_m^*$ definitions in a serial resistance model (0D) versus a continuous parallel resistance model (1D)
- figures (scripts and .svg files used to generate the presented plots. NB: later editing in Adobe Illustrator did follow)
    - scripts
        - Jupyter notebooks for generating blueprints for figures 2A, 2B, 2D, 2E, 2F, 3A, 3B, 3C, 4 (S7), and 5 (S8)
    - vectorgraphics
        - .svg files of the blueprints used for the presented figures. Later editing in Adobe Illustrator has followed
- modules (python files containing often used functionality)
    - estimator.py
        - module containing function for estimation and error propagation of quantities: $C_i, g_m^*, \tau, \gamma$. 
        - note that error propagation occurs through Monte Carlo simulations, hence estimates may vary slightly between executions
    - leaf_model.py
        - module containing the class 'Leaf' that encodes the leaf model and its solution.
        - Since the solution of the presented model exists analytically, there is no numerical approximation at play, only floating point precision
    - matplotlib_config.py
        - module containing often used settings for matplotlib (fonts, colors, colormaps)
- supplementary (material for exploring simplifying assumptions)
    - codebase 
        - lateral
            - This module explores the simplifying assumption of neglecting 3D lateral diffusion originating from discrete stomatal distribution.
            - comparison.py
                - compute the difference in predicted assimilation rate from including lateral diffusion (3D vs 1D)
            - create_mesh.py
                - create a .msh file for simulating reaction-diffusion of gaseous CO2 in a mesophyll plug served by one stomate
            - mpiscan.py
                - reproduce the sensitivity map over $(\tau, \gamma)$ for a 3D mesophyll model
            - solver.py
                - solver class for a 3D description of gaseous reaction-diffusion in a continuous mesophyll model.
            - stomata_analysis.ipynb
                - statistical analysis and probagation from measured physiology like stomatal density to geometric features like mesophyll plug aspect ratio.
        - nonlinear
            - mpiscan.py
                - reproduce the sensitivity map over $(\tau, \gamma)$ for a 1D mesophyll model with non-linear Rubisco kinetics
            - solver.py
                - sovler class for a 1D description of gaseous reaction-diffusion in a continuous mesophyll model with non-linear Rubisco kinetics.
        - steady
            - mpiscan.py
                - reproduce the sensitivity map over $(\tau, \gamma)$ for a 1D mesophyll model with general $D, K$ distributions
            - solver.py
                - solver class for a linear 1D model with general diffusivity $D$ and reactivity $K$ distributions
        - temporal
            - mpiscan.py
                - calculate the variation in assimilation rate prediction for non-steady scenarios when neglecting and accounting for spatial heterogeneity (spatial mean field vs spatial heterogeneity)
            - solver.py
                - solver class for non-steady integration of the 1D linear continuous mesophyll model.
        - utils
            - constants.py
                - centralization module for parameter choices
            - homogeneous.py
                - module for solving the 1D homogeneous and linear leaf model
            - mpiscan2d.py
                - helper module for basic parallelization of 2D parameter scans using MPI
            - paths.py
                - helper module for managing paths
            - plotfunctions.py
                - helper module for often used plotting utilities
            - profiles.py
                - helper module for typical diffusivity $D$, reactivity $K$ and oscillatory profiles in time and space.
    - figures (plots summarizing supplementary exploration results)
    - notebooks (Must be run with /supplementary/ as the working directory)
        - compare3DHetVsHom.ipynb
            - comparison plot between two 3D model solutions with and without spatial heterogeneity in $D, K$
        - sensitivity.ipynb
            - generating plots over sensitivity $\eta(\tau,\gamma)$ for nonlinear kinetics and exponential $D,K$ distributions
        - threedim.ipynb
            - display a .msh file in the GMSH GUI
            - generate comparison plot between 1D and 3D homogeneous model solutions
            - generate sensitivity plot as in main results but for a typical 3D geometry
        - transients.ipynb
            - generate temporal sensitivity scan over amplitude and frequency of an oscillatory driving of $C_a, g_s$ or $K$
        - visualize3DSolutions.ipynb
            - plot a 3D solution over a given mesh for inspection
    - shell scripts (scripts for saturating the supplementary simulation database)
        - crossdim_comparison_scanning.sh
            - MPI scan over 1D vs 3D model predictions of assimilation rates
        - exponential_scanning.sh
            - MPI scan over sensitivity $\eta(\tau,\gamma)$ for exponential $D,K$ distributions
        - nonlinear_scanning.sh
            - MPI scan over sensitivity $\eta(\tau,\gamma)$ for nonlinear Rubisco kinetics
        - temporal_scanning.sh
            - MPI scan over sensitivities for a range of amplitudes and frequencies of sinusoidal driving
        - threedim_scanning.sh
            - MPI scan over sensitivity $\eta(\tau,\gamma)$ for a typical 3D geometry
- environment.yml
    - Anaconda virtual environment file for reproducibility. 
    - If disregarding the supplementary exploration, one will only need standard scientific packages:
        - numpy
        - pandas
        - matplotlib
        - tqdm (optional nicety)

CONTACT

email: andreas.stillits@nbi.ku.dk

[![DOI](https://zenodo.org/badge/1023012859.svg)](https://doi.org/10.5281/zenodo.16541959) 

