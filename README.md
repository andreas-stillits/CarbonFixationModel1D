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


CONTACT

email: andreas.stillits@nbi.ku.dk

[![DOI](https://zenodo.org/badge/1023012859.svg)](https://doi.org/10.5281/zenodo.16541959) 

