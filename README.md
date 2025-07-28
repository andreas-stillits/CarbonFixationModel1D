# CarbonFixationModel1D
Functionality for data analysis and generation of plots published in the article:

Title: Mapping Carbon Fixation to Two Effective Parameters: a framwork towards data-informed model simplifications 
Journal: ...
DOI: ...
Authors: A. Stillits, T.E. Knudsen, A. Trusina

Abstract: ... 

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
            - Data from Knauer et al. 2022 where we have filtered for $\tau,\gamma > 0.5$ aiming at generating figure 5 (S8) that explores amphistomatous boundary conditions.
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
