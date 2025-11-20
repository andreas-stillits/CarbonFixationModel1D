"""

Module for plotting constants

"""

import matplotlib as mpl 
from matplotlib.pyplot import get_cmap
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PlottingConstants:

    CMAP = "viridis"

    def set_standard_layout(self) -> None:
        mpl.rcParams['mathtext.fontset'] = 'stix'  # or 'dejavusans', 'cm', 'custom'
        mpl.rcParams['font.family'] = 'STIXGeneral'  # Matches STIX math font
        # set tick font size
        mpl.rcParams['xtick.labelsize'] = 12
        mpl.rcParams['ytick.labelsize'] = 12
        # set default fontsize
        mpl.rcParams['font.size'] = 14

    def get_cmap(self, bounds: tuple[float, float, float]):
        min, max, res = bounds
        colors = ['#001261',
    '#02236C',
    '#023376',
    '#034481',
    '#06568C',
    '#156798',
    '#307DA6',
    '#4E92B4',
    '#71A8C4',
    '#94BED2',
    '#B3D1DF',
    '#D5E3E9'] #scicolor.get_cmap('oslo25').colors
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', colors, N=21).reversed()
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # force the first color entry to be 
        cmaplist[0] = 'white'
        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)

        # define the bins and normalize
        bounds = np.linspace(min, max, res)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        return cmap, norm

def standardize(self) -> None:
    pass