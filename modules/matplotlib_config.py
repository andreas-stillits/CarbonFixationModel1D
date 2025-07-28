# functionality for setting up often used matplotlib settings

import matplotlib as mpl
from matplotlib.pyplot import get_cmap
import numpy as np

def set_standard_layout():
    ''' 
    Function to set up font settings and return colors used to denote regions I, II, III in figures such as 2D, 2F
    '''
    mpl.rcParams['mathtext.fontset'] = 'stix'  # or 'dejavusans', 'cm', 'custom'
    mpl.rcParams['font.family'] = 'STIXGeneral'  # Matches STIX math font
    # set tick font size
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    # set default fontsize
    mpl.rcParams['font.size'] = 14
    # COLORS
    # I : stomata
    colorI = hex2rgb('71A8C4')
    # II : absorption capacity
    colorII = hex2rgb('EACEBD')
    # III : IAS
    colorIII = hex2rgb('F8A17B')
    return colorI, colorII, colorIII   

# function to generate a list of N colors from a colormap
def generate_colors(colormap_name, N):
    cmap = get_cmap(colormap_name)
    return [cmap(i / (N - 1)) for i in range(N)]

# function to convert hex color to rgb tuple
def hex2rgb(hex):
    rbg = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return tuple([c/255 for c in rbg])

# function to create a custom colormap with specified min, max, and resolution (used as heatmap in figures such as 3C)
def my_cmap(min, max, res):
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