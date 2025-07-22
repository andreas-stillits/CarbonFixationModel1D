import matplotlib as mpl
from matplotlib.pyplot import get_cmap

def set_standard_layout():
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

def generate_colors(colormap_name, N):
    cmap = get_cmap(colormap_name)
    return [cmap(i / (N - 1)) for i in range(N)]

def hex2rgb(hex):
    rbg = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return tuple([c/255 for c in rbg])

