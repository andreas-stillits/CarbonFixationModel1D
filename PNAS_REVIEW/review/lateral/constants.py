


from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Constants3D:
    mesophyll_thickness_min = 20.6
    mesophyll_thickness_mean = 238.7
    mesophyll_thickness_max = 809.0
    # min, mean, max in Liu et al 2017 in units of stomata per mm2
    stomatal_density_min = 8.93
    stomatal_density_mean = 189.6
    stomatal_density_max = 632.4
    # plug radii assuming odered arrangement of stomata
    plug_radius_min = 1000/(2*np.sqrt(stomatal_density_max)) # µm
    plug_radius_mean = 1000/(2*np.sqrt(stomatal_density_mean)) # µm
    plug_radius_max = 1000/(2*np.sqrt(stomatal_density_min)) # µm
    # aspect ratio of the plugs
    plug_aspect_ratio_min = plug_radius_min / mesophyll_thickness_max
    plug_aspect_ratio_mean = plug_radius_mean / mesophyll_thickness_mean
    plug_aspect_ratio_max = plug_radius_max / mesophyll_thickness_min

